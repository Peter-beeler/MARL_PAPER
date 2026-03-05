"""
grpo_textgame.py — Unified GRPO training for text and compound action modes.

Usage:
    # Text mode (two-stage thinking → action word)
    python grpo_textgame.py --action_mode text --sanity_check

    # Compound mode (single-stage thinking + JSON helper call)
    python grpo_textgame.py --action_mode compound --sanity_check

    # Multi-GPU training
    accelerate launch grpo_textgame.py --action_mode text --use_accelerate
"""

import os
# Must happen BEFORE any torch imports
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import torch
# Force the distributed backend to skip specialized hardware checks
if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
import sys
import copy
import json
import logging
import shutil
import signal
import warnings
import time

import numpy as np
import wandb
from transformers import set_seed

# ── Path setup ──
sys.path.insert(0, os.path.dirname(__file__))

from env_move import CleanupEnvMove, Config as EnvConfigMove

from utils.config import GRPOConfig
from utils.model_setup import (
    setup_accelerator, load_tokenizer, load_base_model,
    setup_model_for_training, setup_optimizer_and_scheduler,
    AllowOnlyActionWords, log_cuda_memory,
)
from utils.eval import _generate_eval_states, evaluate, visualize_rollout
from utils.rollout import run_episode, run_parallel_episodes, log_episode_to_file
from utils.loss import compute_advantages, create_minibatch_iterator, flatten_trajectories, compute_loss_on_samples
from utils.args import parse_args

# ── Logging ──
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Main Trainer Class
# ────────────────────────────────────────────────────────────────

class CleanupGameGRPO:
    """
    GRPO trainer for cleanup game — supports text mode and compound mode
    via config.action_mode.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)

        # ── Accelerator ──
        self.accelerator, self.device = setup_accelerator(config)

        # Detect DeepSpeed so we can adjust operations accordingly
        # (gradient clipping is handled by DS engine, etc.)
        self.use_deepspeed = (
            self.accelerator is not None and
            getattr(getattr(self.accelerator, 'state', None), 'deepspeed_plugin', None) is not None
        )

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Loading model: {config.model_name}")
            logger.info(f"Action mode: {config.action_mode}")

        # ── Tokenizer ──
        self.tokenizer = load_tokenizer(config)

        # ── Mode-specific setup ──
        # === TODO: update action_words to match the low-level actions in your game env ===
        # === TODO: update helper_functions to match the high-level actions in observation.py ===
        if config.action_mode == "text":
            self.action_words = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']
            self.action_logits_processor = AllowOnlyActionWords(self.tokenizer, self.action_words)
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info(f"Action words: {self.action_words}")
                logger.info(f"Two-stage generation: {config.use_two_stage}")
                if config.use_two_stage:
                    logger.info(f"Thinking tokens: {config.thinking_tokens}, Action tokens: {config.action_tokens}")
                    logger.info(f"Log probability mode: {config.logprob_mode}")
        else:  # compound
            self.action_words = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']
            self.helper_functions = ['move_to', 'clean_at', 'eat_at', 'random_explore']
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info(f"Helper functions: {self.helper_functions}")
                logger.info(
                    f"Max new tokens: {config.thinking_tokens + config.action_tokens} "
                    f"(thinking={config.thinking_tokens} + action={config.action_tokens})"
                )

        # ── Model ──
        base_model = load_base_model(config)
        self.model = setup_model_for_training(base_model, config, self.accelerator, self.device)

        # ── Old model (PPO-style, frozen during inner epochs) ──
        self.old_model = None
        self.ref_model = None
        self.ref_on_cpu = False
        self.ref_device = None

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info("GRPO with inner epochs: Old model will be created on first group")

        # ── Optimizer & Scheduler ──
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(self.model, config)

        # ── Accelerate prepare ──
        if self.accelerator is not None and not (config.use_8bit or config.use_4bit):
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            logger.info(f"[GPU{self.accelerator.process_index}] Model prepared")
        elif config.use_8bit or config.use_4bit:
            self.device = next(self.model.parameters()).device

        # ── Environment config ──
        self.env_config = EnvConfigMove(
            n_agents=config.num_agents,
            max_steps=config.max_env_steps,
            seed=config.seed,
            eat_reward=config.eat_reward,
            clean_reward=config.clean_reward,
        )

        # ── Training statistics ──
        self.episode_rewards = []
        self.episode_steps = []
        self.training_step = 0

        # ── Wandb ──
        if config.use_wandb and (self.accelerator is None or self.accelerator.is_main_process):
            wandb_config = {
                "model_name": config.model_name,
                "action_mode": config.action_mode,
                "num_agents": config.num_agents,
                "num_episodes": config.num_episodes,
                "episodes_per_update": config.episodes_per_update,
                "learning_rate": config.learning_rate,
                "loss_type": config.loss_type,
                "gamma": config.gamma,
                "epsilon": config.epsilon,
                "max_grad_norm": config.max_grad_norm,
                "temperature": config.temperature,
                "thinking_tokens": config.thinking_tokens,
                "action_tokens": config.action_tokens,
                "use_two_stage": config.use_two_stage if config.action_mode == "text" else False,
                "logprob_mode": config.logprob_mode if config.action_mode == "text" else "N/A",
                "use_lora": config.use_lora,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "max_env_steps": config.max_env_steps,
                "seed": config.seed,
            }
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=wandb_config,
                reinit=True
            )
            logger.info(f"Initialized wandb project: {config.wandb_project}")

        # ── Pre-generate fixed evaluation states ──
        self.eval_states = _generate_eval_states(self.env_config, config, num_states=20)
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Generated {len(self.eval_states)} fixed evaluation states")

        # ── Signal handlers for graceful shutdown ──
        self._interrupted = False
        self._setup_signal_handlers()

    # ────────────────────────────────────
    # Model management
    # ────────────────────────────────────

    def _setup_signal_handlers(self):
        """Register SIGTERM/SIGUSR1 handlers for graceful SLURM shutdown."""
        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info(f"\n⚠ Received {sig_name} — will save checkpoint and exit after current step")
            self._interrupted = True

        # SIGTERM: SLURM sends this before SIGKILL (grace period, typically 30-120s)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGUSR1: some SLURM configs use this for preemption warning
        try:
            signal.signal(signal.SIGUSR1, _handle_signal)
        except (OSError, ValueError):
            pass  # SIGUSR1 not available on all platforms

    def _save_model(self, checkpoint_name: str, episode: int = None):
        """Save model checkpoint atomically to output_dir/checkpoint_name.

        Writes to a temporary directory first, then renames. If interrupted
        mid-write, the previous checkpoint remains intact.
        """
        accelerator = self.accelerator
        config = self.config
        if accelerator is not None and not accelerator.is_main_process:
            return

        checkpoint_path = os.path.join(config.output_dir, checkpoint_name)
        tmp_path = checkpoint_path + "_tmp"
        backup_path = checkpoint_path + "_backup"

        try:
            # Clean up any leftover tmp dir from a previous failed save
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            os.makedirs(tmp_path, exist_ok=True)

            # Write model + tokenizer to tmp dir
            if self.use_deepspeed:
                unwrapped = accelerator.unwrap_model(self.model)
                state_dict = accelerator.get_state_dict(self.model)
                unwrapped.save_pretrained(
                    tmp_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict,
                )
            elif accelerator is not None:
                unwrapped = accelerator.unwrap_model(self.model)
                unwrapped.save_pretrained(tmp_path)
            else:
                self.model.save_pretrained(tmp_path)
            self.tokenizer.save_pretrained(tmp_path)

            # Atomic swap: old → backup, tmp → target, remove backup
            if os.path.exists(checkpoint_path):
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                os.rename(checkpoint_path, backup_path)
            os.rename(tmp_path, checkpoint_path)
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)

            logger.info(f"  Saved checkpoint → {checkpoint_path}")
        except Exception as e:
            # Restore from backup if the swap failed
            if not os.path.exists(checkpoint_path) and os.path.exists(backup_path):
                os.rename(backup_path, checkpoint_path)
                logger.info(f"  Restored previous checkpoint from backup")
            logger.warning(f"  Failed to save model: {e}")

    def _save_training_state(self, episode, group_num, best_eval_reward):
        """Save optimizer, scheduler, and training metadata for resuming.

        Writes atomically alongside the latest checkpoint.
        """
        accelerator = self.accelerator
        config = self.config
        if accelerator is not None and not accelerator.is_main_process:
            return

        state_path = os.path.join(config.output_dir, "training_state")
        tmp_path = state_path + "_tmp"

        try:
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            os.makedirs(tmp_path, exist_ok=True)

            # Save optimizer & scheduler state
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(tmp_path, "optimizer.pt"),
            )
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(tmp_path, "scheduler.pt"),
            )

            # Save training metadata
            metadata = {
                "episode": episode,
                "group_num": group_num,
                "training_step": self.training_step,
                "best_eval_reward": best_eval_reward,
                "episode_rewards": self.episode_rewards,
                "episode_steps": self.episode_steps,
            }
            with open(os.path.join(tmp_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Atomic swap
            backup_path = state_path + "_backup"
            if os.path.exists(state_path):
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                os.rename(state_path, backup_path)
            os.rename(tmp_path, state_path)
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)

            logger.info(f"  Saved training state → {state_path}")
        except Exception as e:
            if not os.path.exists(state_path) and os.path.exists(state_path + "_backup"):
                os.rename(state_path + "_backup", state_path)
            logger.warning(f"  Failed to save training state: {e}")

    def load_training_state(self, resume_dir):
        """Load optimizer, scheduler, and training metadata from a checkpoint.

        Args:
            resume_dir: Path to the output_dir that contains training_state/
                        and a model checkpoint.

        Returns:
            metadata dict with episode, group_num, training_step, best_eval_reward.
        """
        state_path = os.path.join(resume_dir, "training_state")
        if not os.path.exists(state_path):
            logger.warning(f"No training_state found in {resume_dir}, starting fresh")
            return None

        is_main = self.accelerator is None or self.accelerator.is_main_process

        # Load metadata
        metadata_file = os.path.join(state_path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load optimizer state
        opt_file = os.path.join(state_path, "optimizer.pt")
        if os.path.exists(opt_file):
            opt_state = torch.load(opt_file, map_location="cpu")
            self.optimizer.load_state_dict(opt_state)
            if is_main:
                logger.info(f"  Loaded optimizer state from {opt_file}")

        # Load scheduler state
        sched_file = os.path.join(state_path, "scheduler.pt")
        if os.path.exists(sched_file):
            sched_state = torch.load(sched_file, map_location="cpu")
            self.scheduler.load_state_dict(sched_state)
            if is_main:
                logger.info(f"  Loaded scheduler state from {sched_file}")

        # Restore training counters
        self.training_step = metadata.get("training_step", 0)
        self.episode_rewards = metadata.get("episode_rewards", [])
        self.episode_steps = metadata.get("episode_steps", [])

        if is_main:
            logger.info(
                f"  Resumed from episode={metadata['episode']}, "
                f"group={metadata['group_num']}, step={self.training_step}, "
                f"best_eval_reward={metadata['best_eval_reward']:.2f}"
            )

        return metadata

    def update_old_model(self):
        """Copy current model weights to old model (θ_old ← θ)."""
        # With DeepSpeed the model is wrapped in a DeepSpeedEngine which
        # contains unpicklable ProcessGroup objects.  Always operate on
        # the unwrapped model for deepcopy / state_dict operations.
        if self.accelerator is not None:
            unwrapped = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped = self.model

        if self.old_model is None:
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Creating old model (first group)...")
            self.old_model = copy.deepcopy(unwrapped)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("  ✓ Old model created and frozen")
        else:
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Updating old model with current weights...")
            self.old_model.load_state_dict(unwrapped.state_dict())
            self.old_model.eval()
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("  ✓ Old model updated")

    # ────────────────────────────────────
    # Delegation to utils
    # ────────────────────────────────────

    def evaluate(self, num_episodes: int = 20, current_episode: int = None):
        """Evaluate current policy. Delegates to utils.eval.evaluate."""
        return evaluate(self, num_episodes=num_episodes, current_episode=current_episode)

    def visualize_rollout(self, use_ref_model: bool = False, save_to_file=None):
        """Visualize a single rollout. Delegates to utils.eval.visualize_rollout."""
        return visualize_rollout(self, use_ref_model=use_ref_model, save_to_file=save_to_file)

    # ────────────────────────────────────
    # Training loop
    # ────────────────────────────────────

    def train(self, resume_metadata=None):
        """Main training loop.

        Args:
            resume_metadata: If not None, dict with episode, group_num,
                             best_eval_reward from a previous run.
        """
        config = self.config
        accelerator = self.accelerator

        if accelerator is None or accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            logger.info("\n" + "=" * 70)
            logger.info(f"GRPO Training — mode={config.action_mode.upper()}")
            logger.info("=" * 70)
            logger.info(f"[Model] {config.model_name}")
            logger.info(f"[Loss Type] {config.loss_type.upper()}")
            if config.action_mode == "text":
                logger.info(f"[Actions] Plain text: {', '.join(self.action_words)}")
                logger.info(f"[Two-Stage] {config.use_two_stage}")
            else:
                logger.info(f"[Actions] Compound JSON helpers: {self.helper_functions}")
            logger.info(f"[Agents] {config.num_agents}")
            logger.info(f"[Episodes] {config.num_episodes}")
            _penvs = config.parallel_envs if config.parallel_envs > 0 else (
                config.episodes_per_gpu if accelerator is not None else config.episodes_per_update
            )
            logger.info(
                f"[Parallel Rollout] {_penvs} envs/batch, "
                f"macro_infer_batch={config.macro_infer_batch}"
            )
            if accelerator is not None:
                eps_per_group = config.episodes_per_gpu * accelerator.num_processes
                logger.info(
                    f"[Multi-GPU] {accelerator.num_processes} GPUs × {config.episodes_per_gpu} eps/GPU "
                    f"= {eps_per_group} per group"
                )
            logger.info(f"[Old Model Update] every {config.old_model_update_interval} groups")
            logger.info("=" * 70 + "\n")

        if resume_metadata:
            episode = resume_metadata["episode"]
            best_eval_reward = resume_metadata.get("best_eval_reward", float('-inf'))
            group_num = resume_metadata.get("group_num", 0)
            if accelerator is None or accelerator.is_main_process:
                logger.info(f"[Resumed] Starting from episode={episode}, group={group_num}, "
                            f"best_eval_reward={best_eval_reward:.2f}")
        else:
            episode = 0
            best_eval_reward = float('-inf')
            group_num = 0

        while episode < config.num_episodes:
            group_start_episode = episode

            if accelerator is None or accelerator.is_main_process:
                expected_episodes = (
                    config.episodes_per_gpu * accelerator.num_processes
                    if accelerator is not None else config.episodes_per_update
                )
                logger.info(f"\n[Group {group_num}] Episodes {episode}-{episode + expected_episodes - 1} (expected)")

            # Step 1: Update old model every N groups
            if group_num % config.old_model_update_interval == 0:
                self.update_old_model()

            # Step 2: Collect trajectories (parallel envs)
            parallel_envs = config.parallel_envs if config.parallel_envs > 0 else (
                config.episodes_per_gpu if accelerator is not None else config.episodes_per_update
            )

            if accelerator is not None:
                total_needed = config.episodes_per_gpu
                trajectories = []
                batch_idx = 0
                while len(trajectories) < total_needed:
                    batch_size = min(parallel_envs, total_needed - len(trajectories))
                    log_samples_flag = (
                        batch_idx == 0 and
                        group_num % config.log_interval == 0 and
                        accelerator.is_main_process
                    )
                    try:
                        batch_trajs = run_parallel_episodes(
                            self, num_envs=batch_size,
                            use_ref_model=False,
                            log_samples=log_samples_flag,
                            same_init_state=True,
                        )
                        trajectories.extend(batch_trajs)
                        for j, traj in enumerate(batch_trajs):
                            logger.info(
                                f"  [GPU{accelerator.process_index}] Ep{len(trajectories)-len(batch_trajs)+j}: "
                                f"R={traj['total_reward']:.2f}, Steps={traj['steps']}, "
                                f"Time={traj['rollout_time']:.2f}s"
                            )
                    except Exception as e:
                        logger.error(
                            f"  [GPU{accelerator.process_index}] Parallel batch {batch_idx} failed: {e}",
                            exc_info=True
                        )
                        # Fall back to sequential for this batch
                        for i in range(batch_size):
                            try:
                                traj = run_episode(self, use_ref_model=False, log_samples=False)
                                trajectories.append(traj)
                            except Exception as e2:
                                logger.error(f"  Fallback episode also failed: {e2}")
                                continue
                    batch_idx += 1
            else:
                total_needed = config.episodes_per_update
                trajectories = []
                batch_idx = 0
                while len(trajectories) < total_needed:
                    batch_size = min(parallel_envs, total_needed - len(trajectories))
                    log_samples_flag = (batch_idx == 0 and group_num % config.log_interval == 0)
                    try:
                        batch_trajs = run_parallel_episodes(
                            self, num_envs=batch_size,
                            use_ref_model=False,
                            log_samples=log_samples_flag,
                            same_init_state=True,
                        )
                        trajectories.extend(batch_trajs)
                        for j, traj in enumerate(batch_trajs):
                            logger.info(
                                f"  Ep{episode + len(trajectories)-len(batch_trajs)+j}: "
                                f"R={traj['total_reward']:.2f}, Steps={traj['steps']}, "
                                f"Time={traj['rollout_time']:.2f}s"
                            )
                    except Exception as e:
                        logger.error(f"Parallel batch {batch_idx} failed: {e}", exc_info=True)
                        for i in range(batch_size):
                            try:
                                traj = run_episode(self, use_ref_model=False, log_samples=False)
                                trajectories.append(traj)
                            except Exception as e2:
                                logger.error(f"Fallback episode failed: {e2}")
                                continue
                    batch_idx += 1

            # Update episode counter
            if accelerator is not None:
                local_count = len(trajectories)
                count_tensor = torch.tensor(
                    [local_count], dtype=torch.long,
                    device=next(self.model.parameters()).device
                )
                all_counts = accelerator.gather(count_tensor)
                total_collected = all_counts.sum().item()
                episode += total_collected
                if accelerator.is_main_process:
                    logger.info(
                        f"  ✓ Collected {total_collected} episodes "
                        f"({[c.item() for c in all_counts]} per GPU)"
                    )
            else:
                episode += len(trajectories)

            # Log trajectory to file
            if len(trajectories) > 0 and config.log_trajectory:
                if accelerator is None or accelerator.is_main_process:
                    import random
                    random_idx = random.randint(0, len(trajectories) - 1)
                    log_episode_to_file(config, trajectories[random_idx], group_num, random_idx, accelerator)
                    logger.info(f"  Logged episode {random_idx} trajectory to {config.trajectory_log_file}")

            if len(trajectories) == 0:
                if accelerator is None or accelerator.is_main_process:
                    logger.warning("No valid trajectories collected, skipping update")
                group_num += 1
                continue

            # Step 3: Compute advantages
            trajectories = compute_advantages(trajectories, config, accelerator, self.model)

            if accelerator is not None:
                accelerator.wait_for_everyone()

            # Log rollout stats
            avg_reward = self._log_rollout_stats(trajectories, episode, accelerator)

            # Step 4: Inner optimization epochs
            try:
                if accelerator is None or accelerator.is_main_process:
                    logger.info(
                        f"\n  Inner Optimization ({config.num_inner_epochs} epochs, "
                        f"minibatch={config.minibatch_size}):"
                    )

                epoch_losses = []
                epoch_clip_fracs = []
                final_grad_norm = 0.0

                for inner_epoch in range(config.num_inner_epochs):
                    minibatch_iterator = create_minibatch_iterator(trajectories, config.minibatch_size)
                    epoch_loss_sum = 0.0
                    epoch_clip_sum = 0.0
                    epoch_samples = 0
                    num_minibatches = 0

                    for minibatch_idx, minibatch in enumerate(minibatch_iterator):
                        self.optimizer.zero_grad()

                        model_device = (
                            next(self.model.parameters()).device
                            if accelerator is not None else self.device
                        )

                        all_samples = flatten_trajectories(minibatch)
                        total_samples = len(all_samples)
                        if total_samples == 0:
                            continue

                        micro_batch_size = config.micro_batch_size
                        num_chunks = (total_samples + micro_batch_size - 1) // micro_batch_size

                        total_loss_sum = 0.0
                        total_clip_sum = 0.0
                        total_n_samples = 0

                        for i in range(0, total_samples, micro_batch_size):
                            micro_batch = all_samples[i: i + micro_batch_size]
                            loss, clip_fraction, n_samples = compute_loss_on_samples(
                                self.model, micro_batch, self.tokenizer, config, model_device
                            )

                            if n_samples > 0:
                                normalized_loss = loss / num_chunks
                                # Use accelerator.backward so DeepSpeed can
                                # handle loss scaling and gradient communication
                                # correctly under ZeRO-3 / bf16.
                                if accelerator is not None:
                                    accelerator.backward(normalized_loss)
                                else:
                                    normalized_loss.backward()
                                total_loss_sum += loss.item() * n_samples
                                total_clip_sum += clip_fraction * n_samples
                                total_n_samples += n_samples

                        if total_n_samples > 0:
                            if self.use_deepspeed:
                                # DeepSpeed clips gradients internally via the
                                # gradient_clipping value set in DeepSpeedPlugin.
                                # Calling clip_grad_norm_ here would raise a ValueError.
                                grad_norm = torch.tensor(config.max_grad_norm)
                            elif accelerator is not None:
                                grad_norm = accelerator.clip_grad_norm_(
                                    self.model.parameters(), config.max_grad_norm
                                )
                            else:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), config.max_grad_norm
                                )

                            self.optimizer.step()

                            avg_loss = total_loss_sum / total_n_samples
                            avg_clip = total_clip_sum / total_n_samples
                            epoch_loss_sum += total_loss_sum
                            epoch_clip_sum += total_clip_sum
                            epoch_samples += total_n_samples
                            final_grad_norm = grad_norm
                            num_minibatches += 1

                            if inner_epoch == 0 and minibatch_idx == 0:
                                if accelerator is None or accelerator.is_main_process:
                                    logger.info(
                                        f"    [Epoch 1/{config.num_inner_epochs}, Batch 1] "
                                        f"Loss={avg_loss:.4f}, ClipFrac={avg_clip:.3f}, "
                                        f"MicroBatches={num_chunks}"
                                    )

                    if epoch_samples > 0:
                        epoch_avg_loss = epoch_loss_sum / epoch_samples
                        epoch_avg_clip = epoch_clip_sum / epoch_samples
                        epoch_losses.append(epoch_avg_loss)
                        epoch_clip_fracs.append(epoch_avg_clip)

                        if inner_epoch == config.num_inner_epochs - 1:
                            if accelerator is None or accelerator.is_main_process:
                                logger.info(
                                    f"    [Epoch {inner_epoch + 1}/{config.num_inner_epochs}] "
                                    f"Avg Loss={epoch_avg_loss:.4f}, Avg ClipFrac={epoch_avg_clip:.3f}"
                                )

                if self.training_step >= 0:
                    self.scheduler.step()
                self.training_step += 1

                if len(epoch_losses) > 0:
                    final_loss = epoch_losses[-1]
                    final_clip_frac = epoch_clip_fracs[-1]
                    avg_loss_all_epochs = sum(epoch_losses) / len(epoch_losses)

                    if accelerator is None or accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(
                            f"  Final: Loss={final_loss:.4f} (avg={avg_loss_all_epochs:.4f}), "
                            f"ClipFrac={final_clip_frac:.3f}, GradNorm={final_grad_norm:.4f}, "
                            f"LR={current_lr:.2e}"
                        )

                        if config.use_wandb:
                            wandb.log({
                                "train/loss": final_loss,
                                "train/loss_avg_all_epochs": avg_loss_all_epochs,
                                "train/clip_fraction": final_clip_frac,
                                "train/grad_norm": final_grad_norm,
                                "train/learning_rate": current_lr,
                                "train/training_step": self.training_step,
                            }, step=episode)
                else:
                    if accelerator is None or accelerator.is_main_process:
                        logger.warning("No valid samples in any epoch, skipping update")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                if accelerator is None or accelerator.is_main_process:
                    logger.error(f"Training step failed: {e}", exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                group_num += 1
                continue

            # Save training stats atomically (write to tmp, then rename)
            if accelerator is None or accelerator.is_main_process:
                stats = {
                    "episode": episode, "group": group_num,
                    "rewards": self.episode_rewards, "steps": self.episode_steps,
                    "best_eval_reward": best_eval_reward, "training_step": self.training_step
                }
                stats_path = os.path.join(config.output_dir, "training_stats.json")
                stats_tmp = stats_path + ".tmp"
                with open(stats_tmp, "w") as f:
                    json.dump(stats, f, indent=2)
                os.replace(stats_tmp, stats_path)

            # Mid-training evaluation + model saving
            if config.eval_interval > 0 and episode % config.eval_interval == 0 and episode < config.num_episodes:
                if accelerator is None or accelerator.is_main_process:
                    logger.info(f"\n=== Mid-Training Evaluation (Episode {episode}) ===")
                eval_reward, _ = self.evaluate(num_episodes=config.num_eval_episodes, current_episode=episode)
                if accelerator is None or accelerator.is_main_process:
                    self._save_model(f"checkpoint_ep{episode}", episode)
                    self._save_training_state(episode, group_num, best_eval_reward)
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self._save_model("best_model", episode)
                        logger.info(f"  New best eval reward: {best_eval_reward:.2f}")
                    logger.info("=== Resuming Training ===\n")

            # Check for SIGTERM / SIGUSR1 interruption
            if self._interrupted:
                if accelerator is None or accelerator.is_main_process:
                    logger.info(f"\n⚠ Interrupted at episode {episode}, group {group_num}. Saving checkpoint...")
                    self._save_model(f"interrupted_ep{episode}", episode)
                    self._save_training_state(episode, group_num, best_eval_reward)
                    logger.info("  Interrupt checkpoint saved. Exiting training loop.")
                break

            group_num += 1

        self._last_episode = episode

        if accelerator is None or accelerator.is_main_process:
            logger.info(f"\n=== Training Complete ===")
            logger.info(f"Best eval reward: {best_eval_reward:.2f}, Total groups: {group_num}\n")

        return self.model

    def _log_rollout_stats(self, trajectories, episode, accelerator):
        """Log rollout statistics and return avg_reward (0.0 for non-main processes)."""
        config = self.config
        avg_reward = 0.0

        if accelerator is not None:
            local_rewards = [t["total_reward"] for t in trajectories]
            local_steps = [t["steps"] for t in trajectories]
            local_times = [t["rollout_time"] for t in trajectories]

            model_device = next(self.model.parameters()).device
            all_rewards = accelerator.gather(
                torch.tensor(local_rewards, dtype=torch.float32, device=model_device)
            )
            all_steps = accelerator.gather(
                torch.tensor(local_steps, dtype=torch.float32, device=model_device)
            )
            all_times = accelerator.gather(
                torch.tensor(local_times, dtype=torch.float32, device=model_device)
            )

            all_rewards_list = all_rewards.cpu().tolist()
            avg_reward = np.mean(all_rewards_list)
            std_reward = np.std(all_rewards_list)
            max_reward = np.max(all_rewards_list)
            min_reward = np.min(all_rewards_list)
            avg_steps = np.mean(all_steps.cpu().tolist())
            avg_rollout_time = np.mean(all_times.cpu().tolist())
            total_rollout_time = sum(all_times.cpu().tolist())

            if accelerator.is_main_process:
                self.episode_rewards.append(avg_reward)
                self.episode_steps.append(avg_steps)
                logger.info(
                    f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], "
                    f"Steps: {avg_steps:.1f}"
                )
                logger.info(f"  Rollout Time: avg={avg_rollout_time:.2f}s, total={total_rollout_time:.2f}s")

                if config.use_wandb:
                    wandb.log({
                        "episode": episode, "reward/mean": avg_reward, "reward/std": std_reward,
                        "reward/min": min_reward, "reward/max": max_reward,
                        "episode_steps": avg_steps,
                        "rollout_time/avg": avg_rollout_time,
                        "rollout_time/total": total_rollout_time,
                    }, step=episode)

                # Action distribution
                all_actions = [a for traj in trajectories for a in traj["actions"]]
                if all_actions:
                    action_counts = {}
                    for a in all_actions:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    total_a = len(all_actions)
                    dist_str = " | ".join([
                        f"{a}:{action_counts.get(a, 0) * 100 // total_a}%"
                        for a in self.action_words
                    ])
                    logger.info(f"  Actions (GPU0 sample): {dist_str}")

                if std_reward < 0.01:
                    logger.warning("  ⚠ Policy collapse detected!")
        else:
            avg_reward = np.mean([t["total_reward"] for t in trajectories])
            std_reward = np.std([t["total_reward"] for t in trajectories])
            max_reward = np.max([t["total_reward"] for t in trajectories])
            min_reward = np.min([t["total_reward"] for t in trajectories])
            avg_steps = np.mean([t["steps"] for t in trajectories])
            avg_rollout_time = np.mean([t["rollout_time"] for t in trajectories])
            total_rollout_time = sum([t["rollout_time"] for t in trajectories])

            self.episode_rewards.append(avg_reward)
            self.episode_steps.append(avg_steps)

            logger.info(
                f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], "
                f"Steps: {avg_steps:.1f}"
            )
            logger.info(f"  Rollout Time: avg={avg_rollout_time:.2f}s, total={total_rollout_time:.2f}s")

            if config.use_wandb:
                wandb.log({
                    "episode": episode, "reward/mean": avg_reward, "reward/std": std_reward,
                    "reward/min": min_reward, "reward/max": max_reward,
                    "episode_steps": avg_steps,
                    "rollout_time/avg": avg_rollout_time,
                    "rollout_time/total": total_rollout_time,
                }, step=episode)

            all_actions = [a for traj in trajectories for a in traj["actions"]]
            if all_actions:
                action_counts = {}
                for a in all_actions:
                    action_counts[a] = action_counts.get(a, 0) + 1
                total_a = len(all_actions)
                dist_str = " | ".join([
                    f"{a}:{action_counts.get(a, 0) * 100 // total_a}%"
                    for a in self.action_words
                ])
                logger.info(f"  Actions: {dist_str}")

            if std_reward < 0.01:
                logger.warning("  ⚠ Policy collapse detected!")

        return avg_reward


# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    args = parse_args()

    if args.sanity_check:
        logger.info("\n=== SANITY CHECK MODE ===")
        logger.info("Forcing num_agents=1 and num_episodes=50")
        args.num_agents = 1
        args.num_episodes = 50
        if args.wandb_run_name is None:
            args.wandb_run_name = f"sanity_check_{args.action_mode}"

    config = GRPOConfig(
        action_mode=args.action_mode,
        action_tokens=args.action_tokens,
        model_name=args.model_name,
        thinking_tokens=args.thinking_tokens,
        use_two_stage=args.use_two_stage,
        logprob_mode=args.logprob_mode,
        loss_type=args.loss_type,
        num_episodes=args.num_episodes,
        episodes_per_update=args.episodes_per_update,
        episodes_per_gpu=args.episodes_per_gpu,
        parallel_envs=args.parallel_envs,
        macro_infer_batch=args.macro_infer_batch,
        num_agents=args.num_agents,
        max_env_steps=args.max_env_steps,
        eat_reward=args.eat_reward,
        clean_reward=args.clean_reward,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_lora=args.use_lora,
        use_accelerate=args.use_accelerate,
        use_deepspeed=args.use_deepspeed,
        num_inner_epochs=args.num_inner_epochs,
        minibatch_size=args.minibatch_size,
        samples_per_micro_batch=args.samples_per_micro_batch,
        micro_batch_size=args.micro_batch_size,
        old_model_update_interval=args.old_model_update_interval,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        log_trajectory=args.log_trajectory,
        trajectory_log_file=args.trajectory_log_file,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    trainer = CleanupGameGRPO(config)

    # ── Resume from checkpoint if requested ──
    resume_metadata = None
    if args.resume_from:
        logger.info(f"\n=== Resuming from {args.resume_from} ===")
        # Find the most recent model checkpoint in the resume dir
        resume_dir = args.resume_from
        # Try to find a model checkpoint (adapter weights for LoRA)
        import glob as glob_mod
        # Look for checkpoint_ep*, interrupted_ep*, or best_model
        ckpt_candidates = sorted(
            glob_mod.glob(os.path.join(resume_dir, "checkpoint_ep*")) +
            glob_mod.glob(os.path.join(resume_dir, "interrupted_ep*")),
            key=os.path.getmtime,
        )
        model_ckpt = ckpt_candidates[-1] if ckpt_candidates else os.path.join(resume_dir, "best_model")
        if os.path.exists(model_ckpt):
            logger.info(f"  Loading model weights from {model_ckpt}")
            from peft import set_peft_model_state_dict
            try:
                from safetensors.torch import load_file
                adapter_path = os.path.join(model_ckpt, "adapter_model.safetensors")
                if os.path.exists(adapter_path):
                    state_dict = load_file(adapter_path)
                else:
                    adapter_path = os.path.join(model_ckpt, "adapter_model.bin")
                    state_dict = torch.load(adapter_path, map_location="cpu")
                set_peft_model_state_dict(
                    trainer.accelerator.unwrap_model(trainer.model)
                    if trainer.accelerator is not None else trainer.model,
                    state_dict,
                )
                logger.info(f"  Loaded adapter weights from {adapter_path}")
            except Exception as e:
                logger.warning(f"  Could not load adapter weights: {e}")
                logger.info("  Attempting full model state dict load...")
                try:
                    unwrapped = (
                        trainer.accelerator.unwrap_model(trainer.model)
                        if trainer.accelerator is not None else trainer.model
                    )
                    from transformers import AutoModelForCausalLM
                    resumed_model = AutoModelForCausalLM.from_pretrained(
                        model_ckpt, torch_dtype=torch.bfloat16
                    )
                    unwrapped.load_state_dict(resumed_model.state_dict(), strict=False)
                    del resumed_model
                    logger.info(f"  Loaded full model weights from {model_ckpt}")
                except Exception as e2:
                    logger.error(f"  Failed to load model weights: {e2}")
        else:
            logger.warning(f"  No model checkpoint found in {resume_dir}")

        # Load training state (optimizer, scheduler, counters)
        resume_metadata = trainer.load_training_state(resume_dir)
        if resume_metadata:
            logger.info(f"  Will resume training from episode {resume_metadata['episode']}")
        logger.info("=== Resume complete ===\n")

    # Visualization mode
    if args.visualize:
        logger.info("\n=== VISUALIZATION MODE ===")
        logger.info("Running one rollout with step-by-step visualization\n")
        trainer.visualize_rollout(
            use_ref_model=args.viz_use_ref,
            save_to_file=args.viz_save_file
        )
        logger.info("\nVisualization complete. Exiting.")
        return

    # Pre-training evaluation (uses same fixed eval_states as mid-training evals)
    # Skip pre-eval when resuming (baseline already captured in original run)
    if not args.skip_pre_eval and resume_metadata is None:
        logger.info("\n=== Pre-Training Evaluation ===")
        trainer.evaluate(num_episodes=args.num_eval_episodes, current_episode=0)

    # Training
    model = trainer.train(resume_metadata=resume_metadata)

    # Post-training evaluation + final model save
    if not args.skip_post_eval:
        final_ep = getattr(trainer, '_last_episode', config.num_episodes)
        logger.info("\n=== Post-Training Evaluation ===")
        eval_reward, _ = trainer.evaluate(num_episodes=args.num_eval_episodes, current_episode=final_ep)
        trainer._save_model(f"checkpoint_ep{final_ep}", final_ep)
        logger.info(f"  Final model saved at episode {final_ep}")

    # Finish wandb
    if config.use_wandb:
        wandb.finish()
        logger.info("Wandb run finished.")


if __name__ == "__main__":
    main()
