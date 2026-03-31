"""
model_comparison.py — Load multiple model checkpoints and compare them on the
same 20 fixed initial environment states.

Usage:
    # Compare checkpoints (auto-detects LoRA adapters)
    python model_comparison.py \
        --checkpoints path/to/checkpoint_ep128 path/to/checkpoint_ep256 \
        --action_mode compound \
        --num_agents 3

    # Include the base (untrained) model as a baseline
    python model_comparison.py \
        --checkpoints path/to/checkpoint_ep128 \
        --include_base \
        --action_mode compound

    # Use a custom number of eval episodes / env steps
    python model_comparison.py \
        --checkpoints ckpt1 ckpt2 \
        --num_eval_episodes 20 \
        --max_env_steps 30
"""

import os
import sys
import argparse
import time
import logging
import json
import re as _re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from env.cleanup import CleanupEnvMove, Config as EnvConfigMove
from utils.config import GRPOConfig
from utils.model_setup import AllowOnlyActionWords

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Lightweight trainer-compatible wrapper
# ────────────────────────────────────────────────────────────────

class ModelWrapper:
    """Minimal object that satisfies the interface expected by
    ``utils.rollout.run_parallel_episodes`` and the generation helpers.

    Attributes accessed by the rollout/generation code:
        config, tokenizer, model, accelerator, device, env_config,
        action_words, helper_functions (compound mode only).
    """

    def __init__(self, model, tokenizer, config, env_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.env_config = env_config
        self.device = device

        # Not using multi-GPU / DeepSpeed in comparison mode
        self.accelerator = None
        self.ref_model = None
        self.old_model = None

        # Mode-specific attributes expected by generation code
        self.action_words = ["up", "down", "left", "right", "clean", "eat", "stay"]
        if config.action_mode == "compound":
            self.helper_functions = [
                "move_to", "clean_at", "eat_at", "random_explore",
                "find_nearest_apples", "find_nearest_dirts",
            ]

        # Text-mode constrained decoding (set but not used by current generation code)
        if config.action_mode == "text":
            self.action_logits_processor = AllowOnlyActionWords(tokenizer, self.action_words)

    def _perform_role_assignment(self, envs, role_states):
        """Assign roles (eater/cleaner) via LLM meta-call (no gradients)."""
        from utils.observation import get_global_role_context
        from utils.prompts import create_role_assignment_prompt

        config = self.config
        tokenizer = self.tokenizer
        device = self.device

        prompts = []
        for env, rs in zip(envs, role_states):
            global_context = get_global_role_context(env)
            prompts.append(create_role_assignment_prompt(
                global_context, rs['roles'], config.num_agents, tokenizer
            ))

        try:
            inputs = tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.warning(f"Batched role assignment generate failed: {e}, keeping current roles")
            return None

        for idx, (env, rs) in enumerate(zip(envs, role_states)):
            current_roles = rs['roles']
            try:
                gen_ids = outputs[idx][inputs.input_ids[idx].shape[0]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)

                response_clean = _re.sub(
                    r'<think>.*?</think>', '', response, flags=_re.DOTALL
                ).strip()
                if not response_clean:
                    response_clean = response

                json_match = None
                start = response_clean.find('{')
                if start != -1:
                    depth = 0
                    for i in range(start, len(response_clean)):
                        if response_clean[i] == '{':
                            depth += 1
                        elif response_clean[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    json_match = json.loads(response_clean[start:i + 1])
                                except json.JSONDecodeError:
                                    pass
                                break

                if json_match is not None:
                    for aid_str, role_str in json_match.items():
                        try:
                            aid = int(aid_str)
                        except (ValueError, TypeError):
                            continue
                        if role_str in ('eater', 'cleaner') and aid in current_roles:
                            current_roles[aid] = role_str
                else:
                    logger.warning(
                        f"Role assignment env {idx}: no valid JSON "
                        f"(preview='{response[:200]}')"
                    )
            except Exception as e:
                logger.debug(f"Role assignment parse failed for env {idx}: {e}")

        return None


# ────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, base_model_name, device):
    """Load a model from a checkpoint directory.

    Handles three cases:
      1. LoRA adapter checkpoint (contains adapter_config.json)
      2. Full model checkpoint (contains config.json / model.safetensors)
      3. HuggingFace model name (no local dir — download from hub)
    """
    logger.info(f"  Loading from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config)

    if is_lora:
        # Load base model, then attach LoRA adapter
        logger.info(f"  Detected LoRA adapter — loading base model '{base_model_name}' + adapter")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()  # merge for faster inference
        logger.info("  LoRA adapter merged into base model")
    else:
        # Full model checkpoint or HF hub name
        logger.info("  Loading as full model checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


# ────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────

def generate_eval_states(env_config, num_states, seed_offset=1000):
    """Pre-generate fixed initial states (same logic as _generate_eval_states)."""
    states = []
    for i in range(num_states):
        env = CleanupEnvMove(EnvConfigMove(
            n_agents=env_config.n_agents,
            max_steps=env_config.max_steps,
            seed=env_config.seed + seed_offset + i,
            eat_reward=env_config.eat_reward,
            clean_reward=env_config.clean_reward,
        ))
        env.reset()
        states.append(env.get_state())
    return states


def evaluate_model(wrapper, eval_states, parallel_envs):
    """Run evaluation episodes and return per-episode rewards.

    Args:
        wrapper: ModelWrapper instance.
        eval_states: List of state dicts (one per episode).
        parallel_envs: How many envs to batch at once.

    Returns:
        List of per-episode total rewards.
    """
    from utils.rollout import run_parallel_episodes, run_episode

    wrapper.model.eval()
    num_episodes = len(eval_states)
    all_rewards = []

    idx = 0
    while idx < num_episodes:
        batch_states = eval_states[idx : idx + parallel_envs]
        batch_size = len(batch_states)

        try:
            trajs = run_parallel_episodes(
                wrapper,
                num_envs=batch_size,
                use_ref_model=False,
                log_samples=False,
                initial_states=batch_states,
            )
            for j, traj in enumerate(trajs):
                ep_idx = idx + j
                logger.info(
                    f"    Ep {ep_idx + 1:>2}/{num_episodes}: "
                    f"R={traj['total_reward']:.2f}, Steps={traj['steps']}"
                )
                all_rewards.append(traj["total_reward"])
        except Exception as e:
            logger.warning(f"    Parallel batch failed ({e}), falling back to sequential")
            for j, state in enumerate(batch_states):
                ep_idx = idx + j
                try:
                    traj = run_episode(
                        wrapper, use_ref_model=False, log_samples=False, initial_state=state
                    )
                    logger.info(
                        f"    Ep {ep_idx + 1:>2}/{num_episodes}: "
                        f"R={traj['total_reward']:.2f}, Steps={traj['steps']}"
                    )
                    all_rewards.append(traj["total_reward"])
                except Exception as e2:
                    logger.error(f"    Ep {ep_idx + 1} failed: {e2}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        idx += batch_size

    return all_rewards


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare models from different checkpoints")

    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to checkpoint directories (or HF model names)",
    )
    p.add_argument(
        "--labels", nargs="+", default=None,
        help="Display labels for each checkpoint (same order). "
             "Defaults to directory basenames.",
    )
    p.add_argument(
        "--include_base", action="store_true",
        help="Also evaluate the untrained base model as a baseline",
    )

    # Mode
    p.add_argument("--action_mode", type=str, default="compound", choices=["text", "compound"])
    p.add_argument("--use_two_stage", action="store_true", default=True)
    p.add_argument("--no_two_stage", action="store_false", dest="use_two_stage")

    # Model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--thinking_tokens", type=int, default=256)
    p.add_argument("--action_tokens", type=int, default=128)

    # Environment
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--max_env_steps", type=int, default=30)
    p.add_argument("--eat_reward", type=float, default=1.0)
    p.add_argument("--clean_reward", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # Role assignment
    p.add_argument("--role_assignment_interval", type=int, default=0,
                   help="Reassign agent roles every N env steps (0=disabled)")

    # Evaluation
    p.add_argument("--num_eval_episodes", type=int, default=20)
    p.add_argument("--parallel_envs", type=int, default=4)
    p.add_argument("--macro_infer_batch", type=int, default=24)

    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Build config (shared across all models) ──
    config = GRPOConfig(
        action_mode=args.action_mode,
        use_two_stage=args.use_two_stage,
        model_name=args.base_model,
        thinking_tokens=args.thinking_tokens,
        action_tokens=args.action_tokens,
        num_agents=args.num_agents,
        max_env_steps=args.max_env_steps,
        eat_reward=args.eat_reward,
        seed=args.seed,
        macro_infer_batch=args.macro_infer_batch,
        role_assignment_interval=args.role_assignment_interval,
        train_on_role_tokens=False,
        use_accelerate=False,
        use_deepspeed=False,
        device=args.device,
    )

    env_config = EnvConfigMove(
        n_agents=args.num_agents,
        max_steps=args.max_env_steps,
        seed=args.seed,
        eat_reward=args.eat_reward,
        clean_reward=args.clean_reward,
    )

    # ── Generate fixed eval states ──
    eval_states = generate_eval_states(env_config, args.num_eval_episodes)
    logger.info(f"Generated {len(eval_states)} fixed evaluation states (seed={args.seed})")

    # ── Build model list ──
    checkpoints = list(args.checkpoints)
    labels = list(args.labels) if args.labels else [os.path.basename(c) for c in checkpoints]
    if args.include_base:
        checkpoints.insert(0, args.base_model)
        labels.insert(0, "base (untrained)")

    if len(labels) != len(checkpoints):
        logger.warning("Number of --labels doesn't match --checkpoints; using basenames.")
        labels = [os.path.basename(c) for c in checkpoints]

    # ── Evaluate each model ──
    results = {}

    role_desc = f"role_interval={config.role_assignment_interval}" if config.role_assignment_interval > 0 else "no_role"
    logger.info(f"\n{'=' * 70}")
    logger.info(f"MODEL COMPARISON — mode={config.action_mode}, {role_desc}, "
                f"agents={config.num_agents}, steps={config.max_env_steps}")
    logger.info(f"{'=' * 70}\n")

    for ckpt_path, label in zip(checkpoints, labels):
        logger.info(f"── {label} ──")

        t0 = time.time()
        model, tokenizer = load_model(ckpt_path, args.base_model, args.device)

        wrapper = ModelWrapper(model, tokenizer, config, env_config, args.device)

        rewards = evaluate_model(wrapper, eval_states, args.parallel_envs)
        elapsed = time.time() - t0

        if rewards:
            avg = np.mean(rewards)
            std = np.std(rewards)
            mn = np.min(rewards)
            mx = np.max(rewards)
            results[label] = {
                "avg": avg, "std": std, "min": mn, "max": mx,
                "rewards": rewards, "time": elapsed,
            }
            logger.info(
                f"  Result: {avg:.2f} +/- {std:.2f}  [{mn:.2f}, {mx:.2f}]  "
                f"({elapsed:.1f}s)\n"
            )
        else:
            results[label] = {"avg": 0, "std": 0, "min": 0, "max": 0, "rewards": [], "time": elapsed}
            logger.info(f"  No valid episodes.\n")

        # Free GPU memory before loading next model
        del model, wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Model':<30} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Time':>8}")
    logger.info("-" * 70)
    for label in labels:
        r = results.get(label, {})
        logger.info(
            f"{label:<30} {r.get('avg',0):>8.2f} {r.get('std',0):>8.2f} "
            f"{r.get('min',0):>8.2f} {r.get('max',0):>8.2f} {r.get('time',0):>7.1f}s"
        )
    logger.info("=" * 70)

    # ── Per-episode comparison (side-by-side) ──
    if len(results) > 1:
        logger.info(f"\nPER-EPISODE REWARDS:")
        header = f"{'Ep':>4}"
        for label in labels:
            header += f"  {label[:18]:>18}"
        logger.info(header)
        logger.info("-" * len(header))

        for i in range(args.num_eval_episodes):
            row = f"{i + 1:>4}"
            for label in labels:
                rews = results.get(label, {}).get("rewards", [])
                val = rews[i] if i < len(rews) else float("nan")
                row += f"  {val:>18.2f}"
            logger.info(row)


if __name__ == "__main__":
    main()
