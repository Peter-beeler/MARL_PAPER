"""
compare_base_tuned.py — Compare pretrained (base) vs tuned model on N fixed eval states.

Runs each model in two prompt modes:
  1. Non-role: generic compound/text prompts (all agents see the same action set)
  2. Role:    role-assigned prompts (eater/cleaner with role-specific obs, tool actions,
              continuation hints, and dynamic role switching via LLM meta-call)

Reports avg, std, min, max of final rewards for each (model, mode) combination.

Usage:
    python compare_base_tuned.py \\
        --tuned_checkpoint path/to/checkpoint_ep256 \\
        --num_eval_episodes 50 \\
        --num_agents 3 \\
        --max_env_steps 30 \\
        --action_mode compound

    # Load eval states from a training run's saved file:
    python compare_base_tuned.py \\
        --tuned_checkpoint path/to/checkpoint_ep256 \\
        --eval_states_file path/to/eval_states.pt \\
        --num_agents 3

    # Only run one mode:
    python compare_base_tuned.py \\
        --tuned_checkpoint path/to/checkpoint_ep256 \\
        --modes role
"""

import os
import sys
import json
import argparse
import re
import time
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from env.cleanup import CleanupEnvMove, Config as EnvConfigMove
from utils.config import GRPOConfig
from utils.model_setup import AllowOnlyActionWords
from utils.eval import _generate_eval_states

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Lightweight trainer-compatible wrapper
# ────────────────────────────────────────────────────────────────

class ModelWrapper:
    """Minimal object satisfying the interface expected by rollout/generation.

    Includes ``_perform_role_assignment`` so the rollout code can do dynamic
    role switching exactly like the full ``CleanupGameGRPO`` trainer.
    """

    def __init__(self, model, tokenizer, config, env_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.env_config = env_config
        self.device = device

        self.accelerator = None
        self.ref_model = None
        self.old_model = None
        self._role_states = None

        # Use env_move (standard env) for evaluation
        self.train_env_class = CleanupEnvMove

        self.action_words = ["up", "down", "left", "right", "clean", "eat", "stay"]
        if config.action_mode == "compound":
            self.helper_functions = [
                "move_to", "clean_at", "eat_at", "random_explore",
                "find_nearest_apples", "find_nearest_dirts",
            ]
        if config.action_mode == "text":
            self.action_logits_processor = AllowOnlyActionWords(tokenizer, self.action_words)

    # ── Role assignment (mirrors CleanupGameGRPO._perform_role_assignment) ──

    def _perform_role_assignment(self, envs, role_states):
        """Assign roles (eater/cleaner) to agents via a batched meta LLM call."""
        from utils.observation import get_global_role_context
        from utils.prompts import create_role_assignment_prompt

        config = self.config
        tokenizer = self.tokenizer
        device = self.device

        gen_model = self.model

        # Build all prompts
        prompts = []
        for env, rs in zip(envs, role_states):
            global_context = get_global_role_context(env)
            prompts.append(create_role_assignment_prompt(
                global_context, rs['roles'], config.num_agents, tokenizer
            ))

        # Batched generate
        try:
            inputs = tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.warning(f"Batched role assignment generate failed: {e}, keeping current roles")
            return

        # Parse each response
        for idx, (env, rs) in enumerate(zip(envs, role_states)):
            current_roles = rs['roles']
            try:
                gen_ids = outputs[idx][inputs.input_ids[idx].shape[0]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)

                response_clean = re.sub(
                    r'<think>.*?</think>', '', response, flags=re.DOTALL
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
            except Exception as e:
                logger.debug(f"Role assignment parse failed for env {idx}: {e}")


# ────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, base_model_name, device):
    """Load model from checkpoint (LoRA adapter, full checkpoint, or HF name)."""
    logger.info(f"  Loading from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config)

    if is_lora:
        logger.info(f"  Detected LoRA adapter — loading base '{base_model_name}' + adapter")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
        logger.info("  LoRA adapter merged into base model")
    else:
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

def evaluate_model(wrapper, eval_states, parallel_envs):
    """Run evaluation episodes, return list of per-episode total rewards."""
    from utils.rollout import run_parallel_episodes, run_episode

    wrapper.model.eval()
    num_episodes = len(eval_states)
    all_rewards = []

    idx = 0
    while idx < num_episodes:
        batch_states = eval_states[idx:idx + parallel_envs]
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
                    f"    Ep {ep_idx + 1:>3}/{num_episodes}: "
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
                        f"    Ep {ep_idx + 1:>3}/{num_episodes}: "
                        f"R={traj['total_reward']:.2f}, Steps={traj['steps']}"
                    )
                    all_rewards.append(traj["total_reward"])
                except Exception as e2:
                    logger.error(f"    Ep {ep_idx + 1} failed: {e2}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        idx += batch_size

    return all_rewards


def evaluate_one_model(
    checkpoint_path, base_model_name, config_template, env_config,
    eval_states, parallel_envs, device, modes,
):
    """Load a model once and evaluate it in each requested mode.

    Args:
        modes: list of mode names to run, subset of ["norole", "role"].

    Returns:
        dict  mode_name -> {"avg", "std", "min", "max", "rewards", "time"}
    """
    from copy import deepcopy

    model, tokenizer = load_model(checkpoint_path, base_model_name, device)

    mode_results = {}

    for mode in modes:
        config = deepcopy(config_template)
        if mode == "role":
            # Enable role switching
            config.role_assignment_interval = config._role_interval
        else:
            # Disable role switching
            config.role_assignment_interval = 0

        mode_label = "role" if mode == "role" else "non-role"
        logger.info(f"    [{mode_label}] evaluating ...")

        wrapper = ModelWrapper(model, tokenizer, config, env_config, device)

        t0 = time.time()
        rewards = evaluate_model(wrapper, eval_states, parallel_envs)
        elapsed = time.time() - t0

        if rewards:
            mode_results[mode] = {
                "avg": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "rewards": [float(r) for r in rewards],
                "time": elapsed,
            }
            r = mode_results[mode]
            logger.info(
                f"    [{mode_label}] {r['avg']:.2f} +/- {r['std']:.2f}  "
                f"[{r['min']:.2f}, {r['max']:.2f}]  ({elapsed:.1f}s)"
            )
        else:
            mode_results[mode] = {
                "avg": 0, "std": 0, "min": 0, "max": 0,
                "rewards": [], "time": elapsed,
            }
            logger.info(f"    [{mode_label}] No valid episodes.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mode_results


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare pretrained (base) vs tuned model in non-role and role modes"
    )

    # Required
    p.add_argument("--tuned_checkpoint", type=str, required=True,
                    help="Path to tuned model checkpoint directory")

    # Model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HuggingFace base model name (default: Qwen3-4B)")

    # Mode
    p.add_argument("--action_mode", type=str, default="compound", choices=["text", "compound"])
    p.add_argument("--use_two_stage", action="store_true", default=True)
    p.add_argument("--no_two_stage", action="store_false", dest="use_two_stage")
    p.add_argument("--thinking_tokens", type=int, default=256)
    p.add_argument("--action_tokens", type=int, default=128)

    # Prompt modes to run
    p.add_argument("--modes", nargs="+", default=["norole", "role"],
                    choices=["norole", "role"],
                    help="Prompt modes to evaluate (default: both norole and role)")

    # Role switching
    p.add_argument("--role_assignment_interval", type=int, default=10,
                    help="Reassign roles every N env steps in role mode (default: 10)")

    # Environment
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--max_env_steps", type=int, default=30)
    p.add_argument("--eat_reward", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation
    p.add_argument("--num_eval_episodes", type=int, default=50,
                    help="Number of evaluation episodes (default: 50)")
    p.add_argument("--eval_states_file", type=str, default=None,
                    help="Path to saved eval_states.pt file (reuse states from training). "
                         "Overrides --num_eval_episodes with the number of states in the file.")
    p.add_argument("--parallel_envs", type=int, default=4)
    p.add_argument("--macro_infer_batch", type=int, default=24)

    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    p.add_argument("--save_results", type=str, default=None,
                    help="Save per-episode results to this JSON file")

    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    modes = args.modes

    # Role mode only makes sense for compound action mode
    if "role" in modes and args.action_mode != "compound":
        logger.warning("Role mode requires --action_mode compound. Removing 'role' from modes.")
        modes = [m for m in modes if m != "role"]
        if not modes:
            modes = ["norole"]

    # ── Build shared config template ──
    # role_assignment_interval is set per-mode in evaluate_one_model;
    # we stash the user's value in a private field.
    config_template = GRPOConfig(
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
        role_assignment_interval=0,  # set per-mode
        use_accelerate=False,
        use_deepspeed=False,
        device=args.device,
    )
    config_template._role_interval = args.role_assignment_interval

    env_config = EnvConfigMove(
        n_agents=args.num_agents,
        max_steps=args.max_env_steps,
        seed=args.seed,
        eat_reward=args.eat_reward,
    )

    # ── Load or generate eval states ──
    if args.eval_states_file and os.path.exists(args.eval_states_file):
        eval_states = torch.load(args.eval_states_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded {len(eval_states)} eval states from {args.eval_states_file}")
    else:
        if args.eval_states_file:
            logger.warning(f"eval_states_file '{args.eval_states_file}' not found, generating new states")
        eval_states = _generate_eval_states(env_config, config_template, num_states=args.num_eval_episodes)
        logger.info(f"Generated {len(eval_states)} random eval states (seed={args.seed})")

    num_episodes = len(eval_states)
    modes_display = " + ".join(modes)

    logger.info(f"\n{'=' * 78}")
    logger.info(f"PRETRAINED vs TUNED — mode={args.action_mode}, "
                f"agents={args.num_agents}, steps={args.max_env_steps}, "
                f"episodes={num_episodes}")
    logger.info(f"Prompt modes: {modes_display}"
                + (f", role_interval={args.role_assignment_interval}" if "role" in modes else ""))
    logger.info(f"{'=' * 78}")

    # ── Evaluate base (pretrained) model ──
    logger.info(f"\n── Base model: {args.base_model} ──")
    base_results = evaluate_one_model(
        args.base_model, args.base_model, config_template, env_config,
        eval_states, args.parallel_envs, args.device, modes,
    )

    # ── Evaluate tuned model ──
    tuned_label = os.path.basename(args.tuned_checkpoint)
    logger.info(f"\n── Tuned model: {tuned_label} ──")
    tuned_results = evaluate_one_model(
        args.tuned_checkpoint, args.base_model, config_template, env_config,
        eval_states, args.parallel_envs, args.device, modes,
    )

    # ── Summary table ──
    logger.info(f"\n{'=' * 78}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 78}")
    logger.info(f"{'Model':<25} {'Mode':<10} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Time':>8}")
    logger.info("-" * 78)

    # Collect rows for table
    table_rows = []
    for mode in modes:
        mode_label = "non-role" if mode == "norole" else "role"
        br = base_results.get(mode, {})
        tr = tuned_results.get(mode, {})
        table_rows.append(("Base (pretrained)", mode_label, br))
        table_rows.append((f"Tuned ({tuned_label})", mode_label, tr))

    for display, mode_label, r in table_rows:
        logger.info(
            f"{display:<25} {mode_label:<10} {r.get('avg',0):>8.2f} {r.get('std',0):>8.2f} "
            f"{r.get('min',0):>8.2f} {r.get('max',0):>8.2f} {r.get('time',0):>7.1f}s"
        )

    # Deltas per mode
    logger.info("-" * 78)
    for mode in modes:
        mode_label = "non-role" if mode == "norole" else "role"
        br = base_results.get(mode, {})
        tr = tuned_results.get(mode, {})
        if br.get("rewards") and tr.get("rewards"):
            delta = tr["avg"] - br["avg"]
            sign = "+" if delta >= 0 else ""
            logger.info(f"{'Delta (tuned-base)':<25} {mode_label:<10} {sign}{delta:>7.2f}")
    logger.info("=" * 78)

    # ── Per-episode comparison ──
    for mode in modes:
        mode_label = "non-role" if mode == "norole" else "role"
        br = base_results.get(mode, {})
        tr = tuned_results.get(mode, {})
        if not (br.get("rewards") and tr.get("rewards")):
            continue

        logger.info(f"\nPER-EPISODE REWARDS [{mode_label}]:")
        logger.info(f"{'Ep':>4}  {'Base':>10}  {'Tuned':>10}  {'Delta':>10}")
        logger.info("-" * 40)
        for i in range(num_episodes):
            b = br["rewards"][i] if i < len(br["rewards"]) else float("nan")
            t = tr["rewards"][i] if i < len(tr["rewards"]) else float("nan")
            d = t - b
            logger.info(f"{i + 1:>4}  {b:>10.2f}  {t:>10.2f}  {d:>+10.2f}")

    # ── Cross-mode comparison (if both modes ran) ──
    if len(modes) == 2:
        logger.info(f"\nCROSS-MODE COMPARISON:")
        logger.info(f"{'Model':<25} {'norole Avg':>12} {'role Avg':>12} {'role - norole':>14}")
        logger.info("-" * 65)
        for display, results in [("Base (pretrained)", base_results),
                                  (f"Tuned ({tuned_label})", tuned_results)]:
            nr = results.get("norole", {})
            rl = results.get("role", {})
            nr_avg = nr.get("avg", 0)
            rl_avg = rl.get("avg", 0)
            diff = rl_avg - nr_avg
            sign = "+" if diff >= 0 else ""
            logger.info(f"{display:<25} {nr_avg:>12.2f} {rl_avg:>12.2f} {sign}{diff:>13.2f}")
        logger.info("=" * 65)

    # ── Save results ──
    if args.save_results:
        out = {
            "args": vars(args),
            "num_episodes": num_episodes,
            "modes": modes,
        }
        for mode in modes:
            out[f"base_{mode}"] = base_results.get(mode, {})
            out[f"tuned_{mode}"] = tuned_results.get(mode, {})

        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
