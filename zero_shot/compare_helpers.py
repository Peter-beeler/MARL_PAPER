"""
Zero-shot comparison: With Helpers vs Without Helpers.

Runs 20 fixed start states, 3 agents, 50 steps each.
- Without helpers: plain text actions (up/down/left/right/clean/eat/stay)
- With helpers: JSON action dispatch via helpers.py + prompt_template.py

Reports per-test: final rewards, dirts cleaned, apples eaten.
Reports overall: avg +/- std for each metric.
"""

import os
import sys
import json
import time
import copy
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generated"))

from env_move import CleanupEnvMove, Config as EnvConfig

# Helper imports
from helpers import (
    observation_to_text,
    dispatch_action,
    parse_action_json,
)
from prompt_template import create_thinking_prompt, create_single_stage_prompt


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_STATES = 20
NUM_AGENTS = 3
MAX_STEPS = 50
BASE_SEED = 42
ACTION_WORDS = ["up", "down", "left", "right", "clean", "eat", "stay"]


# ============================================================================
# GENERATE FIXED START STATES
# ============================================================================

def generate_eval_states(num_states: int = NUM_STATES) -> List[Dict]:
    """Generate deterministic initial states (same scheme as grpo_text_action.py)."""
    states = []
    for i in range(num_states):
        env = CleanupEnvMove(EnvConfig(
            n_agents=NUM_AGENTS,
            max_steps=MAX_STEPS,
            seed=BASE_SEED + 1000 + i,
        ))
        env.reset()
        states.append(env.get_state())
    return states


# ============================================================================
# OBSERVATION FOR "WITHOUT HELPERS" (matches grpo_text_action.py)
# ============================================================================

def parse_observation_no_helpers(env, agent_id: int) -> str:
    """Convert observation to coordinate-based text (same as grpo_text_action.py)."""
    ax, ay_internal = env.agents[agent_id]
    ay_display = (env.height - 1) - ay_internal

    half_w, half_h = 2, 1
    y0 = max(0, ay_internal - half_h)
    y1 = min(env.height - 1, ay_internal + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    dirt_coords = []
    apple_coords = []
    for y_int in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if env.items[y_int][x] == "#":
                y_disp = (env.height - 1) - y_int
                dirt_coords.append((x, y_disp))
            elif env.items[y_int][x] == "a":
                y_disp = (env.height - 1) - y_int
                apple_coords.append((x, y_disp))

    other_agents_info = []
    for other_id, (ox, oy_int) in env.agents.items():
        if other_id == agent_id:
            continue
        if x0 <= ox <= x1 and y0 <= oy_int <= y1:
            oy_disp = (env.height - 1) - oy_int
            other_agents_info.append(f"Agent {other_id} at ({ox},{oy_disp})")

    obs_text = f"You at ({ax},{ay_display})."
    if dirt_coords:
        obs_text += f" Dirt at {', '.join([f'({x},{y})' for x, y in dirt_coords])}."
    if apple_coords:
        obs_text += f" Apple at {', '.join([f'({x},{y})' for x, y in apple_coords])}."
    if other_agents_info:
        obs_text += " " + " ".join(other_agents_info) + "."
    if not dirt_coords and not apple_coords and not other_agents_info:
        obs_text += " Nothing in your view."

    return obs_text


# ============================================================================
# PROMPT BUILDERS FOR "WITHOUT HELPERS" (matches grpo_text_action.py)
# ============================================================================

def build_thinking_prompt_no_helpers(obs_text: str, agent_id: int) -> List[Dict[str, str]]:
    """Stage 1 prompt for the no-helpers approach."""
    system_msg = (
        "You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+1.0 each). "
        "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
        "- Cleaning dirt itself gives NO points, but is necessary to enable apple spawning. "
        "- You can only eat/clean items at your position. "
        "Available actions: "
        "up = move one step up (increase y) "
        "down = move one step down (decrease y) "
        "left = move one step left (decrease x) "
        "right = move one step right (increase x) "
        "clean = clean dirt if on your cell "
        "eat = eat apple if on your cell "
        "stay = stay in current position. "
        "Think about the situation and give your reasoning in short."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": obs_text},
    ]


def build_action_prompt_no_helpers(
    obs_text: str, thinking_text: str
) -> List[Dict[str, str]]:
    """Stage 2 prompt for the no-helpers approach."""
    system_msg = (
        "You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+1.0 each). "
        "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
        "- Cleaning dirt itself gives NO points, but is necessary to enable apple spawning. "
        "- You can only eat/clean items at your position. "
        "Available actions: "
        "up = move one step up (increase y) "
        "down = move one step down (decrease y) "
        "left = move one step left (decrease x) "
        "right = move one step right (increase x) "
        "clean = clean dirt if on your cell "
        "eat = eat apple if on your cell "
        "stay = stay in current position "
    )
    action_instruction = (
        "Based on your thinking above, choose your best immediate action and output only ONE action word: "
        "up, down, left, right, clean, eat, or stay."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": obs_text},
        {"role": "assistant", "content": thinking_text},
        {"role": "user", "content": action_instruction},
    ]


# ============================================================================
# ACTION EXTRACTION FOR "WITHOUT HELPERS"
# ============================================================================

def get_action_from_response(response: str) -> str:
    """Extract action word from the end of the response (same as grpo_text_action.py)."""
    response = response.strip().lower()
    words = response.split()
    for i in range(min(5, len(words))):
        word = words[-(i + 1)].strip(".,!?;:")
        if word in ACTION_WORDS:
            return word
    return "stay"


# ============================================================================
# MODEL GENERATION
# ============================================================================

def generate_text(
    model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int, device
) -> str:
    """Generate text from chat messages, one prompt at a time."""
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    print("DEBUG: prompt string:", prompt_str)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# RUN EPISODE — WITHOUT HELPERS
# ============================================================================

def run_episode_no_helpers(
    model, tokenizer, env: CleanupEnvMove, device, verbose: bool = False,
    state_idx: int = 0,
) -> Dict:
    """Run one episode with plain text actions (no helpers)."""
    total_reward = 0.0
    apples_eaten = 0
    dirts_cleaned = 0

    for step in range(MAX_STEPS):
        print(
            f"\r  State {state_idx+1:2d}/{NUM_STATES} | Step {step+1:3d}/{MAX_STEPS} | "
            f"reward={total_reward:.1f}",
            end="", flush=True,
        )

        actions = {}
        for agent_id in range(1, NUM_AGENTS + 1):
            obs_text = parse_observation_no_helpers(env, agent_id)

            # Stage 1: thinking
            thinking_msgs = build_thinking_prompt_no_helpers(obs_text, agent_id)
            thinking_text = generate_text(model, tokenizer, thinking_msgs, max_new_tokens=256, device=device)

            # Stage 2: action
            action_msgs = build_action_prompt_no_helpers(obs_text, thinking_text)
            action_text = generate_text(model, tokenizer, action_msgs, max_new_tokens=10, device=device)

            actions[agent_id] = get_action_from_response(action_text)

            if verbose and step == 0:
                print(f"\n  [No-Helper] Agent {agent_id}: obs='{obs_text}' "
                      f"think='{thinking_text[:80]}...' action={actions[agent_id]}")

        obs, rewards, done, info = env.step(actions)

        for agent_id in range(1, NUM_AGENTS + 1):
            total_reward += rewards[agent_id]

        if done:
            break

    return {
        "total_reward": total_reward,
        "apples_eaten": apples_eaten,
        "final_scores": info["scores"],
        "dirt_count": info["dirt_count"],
        "apple_count": info["apple_count"],
        "init_dirt_count": info["init_dirt_count"],
        "steps": step + 1,
    }


# ============================================================================
# RUN EPISODE — WITH HELPERS
# ============================================================================

def run_episode_with_helpers(
    model, tokenizer, env: CleanupEnvMove, device, verbose: bool = False,
    state_idx: int = 0,
) -> Dict:
    """Run one episode using helpers (JSON action dispatch)."""
    total_reward = 0.0
    apples_eaten = 0
    dirts_cleaned = 0

    from helpers import move_toward

    for step in range(MAX_STEPS):
        print(
            f"\r  State {state_idx+1:2d}/{NUM_STATES} | Step {step+1:3d}/{MAX_STEPS} | "
            f"reward={total_reward:.1f}",
            end="", flush=True,
        )

        actions = {}
        for agent_id in range(1, NUM_AGENTS + 1):
            obs_text = observation_to_text(env, agent_id)

            # Stage 1: thinking
            thinking_msgs = create_thinking_prompt(obs_text, agent_id)
            thinking_text = generate_text(model, tokenizer, thinking_msgs, max_new_tokens=256, device=device)

            # Stage 2: action (JSON format)
            action_msgs = create_single_stage_prompt(obs_text, thinking_text, agent_id)
            action_text = generate_text(model, tokenizer, action_msgs, max_new_tokens=80, device=device)

            # Parse JSON action
            action_json = parse_action_json(action_text)
            if action_json is not None:
                action_name = action_json.get("action", "stay")
                args = action_json.get("args", {})

                if action_name == "move_to":
                    coord_x = args.get("coord_x", 0)
                    coord_y = args.get("coord_y", 0)
                    actions[agent_id] = move_toward(env, agent_id, coord_x, coord_y)
                elif action_name in ("move_up", "move_down", "move_left", "move_right"):
                    actions[agent_id] = action_name.replace("move_", "")
                elif action_name in ("eat", "clean", "stay"):
                    actions[agent_id] = action_name
                else:
                    actions[agent_id] = "stay"
            else:
                actions[agent_id] = get_action_from_response(action_text)

            if verbose and step == 0:
                print(f"\n  [Helper] Agent {agent_id}: action_text='{action_text[:80]}...' "
                      f"parsed={actions[agent_id]}")

        # Step env with all actions simultaneously (fair comparison)
        obs, rewards, done, info = env.step(actions)

        for agent_id in range(1, NUM_AGENTS + 1):
            total_reward += rewards[agent_id]

        if done:
            break

    return {
        "total_reward": total_reward,
        "apples_eaten": apples_eaten,
        "final_scores": info["scores"],
        "dirt_count": info["dirt_count"],
        "apple_count": info["apple_count"],
        "init_dirt_count": info["init_dirt_count"],
        "steps": step + 1,
    }


# ============================================================================
# TRACKING WRAPPER — counts cleans via env monkey-patch
# ============================================================================

class CleanTracker:
    """Wraps env.step to count actual dirts cleaned and apples eaten."""

    def __init__(self, env: CleanupEnvMove):
        self.env = env
        self.dirts_cleaned = 0
        self.apples_eaten = 0
        self._original_step = env.step

        # Monkey-patch env.step
        env.step = self._tracked_step

    def _tracked_step(self, actions):
        """Wrapper that counts cleans and eats before delegating."""
        # Check what items are at each agent's position BEFORE step
        for agent_id in range(1, NUM_AGENTS + 1):
            ax, ay = self.env.agents[agent_id]
            item = self.env.items[ay][ax]
            action = actions.get(agent_id, "stay")
            if isinstance(action, str):
                action = action.lower()
            if action == "clean" and item == "#":
                self.dirts_cleaned += 1
            if action == "eat" and item == "a":
                self.apples_eaten += 1

        return self._original_step(actions)

    def restore(self):
        """Restore original step function."""
        self.env.step = self._original_step


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot comparison: helpers vs no-helpers")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="HuggingFace model name or path")
    parser.add_argument("--verbose", action="store_true", help="Print per-step details")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"States: {NUM_STATES}, Agents: {NUM_AGENTS}, Steps: {MAX_STEPS}")
    print("=" * 80)

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")
    print(f"Model Max Position: {model.config.max_position_embeddings}")
    # If this is 2048, change max_length in generate_text to 2048
    # In main(), after loading the model:
    model.resize_token_embeddings(len(tokenizer))
    # Generate fixed start states
    eval_states = generate_eval_states(NUM_STATES)

    # Results storage
    results_no_helper = []
    results_with_helper = []

    # ---- RUN WITHOUT HELPERS ----
    print("=" * 80)
    print("RUNNING WITHOUT HELPERS (plain text actions)")
    print("=" * 80)

    for state_idx, state in enumerate(eval_states):
        env = CleanupEnvMove(EnvConfig(n_agents=NUM_AGENTS, max_steps=MAX_STEPS, seed=BASE_SEED))
        env.set_state(copy.deepcopy(state))

        tracker = CleanTracker(env)
        t0 = time.time()
        result = run_episode_no_helpers(model, tokenizer, env, device, verbose=args.verbose, state_idx=state_idx)
        elapsed = time.time() - t0

        result["dirts_cleaned"] = tracker.dirts_cleaned
        result["apples_eaten"] = tracker.apples_eaten  # override with accurate count
        tracker.restore()

        results_no_helper.append(result)
        print(
            f"\r  State {state_idx+1:2d}/{NUM_STATES}: "
            f"reward={result['total_reward']:6.1f}  "
            f"apples={result['apples_eaten']:3d}  "
            f"dirts_cleaned={result['dirts_cleaned']:3d}  "
            f"({elapsed:.1f}s)          "
        )

    # ---- RUN WITH HELPERS ----
    print()
    print("=" * 80)
    print("RUNNING WITH HELPERS (JSON action dispatch)")
    print("=" * 80)

    for state_idx, state in enumerate(eval_states):
        env = CleanupEnvMove(EnvConfig(n_agents=NUM_AGENTS, max_steps=MAX_STEPS, seed=BASE_SEED))
        env.set_state(copy.deepcopy(state))

        tracker = CleanTracker(env)
        t0 = time.time()
        result = run_episode_with_helpers(model, tokenizer, env, device, verbose=args.verbose, state_idx=state_idx)
        elapsed = time.time() - t0

        result["dirts_cleaned"] = tracker.dirts_cleaned
        result["apples_eaten"] = tracker.apples_eaten
        tracker.restore()

        results_with_helper.append(result)
        print(
            f"\r  State {state_idx+1:2d}/{NUM_STATES}: "
            f"reward={result['total_reward']:6.1f}  "
            f"apples={result['apples_eaten']:3d}  "
            f"dirts_cleaned={result['dirts_cleaned']:3d}  "
            f"({elapsed:.1f}s)          "
        )

    # ---- SUMMARY ----
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def summarize(results: List[Dict], label: str):
        rewards = [r["total_reward"] for r in results]
        apples = [r["apples_eaten"] for r in results]
        dirts = [r["dirts_cleaned"] for r in results]

        print(f"\n{label}:")
        print(f"  Total Reward : {np.mean(rewards):7.2f} +/- {np.std(rewards):6.2f}  "
              f"[min={np.min(rewards):.1f}, max={np.max(rewards):.1f}]")
        print(f"  Apples Eaten : {np.mean(apples):7.2f} +/- {np.std(apples):6.2f}  "
              f"[min={np.min(apples):.0f}, max={np.max(apples):.0f}]")
        print(f"  Dirts Cleaned: {np.mean(dirts):7.2f} +/- {np.std(dirts):6.2f}  "
              f"[min={np.min(dirts):.0f}, max={np.max(dirts):.0f}]")

    summarize(results_no_helper, "WITHOUT HELPERS (plain text actions)")
    summarize(results_with_helper, "WITH HELPERS (JSON action dispatch)")

    # ---- PER-STATE COMPARISON TABLE ----
    print()
    print("=" * 80)
    print("PER-STATE COMPARISON")
    print("=" * 80)
    print(f"{'State':>5} | {'--- No Helpers ---':>30} | {'--- With Helpers ---':>30}")
    print(f"{'':>5} | {'Reward':>8} {'Apples':>8} {'Dirts':>8} | {'Reward':>8} {'Apples':>8} {'Dirts':>8}")
    print("-" * 80)
    for i in range(NUM_STATES):
        nh = results_no_helper[i]
        wh = results_with_helper[i]
        print(
            f"{i+1:5d} | "
            f"{nh['total_reward']:8.1f} {nh['apples_eaten']:8d} {nh['dirts_cleaned']:8d} | "
            f"{wh['total_reward']:8.1f} {wh['apples_eaten']:8d} {wh['dirts_cleaned']:8d}"
        )

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    save_data = {
        "config": {
            "model": args.model,
            "num_states": NUM_STATES,
            "num_agents": NUM_AGENTS,
            "max_steps": MAX_STEPS,
            "base_seed": BASE_SEED,
        },
        "no_helpers": results_no_helper,
        "with_helpers": results_with_helper,
        "summary": {
            "no_helpers": {
                "reward_mean": float(np.mean([r["total_reward"] for r in results_no_helper])),
                "reward_std": float(np.std([r["total_reward"] for r in results_no_helper])),
                "apples_mean": float(np.mean([r["apples_eaten"] for r in results_no_helper])),
                "apples_std": float(np.std([r["apples_eaten"] for r in results_no_helper])),
                "dirts_mean": float(np.mean([r["dirts_cleaned"] for r in results_no_helper])),
                "dirts_std": float(np.std([r["dirts_cleaned"] for r in results_no_helper])),
            },
            "with_helpers": {
                "reward_mean": float(np.mean([r["total_reward"] for r in results_with_helper])),
                "reward_std": float(np.std([r["total_reward"] for r in results_with_helper])),
                "apples_mean": float(np.mean([r["apples_eaten"] for r in results_with_helper])),
                "apples_std": float(np.std([r["apples_eaten"] for r in results_with_helper])),
                "dirts_mean": float(np.mean([r["dirts_cleaned"] for r in results_with_helper])),
                "dirts_std": float(np.std([r["dirts_cleaned"] for r in results_with_helper])),
            },
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
