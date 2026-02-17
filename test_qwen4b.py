"""
Test script for Qwen3-4B on the Cleanup environment using prompt_template.

Runs a full episode where the LLM decides high-level actions for each agent
via two-stage prompting (thinking → action selection).
"""

import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from env_move import CleanupEnvMove, Config
import helpers
import prompt_template


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"       # Qwen3-4B (instruct-tuned by default)
MAX_NEW_TOKENS_THINKING = 256      # tokens for stage-1 reasoning
MAX_NEW_TOKENS_ACTION = 50          # tokens for stage-2 action JSON
MAX_ENV_STEPS = 5            # episode length
NUM_AGENTS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Valid high-level actions that map to helper functions
VALID_ACTIONS = {"smart_clean_step", "smart_forage_step", "random_walk"}

ACTION_FUNC_MAP = {
    "smart_clean_step": helpers.smart_clean_step,
    "smart_forage_step": helpers.smart_forage_step,
    "random_walk": helpers.random_walk,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str):
    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {DEVICE}.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(tokenizer, model, messages: list, max_new_tokens: int) -> str:
    """Apply chat template, tokenize, generate, and decode."""
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_action(text: str) -> str:
    """Extract the action name from the model's JSON output.

    Tries JSON parsing first, then falls back to regex matching.
    Returns a valid action name or 'smart_clean_step' as default.
    """
    # Try JSON parse
    try:
        data = json.loads(text)
        action = data.get("action", "").strip()
        if action in VALID_ACTIONS:
            return action
    except json.JSONDecodeError:
        pass

    # Try to find JSON embedded in surrounding text
    json_match = re.search(r'\{[^}]*"action"\s*:\s*"([^"]+)"[^}]*\}', text)
    if json_match:
        action = json_match.group(1).strip()
        if action in VALID_ACTIONS:
            return action

    # Last resort: look for any valid action string
    for act in VALID_ACTIONS:
        if act in text:
            return act

    print(f"  [WARN] Could not parse action from: {text!r}; defaulting to smart_clean_step")
    return "smart_clean_step"


# ---------------------------------------------------------------------------
# Two-stage agent decision
# ---------------------------------------------------------------------------
def decide_action(tokenizer, model, obs_text: str, agent_id: int) -> tuple:
    """Run the two-stage prompt pipeline for one agent.

    Returns (action_name, thinking_text, action_raw_text).
    """
    # Stage 1: thinking / reasoning
    thinking_msgs = prompt_template.create_thinking_prompt(obs_text, agent_id)
    thinking_response = generate_text(tokenizer, model, thinking_msgs, MAX_NEW_TOKENS_THINKING)

    # Stage 2: action selection
    action_msgs = prompt_template.create_single_stage_prompt(obs_text, thinking_response, agent_id)
    action_response = generate_text(tokenizer, model, action_msgs, MAX_NEW_TOKENS_ACTION)

    action_name = parse_action(action_response)
    return action_name, thinking_response, action_response


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(tokenizer, model):
    cfg = Config(
        width=15,
        height=9,
        n_agents=NUM_AGENTS,
        river_width=3,
        max_steps=MAX_ENV_STEPS,
        seed=42,
    )
    env = CleanupEnvMove(cfg)
    obs = env.reset()

    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    print(env.render())
    print()

    total_rewards = {i: 0.0 for i in range(1, NUM_AGENTS + 1)}
    step_times = []

    for step in range(1, MAX_ENV_STEPS + 1):
        step_start = time.time()
        actions = {}

        for agent_id in range(1, NUM_AGENTS + 1):
            obs_text = obs[agent_id]
            action_name, thinking, raw_action = decide_action(
                tokenizer, model, obs_text, agent_id
            )
            actions[agent_id] = action_name

            print(f"--- Step {step} | Agent {agent_id} ---")
            print(f"  Observation:\n    " + obs_text.replace("\n", "\n    "))
            print(f"  Thinking: {thinking}")
            print(f"  Raw action output: {raw_action}")
            print(f"  Chosen action: {action_name}")

        # Convert high-level action names to low-level env actions via helpers
        low_level_actions = {}
        for agent_id, action_name in actions.items():
            func = ACTION_FUNC_MAP[action_name]
            low_level_actions[agent_id] = func(env, agent_id)

        obs, rewards, done, info = env.step(low_level_actions)

        for agent_id in range(1, NUM_AGENTS + 1):
            total_rewards[agent_id] += rewards[agent_id]

        elapsed = time.time() - step_start
        step_times.append(elapsed)

        print(f"\n  Step {step} results: rewards={rewards}  "
              f"dirt={info['dirt_count']}  apples={info['apple_count']}  "
              f"time={elapsed:.2f}s")
        print(env.render())
        print()

        if done:
            break

    # Summary
    print("=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"  Steps completed : {info['step']}")
    print(f"  Final scores    : {info['scores']}")
    print(f"  Total rewards   : {dict(total_rewards)}")
    print(f"  Final dirt count: {info['dirt_count']}")
    print(f"  Final apple cnt : {info['apple_count']}")
    print(f"  Avg step time   : {sum(step_times)/len(step_times):.2f}s")
    print(f"  Total time      : {sum(step_times):.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tokenizer, model = load_model(MODEL_NAME)

    # print("\n--- Prompt preview (stage 1, agent 1) ---")
    # sample_obs = "**x##\n*1x#x\n**x*x"
    # sample_msgs = prompt_template.create_thinking_prompt(sample_obs, 1)
    # for msg in sample_msgs:
    #     print(f"[{msg['role']}] {msg['content'][:200]}...")
    # print()

    run_episode(tokenizer, model)
