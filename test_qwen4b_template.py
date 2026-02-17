"""
Template for LLM-driven agent test scripts.

This template provides the skeleton for running an episode where an LLM
decides high-level actions (from helpers.py) for each agent via two-stage
prompting (thinking -> action selection).

=== FOR THE CODE-GENERATING LLM ===
Fill in / adapt the sections marked with # >>> TODO based on:
  - helpers.py       : available action functions and their signatures
  - env_move.py      : environment class, Config fields, action space
  - prompt_template.py : prompt-building functions for the LLM agent
  - helper_readme.md : documentation of helper functions

Key design point: helper functions have DIFFERENT signatures.
  - Some take only (env, agent_id)          -> e.g. eat, clean
  - Some take (env, agent_id, **kwargs)     -> e.g. move_to(env, agent_id, coord_x, coord_y)
The ACTION_FUNC_MAP and execute_action() below handle this generically
by unpacking the "args" dict from the LLM's JSON response.
"""

import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# >>> TODO: Update these imports to match your environment / helpers / prompt_template
from env_move import CleanupEnvMove, Config
import helpers
import prompt_template


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# >>> TODO: Set model name and generation parameters
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS_THINKING = 256       # tokens for stage-1 reasoning
MAX_NEW_TOKENS_ACTION = 128         # tokens for stage-2 action JSON (needs room for args)
MAX_ENV_STEPS = 5                   # episode length
NUM_AGENTS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------
# >>> TODO: Populate from helpers.py.
#
# Each entry maps an action name (str) to a dict with:
#   "func"       : the callable from helpers.py
#   "has_args"   : whether the LLM must supply extra arguments beyond (env, agent_id)
#   "arg_keys"   : list of expected kwarg names (order doesn't matter)
#                  Only needed when has_args=True.
#
# The LLM's JSON response format is:
#   {"action": "<name>", "agent_id": <int>, "args": {<key>: <value>, ...}}
# When has_args=False, "args" can be {} or omitted.

ACTION_REGISTRY = {
    "move_to": {
        "func": helpers.move_to,
        "has_args": True,
        "arg_keys": ["coord_x", "coord_y"],
    },
    "eat": {
        "func": helpers.eat,
        "has_args": False,
        "arg_keys": [],
    },
    "clean": {
        "func": helpers.clean,
        "has_args": False,
        "arg_keys": [],
    },
    # >>> TODO: Add more actions here as helpers.py grows. Examples:
    # "fire_weapon": {
    #     "func": helpers.fire_weapon,
    #     "has_args": True,
    #     "arg_keys": ["target_x", "target_y"],
    # },
    # "random_walk": {
    #     "func": helpers.random_walk,
    #     "has_args": False,
    #     "arg_keys": [],
    # },
}

VALID_ACTIONS = set(ACTION_REGISTRY.keys())

# >>> TODO: Choose which action to use when parsing fails
DEFAULT_ACTION = "clean"


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


# ---------------------------------------------------------------------------
# Action parsing  (handles args generically)
# ---------------------------------------------------------------------------
def parse_action(text: str) -> dict:
    """Extract the action name AND args from the model's JSON output.

    Returns a dict: {"action": str, "args": dict}
    Falls back to DEFAULT_ACTION with empty args on parse failure.
    """
    fallback = {"action": DEFAULT_ACTION, "args": {}}

    # --- Attempt 1: direct JSON parse of the whole output ---
    try:
        data = json.loads(text)
        action = data.get("action", "").strip()
        if action in VALID_ACTIONS:
            return {"action": action, "args": data.get("args", {})}
    except (json.JSONDecodeError, AttributeError):
        pass

    # --- Attempt 2: find a JSON object embedded in surrounding text ---
    json_match = re.search(r'\{[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = data.get("action", "").strip()
            if action in VALID_ACTIONS:
                return {"action": action, "args": data.get("args", {})}
        except (json.JSONDecodeError, AttributeError):
            pass

    # --- Attempt 3: find nested JSON (for actions with args) ---
    # Match pattern like {"action": "move_to", ..., "args": {"coord_x": 5, "coord_y": 3}}
    nested_match = re.search(r'\{[^{}]*"args"\s*:\s*\{[^{}]*\}[^{}]*\}', text)
    if nested_match:
        try:
            data = json.loads(nested_match.group())
            action = data.get("action", "").strip()
            if action in VALID_ACTIONS:
                return {"action": action, "args": data.get("args", {})}
        except (json.JSONDecodeError, AttributeError):
            pass

    # --- Attempt 4: look for any valid action string in the text ---
    for act in VALID_ACTIONS:
        if act in text:
            return {"action": act, "args": {}}

    print(f"  [WARN] Could not parse action from: {text!r}; defaulting to {DEFAULT_ACTION}")
    return fallback


def execute_action(env, agent_id: int, parsed: dict) -> int:
    """Call the helper function from ACTION_REGISTRY and return the low-level action int.

    Args:
        env:       The environment instance.
        agent_id:  Agent performing the action.
        parsed:    Output of parse_action(): {"action": str, "args": dict}

    Returns:
        int: The low-level action to pass to env.step().
    """
    action_name = parsed["action"]
    args = parsed.get("args", {})
    entry = ACTION_REGISTRY[action_name]
    func = entry["func"]

    if entry["has_args"]:
        # Validate that all required arg keys are present
        missing = [k for k in entry["arg_keys"] if k not in args]
        if missing:
            print(f"  [WARN] Action '{action_name}' missing args {missing}; defaulting to stay")
            return 0  # stay

        # Call with unpacked kwargs: func(env, agent_id, coord_x=5, coord_y=3, ...)
        kwargs = {k: args[k] for k in entry["arg_keys"]}
        _is_done, action_int = func(env, agent_id, **kwargs)
    else:
        # Call with just (env, agent_id)
        _is_done, action_int = func(env, agent_id)

    return action_int


# ---------------------------------------------------------------------------
# Two-stage agent decision
# ---------------------------------------------------------------------------
def decide_action(tokenizer, model, env, agent_id: int) -> tuple:
    """Run the two-stage prompt pipeline for one agent.

    Returns (parsed_action_dict, thinking_text, action_raw_text).
    """
    # >>> TODO: Adapt if your prompt_template has different function signatures
    # Stage 1: thinking / reasoning
    thinking_msgs = prompt_template.create_thinking_prompt(env, agent_id)
    thinking_response = generate_text(tokenizer, model, thinking_msgs, MAX_NEW_TOKENS_THINKING)

    # Stage 2: action selection
    action_msgs = prompt_template.create_single_stage_prompt(env, agent_id, thinking_response)
    action_response = generate_text(tokenizer, model, action_msgs, MAX_NEW_TOKENS_ACTION)

    parsed = parse_action(action_response)
    return parsed, thinking_response, action_response


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(tokenizer, model):
    # >>> TODO: Adjust Config fields to match your environment
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
        low_level_actions = {}

        for agent_id in range(1, NUM_AGENTS + 1):
            parsed, thinking, raw_action = decide_action(
                tokenizer, model, env, agent_id
            )

            print(f"--- Step {step} | Agent {agent_id} ---")
            print(f"  Observation:\n    " + obs[agent_id].replace("\n", "\n    "))
            print(f"  Thinking: {thinking}")
            print(f"  Raw action output: {raw_action}")
            print(f"  Parsed: {parsed}")

            # Execute: dispatch to the right helper with correct args
            low_level_actions[agent_id] = execute_action(env, agent_id, parsed)

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
    run_episode(tokenizer, model)
