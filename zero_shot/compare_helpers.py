"""
zeroshot.py

Zero-shot evaluation and comparison of two LLM agent planning methods:
1. High-Level (JSON + helpers): Agents output JSON calling functions like move_to, clean_at.
2. Low-Level (Text Actions): Agents output raw directional text (up, down, left, right, clean, eat, stay).

Logs a sample trajectory for each method and provides statistical comparisons.
"""

import os
import json
import re
import time
import copy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from env_move import CleanupEnvMove, Config as EnvConfigMove
import helpers
import prompt_template

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NUM_EVAL_EPISODES = 20
MAX_ENV_STEPS = 20
NUM_AGENTS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------------------------------------------------------------------
# High-Level Method Action Registry
# ---------------------------------------------------------------------------
ACTION_REGISTRY = {
    "move_to": {"func": helpers.move_to, "has_args": True, "arg_keys": ["coord_x", "coord_y"]},
    "eat_at": {"func": helpers.eat_at, "has_args": True, "arg_keys": ["coord_x", "coord_y"]},
    "clean_at": {"func": helpers.clean_at, "has_args": True, "arg_keys": ["coord_x", "coord_y"]},
    "random_explore": {"func": helpers.random_explore, "has_args": False, "arg_keys": []},
}
VALID_ACTIONS = set(ACTION_REGISTRY.keys())
DEFAULT_ACTION = "random_explore"

# ---------------------------------------------------------------------------
# Shared Utilities
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
    return tokenizer, model

@torch.no_grad()
def generate_text(tokenizer, model, messages: list, max_new_tokens: int) -> str:
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Method 1: High-Level JSON Logic
# ---------------------------------------------------------------------------
def get_high_level_action_prompt(env, agent_id, thinking_response):
    """Fallback construct since create_single_stage_prompt was missing standard definition."""
    system_msg = prompt_template.get_system_context() + "\n" + prompt_template.get_action_api()
    obs_text = helpers.get_observation_description(env, agent_id)
    user_content = f"You are Agent {agent_id}.\nObservation: {obs_text}\n\nYour Analysis:\n{thinking_response}\n\nBased on your analysis, output the specific JSON action to execute now."
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]

def parse_high_level_action(text: str) -> dict:
    fallback = {"action": DEFAULT_ACTION, "args": {}}
    text = text.replace("```json", "").replace("```", "").strip()
    
    match_methods = [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'\{[^{}]*\}', t, re.DOTALL).group()),
        lambda t: json.loads(re.search(r'\{.*"args"\s*:\s*\{[^{}]*\}.*\}', t, re.DOTALL).group())
    ]
    
    for method in match_methods:
        try:
            data = method(text)
            action = data.get("action", "").strip()
            if action in VALID_ACTIONS:
                return {"action": action, "args": data.get("args", {})}
        except Exception:
            continue
            
    return fallback

def execute_high_level_action(env, agent_id: int, parsed: dict) -> str:
    action_name = parsed.get("action", DEFAULT_ACTION)
    args = parsed.get("args", {})
    if action_name not in ACTION_REGISTRY:
        action_name = DEFAULT_ACTION
        
    entry = ACTION_REGISTRY[action_name]
    func = entry["func"]

    if entry["has_args"]:
        missing = [k for k in entry["arg_keys"] if k not in args]
        if missing:
            return helpers.random_explore(env, agent_id)[0]
        clean_args = {k: int(v) if str(v).isdigit() else v for k, v in args.items()}
        action_str, _ = func(env, agent_id, **clean_args)
    else:
        action_str, _ = func(env, agent_id)

    return action_str

# ---------------------------------------------------------------------------
# Method 2: Low-Level Text Logic
# ---------------------------------------------------------------------------
def parse_observation_to_coords(env, agent_id: int) -> str:
    ax, ay_internal = env.agents[agent_id]
    ay_display = (env.height - 1) - ay_internal
    half_w, half_h = 2, 1
    y0, y1 = max(0, ay_internal - half_h), min(env.height - 1, ay_internal + half_h)
    x0, x1 = max(0, ax - half_w), min(env.width - 1, ax + half_w)

    dirt_coords, apple_coords = [], []
    for y_int in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if env.items[y_int][x] == '#':
                dirt_coords.append((x, (env.height - 1) - y_int))
            elif env.items[y_int][x] == 'a':
                apple_coords.append((x, (env.height - 1) - y_int))

    other_agents = []
    for other_id, (ox, oy_int) in env.agents.items():
        if other_id != agent_id and x0 <= ox <= x1 and y0 <= oy_int <= y1:
            other_agents.append(f"Agent {other_id} at ({ox},{(env.height - 1) - oy_int})")

    obs_text = f"You at ({ax},{ay_display})."
    if dirt_coords: obs_text += f" Dirt at {', '.join([f'({x},{y})' for x, y in dirt_coords])}."
    if apple_coords: obs_text += f" Apple at {', '.join([f'({x},{y})' for x, y in apple_coords])}."
    if other_agents: obs_text += " " + " ".join(other_agents) + "."
    if not (dirt_coords or apple_coords or other_agents): obs_text += " Nothing in your view."
    return obs_text

def get_low_level_prompts(obs_text, agent_id, thinking_text=None):
    system_msg = (
        "You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+1.0 each). "
        "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
        "- Cleaning dirt itself gives NO points, but is necessary to enable apple spawning. "
        "- You can only eat/clean items at your position. "
        "Available actions: up = move up, down = move down, left = move left, right = move right, "
        "clean = clean dirt if on your cell, eat = eat apple if on your cell, stay = stay. "
    )
    
    if thinking_text is None:
        return [
            {"role": "system", "content": system_msg + "Think about the situation and give your reasoning in short."},
            {"role": "user", "content": obs_text}
        ]
    else:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": obs_text},
            {"role": "assistant", "content": thinking_text},
            {"role": "user", "content": "Based on your thinking above, choose your best immediate action and output only ONE action word: up, down, left, right, clean, eat, or stay."}
        ]

def extract_low_level_action(response: str) -> str:
    words = response.strip().lower().split()
    valid_actions = {'up', 'down', 'left', 'right', 'clean', 'eat', 'stay'}
    for i in range(min(5, len(words))):
        word = words[-(i+1)].strip('.,!?;:"\'')
        if word in valid_actions:
            return word
    return "stay"

# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
class ZeroShotEvaluator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = EnvConfigMove(
            width=15, height=9, n_agents=NUM_AGENTS, river_width=3, 
            max_steps=MAX_ENV_STEPS, init_dirt_prob=0.3, apple_spawn_base=0.1
        )
        self.init_states = []
        
    def generate_states(self):
        print(f"Generating {NUM_EVAL_EPISODES} initial states...")
        for i in range(NUM_EVAL_EPISODES):
            self.cfg.seed = SEED + i
            env = CleanupEnvMove(self.cfg)
            env.reset()
            self.init_states.append(env.get_state())

    def run_episode(self, method: str, state_dict: dict, log_traj: bool = False):
        env = CleanupEnvMove(self.cfg)
        env.set_state(copy.deepcopy(state_dict))
        
        total_reward = 0.0
        actual_dirts_cleaned = 0
        actual_apples_eaten = 0

        if log_traj:
            print(f"\n{'='*60}\nSAMPLE TRAJECTORY: {method.upper()}\n{'='*60}")
            print("INITIAL STATE:\n" + env.render() + "\n")

        for step in range(1, MAX_ENV_STEPS + 1):
            env_agents_before = copy.deepcopy(env.agents)
            env_items_before = copy.deepcopy(env.items)
            actions_to_step = {}

            if log_traj: print(f"--- Step {step} ---")

            for agent_id in range(1, NUM_AGENTS + 1):
                if method == "high_level":
                    thinking_msgs = prompt_template.create_thinking_prompt(env, agent_id)
                    thinking = generate_text(self.tokenizer, self.model, thinking_msgs, 256)
                    
                    action_msgs = get_high_level_action_prompt(env, agent_id, thinking)
                    action_resp = generate_text(self.tokenizer, self.model, action_msgs, 128)
                    
                    parsed = parse_high_level_action(action_resp)
                    act_str = execute_high_level_action(env, agent_id, parsed)
                    actions_to_step[agent_id] = act_str

                    if log_traj:
                        print(f"Agent {agent_id} (High-Level):")
                        print(f"  Obs: {helpers.get_observation_description(env, agent_id)}")
                        print(f"  Thinking: {thinking}")
                        print(f"  Decision: {parsed} -> Env Action: '{act_str}'")

                elif method == "low_level":
                    obs_text = parse_observation_to_coords(env, agent_id)
                    
                    thinking_msgs = get_low_level_prompts(obs_text, agent_id)
                    thinking = generate_text(self.tokenizer, self.model, thinking_msgs, 256)
                    
                    action_msgs = get_low_level_prompts(obs_text, agent_id, thinking)
                    action_resp = generate_text(self.tokenizer, self.model, action_msgs, 32)
                    
                    act_str = extract_low_level_action(action_resp)
                    actions_to_step[agent_id] = act_str

                    if log_traj:
                        print(f"Agent {agent_id} (Low-Level):")
                        print(f"  Obs: {obs_text}")
                        print(f"  Thinking: {thinking}")
                        print(f"  Decision: '{action_resp}' -> Env Action: '{act_str}'")

            _, rewards, done, _ = env.step(actions_to_step)
            
            step_reward = sum(rewards.values())
            total_reward += step_reward
            actual_apples_eaten += int(step_reward) 

            # Calculate actual dirts cleaned (clean succeeds only if agent is on dirt and stays)
            for aid, act_str in actions_to_step.items():
                if act_str == "clean":
                    ax, ay = env_agents_before[aid]
                    if env_items_before[ay][ax] == '#':
                        actual_dirts_cleaned += 1

            if log_traj:
                print(f"Step Results: Rewards={rewards} | Dirts Cleaned This Step: {actual_dirts_cleaned}")
                print(env.render() + "\n")

            if done: break

        return total_reward, actual_dirts_cleaned, actual_apples_eaten

    def evaluate(self):
        self.generate_states()
        
        results = {
            "high_level": {"rewards": [], "dirts": [], "apples": []},
            "low_level": {"rewards": [], "dirts": [], "apples": []}
        }

        methods = ["high_level", "low_level"]
        
        for method in methods:
            print(f"\nEvaluating Method: {method.upper()}...")
            for i, state in enumerate(self.init_states):
                # Log the trajectory only for the very first episode of each method
                log_traj = (i == 0)
                
                start_t = time.time()
                r, d, a = self.run_episode(method, state, log_traj=log_traj)
                
                results[method]["rewards"].append(r)
                results[method]["dirts"].append(d)
                results[method]["apples"].append(a)
                
                print(f"  Episode {i+1}/{NUM_EVAL_EPISODES} | Reward: {r} | Dirts: {d} | Apples: {a} | Time: {time.time() - start_t:.1f}s")

        print("\n" + "="*60)
        print("ZERO-SHOT EVALUATION RESULTS (Mean ± Std)")
        print("="*60)
        
        for method in methods:
            r_arr = np.array(results[method]["rewards"])
            d_arr = np.array(results[method]["dirts"])
            a_arr = np.array(results[method]["apples"])
            
            print(f"Method: {method.upper()}")
            print(f"  Final Rewards  : {r_arr.mean():.2f} ± {r_arr.std():.2f}")
            print(f"  Dirts Cleaned  : {d_arr.mean():.2f} ± {d_arr.std():.2f}")
            print(f"  Apples Eaten   : {a_arr.mean():.2f} ± {a_arr.std():.2f}\n")

if __name__ == "__main__":
    tokenizer, model = load_model(MODEL_NAME)
    evaluator = ZeroShotEvaluator(tokenizer, model)
    evaluator.evaluate()