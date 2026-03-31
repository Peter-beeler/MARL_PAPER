"""
diagnostic_eval.py — Verbose diagnostic evaluation of a single checkpoint.

Logs every step: global map, agent observations, full prompts, LLM responses,
parsed actions, rewards, and role assignments.

Usage:
    python diagnostic_eval.py \
        --checkpoint ./grpo_textgame_checkpoints_.../checkpoint_ep960 \
        --num_eval_episodes 20 \
        --output diagnostic_V1.log
"""

import os
import sys
import argparse
import time
import json
import re as _re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from env.cleanup import CleanupEnvMove, Config as EnvConfigMove
from utils.config import GRPOConfig
from utils.observation import (
    obs_to_text, get_observation_description,
    get_role_specific_observation, get_global_role_context,
    check_action_continuation,
)
from utils.prompts import (
    create_single_stage_prompt_compound,
    create_role_compound_prompt,
    create_role_assignment_prompt,
)


def load_model(checkpoint_path, base_model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config)

    if is_lora:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def perform_role_assignment(model, tokenizer, env, role_dict, num_agents, device):
    """Single-env role assignment. Returns new role_dict and the raw LLM response."""
    global_context = get_global_role_context(env)
    prompt = create_role_assignment_prompt(global_context, role_dict, num_agents, tokenizer)

    inputs = tokenizer(
        [prompt], return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs.input_ids[0].shape[0]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    response_clean = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
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
            if role_str in ('eater', 'cleaner') and aid in role_dict:
                role_dict[aid] = role_str

    return role_dict, prompt, response


def generate_action(model, tokenizer, env, agent_id, role, config, device,
                    last_action=None):
    """Generate a single compound action for one agent. Returns all intermediate data."""
    # Build observation
    if role:
        obs_text = get_role_specific_observation(env, agent_id, role)
    else:
        obs_text = get_observation_description(env, agent_id)

    # Check continuation
    cont_hint = ""
    if last_action:
        cont_hint = check_action_continuation(env, agent_id, last_action)
    if cont_hint:
        obs_text = obs_text + "\n" + cont_hint

    # Build prompt
    if role:
        prompt = create_role_compound_prompt(obs_text, config, tokenizer, env, agent_id, role)
    else:
        prompt = create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)

    # Tokenize
    inputs = tokenizer(
        [prompt], return_tensors="pt", truncation=True,
        max_length=config.max_length, padding=True
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.thinking_tokens + config.action_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs.input_ids[0].shape[0]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Parse JSON action
    response_clean = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
    if not response_clean:
        response_clean = response

    action_data = None
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
                        action_data = json.loads(response_clean[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break

    # Execute compound action
    low_level = "stay"
    action_name = "PARSE_FAIL"
    if action_data and "action" in action_data:
        action_name = action_data["action"]
        args = action_data.get("args", {})
        try:
            from utils.observation import (
                move_to, clean_at, eat_at, random_explore,
                find_nearest_apples, find_nearest_dirts,
            )
            fn_map = {
                "move_to": move_to,
                "clean_at": clean_at,
                "eat_at": eat_at,
                "random_explore": random_explore,
                "find_nearest_apples": find_nearest_apples,
                "find_nearest_dirts": find_nearest_dirts,
            }
            if action_name in fn_map:
                result = fn_map[action_name](env, agent_id, **args)
                if isinstance(result, tuple):
                    low_level = result[0]
                else:
                    low_level = result
            else:
                low_level = "stay"
                action_name = f"UNKNOWN({action_name})"
        except Exception as e:
            low_level = "stay"
            action_name = f"EXEC_ERR({action_name}: {e})"

    return {
        "obs_text": obs_text,
        "prompt": prompt,
        "response": response,
        "response_clean": response_clean,
        "action_data": action_data,
        "action_name": action_name,
        "low_level": low_level,
    }


def run_diagnostic(args):
    device = args.device
    out = open(args.output, "w") if args.output else sys.stdout

    def log(msg=""):
        print(msg, file=out, flush=True)

    config = GRPOConfig(
        action_mode="compound",
        model_name=args.base_model,
        thinking_tokens=256,
        action_tokens=128,
        num_agents=args.num_agents,
        max_env_steps=args.max_env_steps,
        role_assignment_interval=10,
        train_on_role_tokens=False,
        use_accelerate=False,
        use_deepspeed=False,
        device=device,
    )

    env_config = EnvConfigMove(
        n_agents=args.num_agents,
        max_steps=args.max_env_steps,
        seed=args.seed,
        eat_reward=1.0,
        clean_reward=0.0,
    )

    log(f"Loading model from: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, args.base_model, device)
    log(f"Model loaded on {device}\n")

    # Generate fixed eval states
    eval_states = []
    for i in range(args.num_eval_episodes):
        env = CleanupEnvMove(EnvConfigMove(
            n_agents=args.num_agents,
            max_steps=args.max_env_steps,
            seed=args.seed + 1000 + i,
            eat_reward=1.0,
            clean_reward=0.0,
        ))
        env.reset()
        eval_states.append(env.get_state())

    all_rewards = []

    for ep_idx in range(args.num_eval_episodes):
        log("=" * 80)
        log(f"EPISODE {ep_idx + 1}/{args.num_eval_episodes}")
        log("=" * 80)

        env = CleanupEnvMove(env_config)
        env.reset()
        env.set_state(eval_states[ep_idx])

        roles = {a: "eater" for a in range(1, args.num_agents + 1)}
        last_actions = {a: None for a in range(1, args.num_agents + 1)}
        total_reward = 0.0

        # Initial role assignment
        log("\n--- INITIAL ROLE ASSIGNMENT ---")
        roles, ra_prompt, ra_response = perform_role_assignment(
            model, tokenizer, env, roles, args.num_agents, device
        )
        log(f"  Global context: {get_global_role_context(env)}")
        log(f"  Roles assigned: {roles}")
        log(f"  LLM response: {ra_response[:300]}")

        for step in range(args.max_env_steps):
            # Role reassignment every 10 steps
            if step > 0 and step % 10 == 0:
                old_roles = dict(roles)
                roles, ra_prompt, ra_response = perform_role_assignment(
                    model, tokenizer, env, roles, args.num_agents, device
                )
                changed = (roles != old_roles)
                log(f"\n  [ROLE REASSIGN step {step}] {old_roles} -> {roles} (changed={changed})")
                log(f"    Context: {get_global_role_context(env)}")
                log(f"    LLM: {ra_response[:200]}")

            # Log map state
            dirt_count = env._count_items('#')
            apple_count = env._count_items('a')
            log(f"\n--- Step {step} | dirt={dirt_count} apples={apple_count} "
                f"scores={dict(env.scores)} total_so_far={total_reward} ---")
            log(env.render())

            # Generate actions for each agent
            actions = {}
            for agent_id in range(1, args.num_agents + 1):
                role = roles.get(agent_id, "eater")
                result = generate_action(
                    model, tokenizer, env, agent_id, role, config, device,
                    last_action=last_actions[agent_id],
                )

                actions[agent_id] = result["low_level"]
                last_actions[agent_id] = result["action_data"]

                log(f"\n  Agent {agent_id} (role={role}) at {env.agents[agent_id]}:")
                log(f"    Obs: {result['obs_text'][:200]}")
                log(f"    LLM response: {result['response'][:300]}")
                log(f"    Parsed: {result['action_name']} -> low_level='{result['low_level']}'")
                if result["action_data"]:
                    log(f"    JSON: {result['action_data']}")

            # Step environment
            action_map = {}
            action_names = ["stay", "eat", "clean", "up", "down", "left", "right"]
            for aid, act_str in actions.items():
                act_idx = action_names.index(act_str) if act_str in action_names else 0
                action_map[aid] = act_idx

            obs, rewards, done, info = env.step(action_map)

            step_reward = sum(rewards.values())
            total_reward += step_reward
            if step_reward > 0:
                log(f"\n  >> REWARD: {rewards} (step total={step_reward})")

            if done:
                log(f"\n  >> EPISODE DONE at step {step}")
                break

        log(f"\n{'=' * 80}")
        log(f"EPISODE {ep_idx + 1} TOTAL REWARD: {total_reward}")
        log(f"Final scores: {dict(env.scores)}")
        log(f"{'=' * 80}\n")
        all_rewards.append(total_reward)

        # Free cache between episodes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    log("\n" + "=" * 80)
    log("DIAGNOSTIC SUMMARY")
    log("=" * 80)
    rewards_arr = np.array(all_rewards)
    log(f"Episodes: {len(all_rewards)}")
    log(f"Mean:     {rewards_arr.mean():.2f}")
    log(f"Std:      {rewards_arr.std():.2f}")
    log(f"Min:      {rewards_arr.min():.2f}")
    log(f"Max:      {rewards_arr.max():.2f}")
    log(f"Per-episode: {all_rewards}")
    zero_eps = [i + 1 for i, r in enumerate(all_rewards) if r == 0]
    if zero_eps:
        log(f"ZERO-REWARD EPISODES: {zero_eps}")
    log("=" * 80)

    if args.output:
        out.close()
        print(f"Diagnostic log written to: {args.output}")
        print(f"Mean={rewards_arr.mean():.2f}, Std={rewards_arr.std():.2f}, "
              f"Zeros={len(zero_eps)}/{len(all_rewards)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diagnostic evaluation with full logging")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    p.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--max_env_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_eval_episodes", type=int, default=20)
    p.add_argument("--output", type=str, default="diagnostic.log")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_diagnostic(args)
