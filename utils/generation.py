"""
Action generation for both text mode and compound mode.

All generate_* functions return a standardized 8-tuple:
    (action, log_prob, thinking_text, full_response, action_text, action_prompt,
     action_input_ids, action_ids)

- text two-stage:  thinking_text=reasoning, action_text=action word, action_prompt=stage2 prompt
- text single:     thinking_text=response, action_text=response, action_prompt=""
- compound:        thinking_text=response (think+JSON), action_text=response, action_prompt=""
"""

import json
import re
import logging
import torch
from typing import Dict, List, Optional, Tuple

from .logprob import compute_batch_sequence_log_prob
from .observation import (
    obs_to_text, move_to, clean_at, eat_at, random_explore,
)
from .prompts import (
    create_thinking_prompt, create_action_prompt,
    create_single_stage_prompt_text, create_single_stage_prompt_compound,
)

logger = logging.getLogger(__name__)

ACTION_WORDS = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']


# ─────────────────────────────────────────────
# TEXT MODE HELPERS
# ─────────────────────────────────────────────

def get_action_from_response(response: str, action_words: List[str] = None) -> str:
    """Extract action word from the end of a response (text mode)."""
    if action_words is None:
        action_words = ACTION_WORDS
    response = response.strip().lower()
    words = response.split()
    for i in range(min(5, len(words))):
        word = words[-(i + 1)].strip('.,!?;:')
        if word in action_words:
            return word
    logger.debug(f"No valid action found in response: '{response}', defaulting to 'stay'")
    return "stay"


# ─────────────────────────────────────────────
# COMPOUND MODE HELPERS
# ─────────────────────────────────────────────

def parse_and_execute_action(response: str, agent_id: int, env) -> str:
    """
    Parse JSON from model response, call the appropriate helper, return low-level env action.

    Returns:
        Low-level action string: up/down/left/right/clean/eat/stay.
    """
    response = response.strip()
    logger.debug(f"Raw model action response: {response}")

    # Step 1: strip markdown code fence
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)
    else:
        # Step 2: balanced-brace extraction
        start = response.find('{')
        if start == -1:
            logger.debug(f"No JSON found in response: '{response[:80]}', falling back to random_explore")
            action, _ = random_explore(env, agent_id)
            return action
        depth = 0
        end = -1
        for i in range(start, len(response)):
            if response[i] == '{':
                depth += 1
            elif response[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            logger.debug(f"Unbalanced braces in response: '{response[:80]}', falling back to random_explore")
            action, _ = random_explore(env, agent_id)
            return action
        json_str = response[start:end + 1]

    try:
        data = json.loads(json_str)
        action_name = data.get('action', '')
        args = data.get('args', {})

        if action_name == 'move_to':
            action, _ = move_to(env, agent_id, int(args.get('coord_x', 0)), int(args.get('coord_y', 0)))
        elif action_name == 'clean_at':
            action, _ = clean_at(env, agent_id, int(args.get('coord_x', 0)), int(args.get('coord_y', 0)))
        elif action_name == 'eat_at':
            action, _ = eat_at(env, agent_id, int(args.get('coord_x', 0)), int(args.get('coord_y', 0)))
        elif action_name == 'random_explore':
            action, _ = random_explore(env, agent_id)
        else:
            logger.debug(f"Unknown action name: '{action_name}', falling back to random_explore")
            action, _ = random_explore(env, agent_id)

        return action

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Failed to parse JSON action: {e}, falling back to random_explore")
        action, _ = random_explore(env, agent_id)
        return action


# ─────────────────────────────────────────────
# CORE: generate_action (single agent)
# ─────────────────────────────────────────────

def generate_action(trainer, obs: str, agent_id: int, step: int, env, model):
    """
    Generate one action for a single agent (all modes).

    Args:
        trainer: CleanupGameGRPO instance.
        obs: Raw observation string.
        agent_id: Agent ID.
        step: Current env step.
        env: CleanupEnvMove instance.
        model: Model to use for generation.

    Returns:
        8-tuple: (action, log_prob, thinking_text, full_response, action_text,
                  action_prompt, action_input_ids, action_ids)
    """
    config = trainer.config
    tokenizer = trainer.tokenizer
    accelerator = trainer.accelerator
    device = trainer.device

    is_ref_model = (model is getattr(trainer, 'ref_model', None))
    target_device = getattr(trainer, 'ref_device', device) if is_ref_model else device
    # With ZeRO-3 the model is a DeepSpeedEngine; model.module has empty (sharded)
    # parameters so we must use the engine itself for generate() / forward().
    # For plain DDP we still unwrap to model.module so AllGather isn't involved.
    _is_zero3 = (
        accelerator is not None and
        getattr(getattr(accelerator, 'state', None), 'deepspeed_plugin', None) is not None
    )
    gen_model = model if _is_zero3 else (
        model.module if (accelerator is not None and hasattr(model, 'module')) else model
    )

    # ── TEXT TWO-STAGE ──
    if config.action_mode == "text" and config.use_two_stage:
        obs_text = obs_to_text(obs, env, agent_id, config)

        thinking_prompt = create_thinking_prompt(obs_text, agent_id, config, tokenizer)
        thinking_inputs = tokenizer(
            thinking_prompt, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=False
        ).to(target_device)
        if "attention_mask" not in thinking_inputs:
            thinking_inputs["attention_mask"] = torch.ones_like(thinking_inputs["input_ids"])

        try:
            with torch.no_grad():
                thinking_outputs = gen_model.generate(
                    **thinking_inputs,
                    max_new_tokens=config.thinking_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Thinking generation failed: {e}, using default action")
            return ("stay", torch.tensor(0.0, device=target_device), "", "", "", "", None, None)

        thinking_ids = thinking_outputs.sequences[0][thinking_inputs.input_ids.shape[1]:]
        thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=True)

        action_prompt = create_action_prompt(obs_text, thinking_text, config, tokenizer)
        action_inputs = tokenizer(
            action_prompt, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=False
        ).to(target_device)
        if "attention_mask" not in action_inputs:
            action_inputs["attention_mask"] = torch.ones_like(action_inputs["input_ids"])

        try:
            with torch.no_grad():
                action_outputs = gen_model.generate(
                    **action_inputs,
                    max_new_tokens=config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                    output_scores=True,
                )
        except RuntimeError as e:
            logger.warning(f"Action generation failed: {e}, using default action")
            return ("stay", torch.tensor(0.0, device=target_device), thinking_text, thinking_text, "", action_prompt, None, None)

        action_ids = action_outputs.sequences[0][action_inputs.input_ids.shape[1]:]
        action_text = tokenizer.decode(action_ids, skip_special_tokens=True)
        action = get_action_from_response(action_text, trainer.action_words)

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            if config.logprob_mode == "action+thinking":
                thinking_lp = compute_batch_sequence_log_prob(
                    log_prob_model, [thinking_inputs.input_ids], [thinking_ids],
                    target_device, tokenizer.pad_token_id, need_grad=False
                )[0]
                action_lp = compute_batch_sequence_log_prob(
                    log_prob_model, [action_inputs.input_ids], [action_ids],
                    target_device, tokenizer.pad_token_id, need_grad=False
                )[0]
                log_prob = thinking_lp + action_lp
            else:
                log_prob = compute_batch_sequence_log_prob(
                    log_prob_model, [action_inputs.input_ids], [action_ids],
                    target_device, tokenizer.pad_token_id, need_grad=False
                )[0]
        except Exception as e:
            logger.warning(f"Log prob calculation failed: {e}")
            log_prob = torch.tensor(-10.0, device=target_device)

        full_response = f"{thinking_text} -> {action_text}"
        return (action, log_prob, thinking_text, full_response, action_text, action_prompt,
                action_inputs.input_ids, action_ids)

    # ── TEXT SINGLE-STAGE ──
    elif config.action_mode == "text" and not config.use_two_stage:
        obs_text = obs_to_text(obs, env, agent_id, config)
        prompt = create_single_stage_prompt_text(obs_text, config, tokenizer)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=False
        ).to(target_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                    output_scores=True,
                )
        except RuntimeError as e:
            logger.warning(f"Generation failed: {e}, using default action")
            return ("stay", torch.tensor(0.0, device=target_device), "", "", "", "", None, None)

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        action = get_action_from_response(response, trainer.action_words)

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_prob = compute_batch_sequence_log_prob(
                log_prob_model, [inputs.input_ids], [generated_ids],
                target_device, tokenizer.pad_token_id, need_grad=False
            )[0]
        except Exception as e:
            logger.warning(f"Log prob calculation failed: {e}")
            log_prob = torch.tensor(-10.0, device=target_device)

        return (action, log_prob, response, response, response, "", inputs.input_ids, generated_ids)

    # ── COMPOUND MODE ──
    else:  # config.action_mode == "compound"
        obs_text = obs_to_text(obs, env, agent_id, config)
        prompt = create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=False
        ).to(target_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=config.thinking_tokens + config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Generation failed: {e}, using default action")
            return ("stay", torch.tensor(0.0, device=target_device), "", "", "", "", None, None)

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        action = parse_and_execute_action(response, agent_id, env)

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_prob = compute_batch_sequence_log_prob(
                log_prob_model, [inputs.input_ids], [generated_ids],
                target_device, tokenizer.pad_token_id, need_grad=False
            )[0]
        except Exception as e:
            logger.warning(f"Log prob calculation failed: {e}")
            log_prob = torch.tensor(-10.0, device=target_device)

        return (action, log_prob, response, response, response, "", inputs.input_ids, generated_ids)


# ─────────────────────────────────────────────
# CORE: generate_actions_batch (all agents)
# ─────────────────────────────────────────────

def generate_actions_batch(trainer, obs_dict: Dict[int, str], step: int, env, model) -> Dict:
    """
    Generate actions for all agents in a batch (all modes).

    Args:
        trainer: CleanupGameGRPO instance.
        obs_dict: Dict mapping agent_id → raw obs string.
        step: Current env step.
        env: CleanupEnvMove instance.
        model: Model to use for generation.

    Returns:
        Dict mapping agent_id → 8-tuple:
        (action, log_prob, thinking_text, full_response, action_text,
         action_prompt, action_input_ids, action_ids)
    """
    config = trainer.config
    tokenizer = trainer.tokenizer
    accelerator = trainer.accelerator
    device = trainer.device

    target_device = device
    # Same ZeRO-3 safety as in generate_action — use the engine directly.
    _is_zero3 = (
        accelerator is not None and
        getattr(getattr(accelerator, 'state', None), 'deepspeed_plugin', None) is not None
    )
    gen_model = model if _is_zero3 else (
        model.module if (accelerator is not None and hasattr(model, 'module')) else model
    )

    agent_ids = sorted(obs_dict.keys())
    num_agents = len(agent_ids)

    # ── TEXT TWO-STAGE BATCH ──
    if config.action_mode == "text" and config.use_two_stage:
        # Stage 1: batch thinking
        obs_texts = []
        thinking_prompts = []
        for agent_id in agent_ids:
            obs_text = obs_to_text(obs_dict[agent_id], env, agent_id, config)
            obs_texts.append(obs_text)
            thinking_prompts.append(create_thinking_prompt(obs_text, agent_id, config, tokenizer))

        thinking_inputs = tokenizer(
            thinking_prompts, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True
        ).to(target_device)
        if "attention_mask" not in thinking_inputs:
            thinking_inputs["attention_mask"] = torch.ones_like(thinking_inputs["input_ids"])

        try:
            with torch.no_grad():
                thinking_outputs = gen_model.generate(
                    **thinking_inputs,
                    max_new_tokens=config.thinking_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch thinking generation failed: {e}, falling back to sequential")
            results = {}
            for agent_id in agent_ids:
                results[agent_id] = generate_action(trainer, obs_dict[agent_id], agent_id, step, env, model)
            return results

        thinking_texts = []
        for i in range(num_agents):
            t_ids = thinking_outputs.sequences[i][thinking_inputs.input_ids[i].shape[0]:]
            thinking_texts.append(tokenizer.decode(t_ids, skip_special_tokens=True))

        # Stage 2: batch action
        action_prompts = []
        for i, agent_id in enumerate(agent_ids):
            action_prompts.append(create_action_prompt(obs_texts[i], thinking_texts[i], config, tokenizer))

        action_inputs = tokenizer(
            action_prompts, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True
        ).to(target_device)
        if "attention_mask" not in action_inputs:
            action_inputs["attention_mask"] = torch.ones_like(action_inputs["input_ids"])

        try:
            with torch.no_grad():
                action_outputs = gen_model.generate(
                    **action_inputs,
                    max_new_tokens=config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                    output_scores=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch action generation failed: {e}, falling back to sequential")
            results = {}
            for agent_id in agent_ids:
                results[agent_id] = generate_action(trainer, obs_dict[agent_id], agent_id, step, env, model)
            return results

        actions, action_texts, action_ids_list, action_input_ids_list = [], [], [], []
        for i in range(num_agents):
            a_ids = action_outputs.sequences[i][action_inputs.input_ids[i].shape[0]:]
            a_text = tokenizer.decode(a_ids, skip_special_tokens=True)
            a = get_action_from_response(a_text, trainer.action_words)
            actions.append(a)
            action_texts.append(a_text)
            action_ids_list.append(a_ids)
            action_input_ids_list.append(action_inputs.input_ids[i])

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_probs = compute_batch_sequence_log_prob(
                log_prob_model, action_input_ids_list, action_ids_list,
                target_device, tokenizer.pad_token_id, need_grad=False
            )
        except Exception as e:
            logger.warning(f"Batch log prob calculation failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device) for _ in range(num_agents)]

        results = {}
        for i, agent_id in enumerate(agent_ids):
            full_response = f"{thinking_texts[i]} -> {action_texts[i]}"
            results[agent_id] = (
                actions[i], log_probs[i], thinking_texts[i], full_response,
                action_texts[i], action_prompts[i],
                action_input_ids_list[i], action_ids_list[i]
            )
        return results

    # ── TEXT SINGLE-STAGE BATCH ──
    elif config.action_mode == "text" and not config.use_two_stage:
        prompts = []
        for agent_id in agent_ids:
            obs_text = obs_to_text(obs_dict[agent_id], env, agent_id, config)
            prompts.append(create_single_stage_prompt_text(obs_text, config, tokenizer))

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True
        ).to(target_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                    output_scores=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch generation failed: {e}, falling back to sequential")
            results = {}
            for agent_id in agent_ids:
                results[agent_id] = generate_action(trainer, obs_dict[agent_id], agent_id, step, env, model)
            return results

        responses, actions, gen_ids_list, input_ids_list = [], [], [], []
        for i in range(num_agents):
            gen_ids = outputs.sequences[i][inputs.input_ids[i].shape[0]:]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            a = get_action_from_response(resp, trainer.action_words)
            responses.append(resp)
            actions.append(a)
            gen_ids_list.append(gen_ids)
            input_ids_list.append(inputs.input_ids[i])

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_probs = compute_batch_sequence_log_prob(
                log_prob_model, input_ids_list, gen_ids_list,
                target_device, tokenizer.pad_token_id, need_grad=False
            )
        except Exception as e:
            logger.warning(f"Batch log prob calculation failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device) for _ in range(num_agents)]

        results = {}
        for i, agent_id in enumerate(agent_ids):
            results[agent_id] = (
                actions[i], log_probs[i], responses[i], responses[i], responses[i],
                "", input_ids_list[i], gen_ids_list[i]
            )
        return results

    # ── COMPOUND BATCH ──
    else:  # compound
        prompts = []
        for agent_id in agent_ids:
            obs_text = obs_to_text(obs_dict[agent_id], env, agent_id, config)
            prompts.append(create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id))

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True
        ).to(target_device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=config.thinking_tokens + config.action_tokens,
                    temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch generation failed: {e}, falling back to sequential")
            results = {}
            for agent_id in agent_ids:
                results[agent_id] = generate_action(trainer, obs_dict[agent_id], agent_id, step, env, model)
            return results

        responses, actions, gen_ids_list, input_ids_list = [], [], [], []
        for i, agent_id in enumerate(agent_ids):
            gen_ids = outputs.sequences[i][inputs.input_ids[i].shape[0]:]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            a = parse_and_execute_action(resp, agent_id, env)
            responses.append(resp)
            actions.append(a)
            gen_ids_list.append(gen_ids)
            input_ids_list.append(inputs.input_ids[i])

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_probs = compute_batch_sequence_log_prob(
                log_prob_model, input_ids_list, gen_ids_list,
                target_device, tokenizer.pad_token_id, need_grad=False
            )
        except Exception as e:
            logger.warning(f"Batch log prob calculation failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device) for _ in range(num_agents)]

        results = {}
        for i, agent_id in enumerate(agent_ids):
            results[agent_id] = (
                actions[i], log_probs[i], responses[i], responses[i], responses[i],
                "", input_ids_list[i], gen_ids_list[i]
            )
        return results
