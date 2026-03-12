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
# === TODO: observation.py and prompts.py must be implemented before this file works ===
from .observation import (
    obs_to_text, get_role_specific_observation, check_action_continuation,
    pop_tool_result, TOOL_ACTIONS,
)
from .prompts import (
    create_thinking_prompt, create_action_prompt,
    create_single_stage_prompt_text, create_single_stage_prompt_compound,
    create_role_compound_prompt,
)
# Action dispatch uses dynamic lookup — no need to import individual actions
from . import observation as _obs_module

logger = logging.getLogger(__name__)

ACTION_WORDS = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']


# ─────────────────────────────────────────────
# CHUNKED HELPERS (for parallel multi-env rollout)
# ─────────────────────────────────────────────

def _chunked_generate(gen_model, tokenizer, prompts, max_new_tokens, config, device, macro_batch):
    """
    Run model.generate() on a list of prompts, chunked by macro_batch.

    Returns:
        (input_ids_list, generated_ids_list, generated_texts_list)
        All lists have length == len(prompts).
    """
    all_input_ids = []
    all_gen_ids = []
    all_gen_texts = []

    for start in range(0, len(prompts), macro_batch):
        chunk = prompts[start:start + macro_batch]
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=config.max_length, padding=True
        ).to(device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        for j in range(len(chunk)):
            gen_ids = outputs.sequences[j][inputs.input_ids[j].shape[0]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_input_ids.append(inputs.input_ids[j])
            all_gen_ids.append(gen_ids)
            all_gen_texts.append(gen_text)

    return all_input_ids, all_gen_ids, all_gen_texts


def _chunked_log_probs(model, input_ids_list, gen_ids_list, device, pad_token_id, macro_batch):
    """Compute log probs in chunks to limit memory."""
    all_log_probs = []
    for start in range(0, len(input_ids_list), macro_batch):
        chunk_inputs = input_ids_list[start:start + macro_batch]
        chunk_gens = gen_ids_list[start:start + macro_batch]
        lps = compute_batch_sequence_log_prob(
            model, chunk_inputs, chunk_gens,
            device, pad_token_id, need_grad=False
        )
        all_log_probs.extend([lp for lp in lps])
    return all_log_probs


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

def _get_fallback_action(env, agent_id) -> str:
    """Return a fallback action when JSON parsing fails. Uses 'stay' by default."""
    # === TODO: change this if your game has a different default/fallback action ===
    return "stay"


def parse_and_execute_action(response: str, agent_id: int, env) -> str:
    """
    Parse JSON from model response, call the appropriate helper, return low-level env action.

    Expected JSON format from LLM:
        {"action": "<name>", "agent_id": N, "args": {"key": value, ...}}

    Returns:
        Low-level action string understood by env.step().
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
            logger.debug(f"No JSON found in response: '{response[:80]}', using fallback")
            return _get_fallback_action(env, agent_id)
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
            logger.debug(f"Unbalanced braces in response: '{response[:80]}', using fallback")
            return _get_fallback_action(env, agent_id)
        json_str = response[start:end + 1]

    try:
        data = json.loads(json_str)
        action_name = data.get('action', '')
        args = data.get('args', {})

        # === TODO: this dispatch calls your high-level actions from observation.py ===
        # === TODO: update it to match the actions you defined there ===
        action_fn = getattr(_obs_module, action_name, None)
        if action_fn is not None and callable(action_fn):
            # Convert args values to int where possible (coords are ints)
            typed_args = {}
            for k, v in args.items():
                try:
                    typed_args[k] = int(v)
                except (ValueError, TypeError):
                    typed_args[k] = v
            action, _ = action_fn(env, agent_id, **typed_args)
        else:
            logger.debug(f"Unknown action name: '{action_name}', using fallback")
            action = _get_fallback_action(env, agent_id)

        return action

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Failed to parse JSON action: {e}, using fallback")
        return _get_fallback_action(env, agent_id)


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

    # ── COMPOUND BATCH (single env, kept for backward compat) ──
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


# ─────────────────────────────────────────────
# MULTI-ENV BATCHED GENERATION (parallel rollout)
# ─────────────────────────────────────────────

def _fallback_per_env(trainer, active_env_data, step, model):
    """Fall back to per-env sequential generation when batched generation fails."""
    results = {}
    for env_idx, env, obs_dict in active_env_data:
        results[env_idx] = generate_actions_batch(trainer, obs_dict, step, env, model)
    return results


def generate_actions_multi_env_batch(
    trainer, active_env_data, step, model, macro_infer_batch=24
):
    """
    Generate actions for all agents across multiple environments with chunked batching.

    Flattens all (env, agent) pairs into a single prompt list, runs model.generate()
    in chunks of macro_infer_batch, then reassembles results per env.

    Args:
        trainer: CleanupGameGRPO instance.
        active_env_data: list of (env_idx, env, obs_dict) tuples for active envs.
        step: Current env step.
        model: Model to use for generation.
        macro_infer_batch: Max prompts per model.generate() call.

    Returns:
        dict mapping env_idx -> dict mapping agent_id -> 8-tuple
    """
    config = trainer.config
    tokenizer = trainer.tokenizer
    accelerator = trainer.accelerator
    device = trainer.device

    target_device = device
    _is_zero3 = (
        accelerator is not None and
        getattr(getattr(accelerator, 'state', None), 'deepspeed_plugin', None) is not None
    )
    gen_model = model if _is_zero3 else (
        model.module if (accelerator is not None and hasattr(model, 'module')) else model
    )

    # Build flat list: (env_idx, agent_id, obs_text, env, role_or_None)
    # When role switching is active, use role-specific observations with
    # tool results and continuation hints from the previous step.
    role_states = getattr(trainer, '_role_states', None)
    flat_meta = []
    for env_idx, env, obs_dict in active_env_data:
        for agent_id in sorted(obs_dict.keys()):
            role = None
            if role_states is not None and env_idx < len(role_states):
                rs = role_states[env_idx]
                role = rs.get('roles', {}).get(agent_id)
            if role and config.action_mode == "compound":
                ot = get_role_specific_observation(env, agent_id, role)
                # Append tool results from previous step
                tool_res = role_states[env_idx].get('tool_results', {}).get(agent_id, '')
                if tool_res:
                    ot += f"\n[TOOL RESULTS] {tool_res}"
                # Append continuation hints
                last_act = role_states[env_idx].get('last_actions', {}).get(agent_id)
                cont_hint = check_action_continuation(env, agent_id, last_act)
                if cont_hint:
                    ot += f"\n[CONTINUE] {cont_hint}"
            else:
                ot = obs_to_text(obs_dict[agent_id], env, agent_id, config)
            flat_meta.append((env_idx, agent_id, ot, env, role))

    N = len(flat_meta)
    if N == 0:
        return {}

    # ── TEXT TWO-STAGE ──
    if config.action_mode == "text" and config.use_two_stage:
        thinking_prompts = [
            create_thinking_prompt(m[2], m[1], config, tokenizer) for m in flat_meta
        ]

        try:
            thinking_input_ids, thinking_gen_ids, thinking_texts = _chunked_generate(
                gen_model, tokenizer, thinking_prompts,
                config.thinking_tokens, config, target_device, macro_infer_batch
            )
        except RuntimeError as e:
            logger.warning(f"Multi-env batch thinking failed: {e}, falling back to per-env")
            return _fallback_per_env(trainer, active_env_data, step, model)

        action_prompts = [
            create_action_prompt(flat_meta[i][2], thinking_texts[i], config, tokenizer)
            for i in range(N)
        ]

        try:
            action_input_ids, action_gen_ids, action_texts = _chunked_generate(
                gen_model, tokenizer, action_prompts,
                config.action_tokens, config, target_device, macro_infer_batch
            )
        except RuntimeError as e:
            logger.warning(f"Multi-env batch action failed: {e}, falling back to per-env")
            return _fallback_per_env(trainer, active_env_data, step, model)

        actions = [get_action_from_response(t, trainer.action_words) for t in action_texts]

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            if config.logprob_mode == "action+thinking":
                thinking_lps = _chunked_log_probs(
                    log_prob_model, thinking_input_ids, thinking_gen_ids,
                    target_device, tokenizer.pad_token_id, macro_infer_batch
                )
                action_lps = _chunked_log_probs(
                    log_prob_model, action_input_ids, action_gen_ids,
                    target_device, tokenizer.pad_token_id, macro_infer_batch
                )
                log_probs = [t_lp + a_lp for t_lp, a_lp in zip(thinking_lps, action_lps)]
            else:
                log_probs = _chunked_log_probs(
                    log_prob_model, action_input_ids, action_gen_ids,
                    target_device, tokenizer.pad_token_id, macro_infer_batch
                )
        except Exception as e:
            logger.warning(f"Multi-env batch log prob failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device)] * N

        results = {}
        for i in range(N):
            env_idx, agent_id = flat_meta[i][0], flat_meta[i][1]
            if env_idx not in results:
                results[env_idx] = {}
            full_response = f"{thinking_texts[i]} -> {action_texts[i]}"
            results[env_idx][agent_id] = (
                actions[i], log_probs[i], thinking_texts[i], full_response,
                action_texts[i], action_prompts[i],
                action_input_ids[i], action_gen_ids[i]
            )
        return results

    # ── TEXT SINGLE-STAGE ──
    elif config.action_mode == "text" and not config.use_two_stage:
        prompts = [
            create_single_stage_prompt_text(m[2], config, tokenizer) for m in flat_meta
        ]

        try:
            input_ids, gen_ids, gen_texts = _chunked_generate(
                gen_model, tokenizer, prompts,
                config.action_tokens, config, target_device, macro_infer_batch
            )
        except RuntimeError as e:
            logger.warning(f"Multi-env batch generation failed: {e}, falling back to per-env")
            return _fallback_per_env(trainer, active_env_data, step, model)

        actions = [get_action_from_response(t, trainer.action_words) for t in gen_texts]

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_probs = _chunked_log_probs(
                log_prob_model, input_ids, gen_ids,
                target_device, tokenizer.pad_token_id, macro_infer_batch
            )
        except Exception as e:
            logger.warning(f"Multi-env batch log prob failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device)] * N

        results = {}
        for i in range(N):
            env_idx, agent_id = flat_meta[i][0], flat_meta[i][1]
            if env_idx not in results:
                results[env_idx] = {}
            results[env_idx][agent_id] = (
                actions[i], log_probs[i], gen_texts[i], gen_texts[i], gen_texts[i],
                "", input_ids[i], gen_ids[i]
            )
        return results

    # ── COMPOUND ──
    else:
        prompts = []
        for m in flat_meta:
            env_idx, agent_id, ot, env_ref, role = m
            if role:
                prompts.append(create_role_compound_prompt(ot, config, tokenizer, env_ref, agent_id, role))
            else:
                prompts.append(create_single_stage_prompt_compound(ot, config, tokenizer, env_ref, agent_id))

        try:
            input_ids, gen_ids, gen_texts = _chunked_generate(
                gen_model, tokenizer, prompts,
                config.thinking_tokens + config.action_tokens,
                config, target_device, macro_infer_batch
            )
        except RuntimeError as e:
            logger.warning(f"Multi-env batch generation failed: {e}, falling back to per-env")
            return _fallback_per_env(trainer, active_env_data, step, model)

        # Parse and execute per-env (CPU-only, fast)
        actions = []
        for i in range(N):
            a = parse_and_execute_action(gen_texts[i], flat_meta[i][1], flat_meta[i][3])
            actions.append(a)

        try:
            log_prob_model = trainer.old_model if trainer.old_model is not None else gen_model
            log_probs = _chunked_log_probs(
                log_prob_model, input_ids, gen_ids,
                target_device, tokenizer.pad_token_id, macro_infer_batch
            )
        except Exception as e:
            logger.warning(f"Multi-env batch log prob failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device)] * N

        results = {}
        for i in range(N):
            env_idx, agent_id = flat_meta[i][0], flat_meta[i][1]
            if env_idx not in results:
                results[env_idx] = {}
            results[env_idx][agent_id] = (
                actions[i], log_probs[i], gen_texts[i], gen_texts[i], gen_texts[i],
                "", input_ids[i], gen_ids[i]
            )

        # Collect tool results from tool actions executed during parse_and_execute_action
        if role_states is not None:
            for i in range(N):
                env_idx, agent_id = flat_meta[i][0], flat_meta[i][1]
                tool_res = pop_tool_result(flat_meta[i][3], agent_id)
                if env_idx < len(role_states):
                    role_states[env_idx].setdefault('tool_results', {})[agent_id] = tool_res

        return results
