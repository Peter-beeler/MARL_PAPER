"""
Prompt builders for both text mode and compound mode.

This file builds the chat-formatted prompts that are sent to the LLM.
Each function returns a string produced by tokenizer.apply_chat_template().

Read the game environment file to understand game rules, mechanics, rewards,
and available actions before implementing the prompts.
Read observation.py to see what helper actions are available (compound mode)
and what observation text looks like (both modes).
"""

from typing import Optional


# ─────────────────────────────────────────────
# TEXT MODE PROMPTS
# ─────────────────────────────────────────────

def create_thinking_prompt(obs_text: str, agent_id: int, config, tokenizer) -> str:
    """
    Stage 1 prompt (text mode two-stage): ask the agent to reason about the situation.

    Should include:
      - Game context (rules, rewards, mechanics)
      - Available low-level actions
      - Instruction to think/reason about the observation

    Args:
        obs_text: Pre-formatted observation text from observation.py.
        agent_id: Agent ID.
        config: GRPOConfig (has reward values, game settings, etc.).
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    # === TODO: implement thinking prompt ===
    raise NotImplementedError


def create_action_prompt(obs_text: str, thinking_text: str, config, tokenizer) -> str:
    """
    Stage 2 prompt (text mode two-stage): given thinking, output ONE action word.

    Should include:
      - Game context
      - The original observation
      - The thinking output from stage 1 (as assistant message)
      - Instruction to output exactly one action word

    Args:
        obs_text: Pre-formatted observation text.
        thinking_text: Generated thinking/reasoning from stage 1.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    # === TODO: implement action prompt (stage 2) ===
    raise NotImplementedError


def create_single_stage_prompt_text(obs_text: str, config, tokenizer) -> str:
    """
    Single-stage prompt (text mode): output ONE action word directly.

    Should include:
      - Game context (rules, rewards, mechanics)
      - Available low-level actions
      - Instruction to output exactly one action word

    Args:
        obs_text: Pre-formatted observation text.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    # === TODO: implement single-stage text prompt ===
    raise NotImplementedError


# ─────────────────────────────────────────────
# COMPOUND MODE PROMPTS
# ─────────────────────────────────────────────

def create_single_stage_prompt_compound(
    obs_text: str, config, tokenizer, env=None, agent_id: int = 0
) -> str:
    """
    Single-stage prompt (compound mode): think then output ONE JSON action.

    Should include:
      - Game context (rules, rewards, mechanics)
      - Documentation of available high-level actions from observation.py
        (function names, arguments, JSON format)
      - Optionally: global scan info from observation.py helpers
      - Instruction to reason briefly then output one JSON action object

    Args:
        obs_text: Natural-language observation from get_observation_description.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().
        env: Environment instance (for calling scan/info helpers).
        agent_id: Agent ID.

    Returns:
        Chat-template formatted prompt string.
    """
    # === TODO: implement compound-mode prompt ===
    raise NotImplementedError


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def build_thinking_prompt(obs_text: str, agent_id: int, config, tokenizer) -> str:
    """Build thinking prompt (text mode only; compound mode uses single-stage)."""
    return create_thinking_prompt(obs_text, agent_id, config, tokenizer)


def build_action_prompt(
    obs_text: str, thinking_text: str, config, tokenizer,
    env=None, agent_id: int = 0
) -> str:
    """
    Build action prompt based on action_mode.
    - text mode (two-stage): create_action_prompt (stage 2)
    - text mode (single-stage): create_single_stage_prompt_text
    - compound mode: create_single_stage_prompt_compound
    """
    if config.action_mode == "compound":
        return create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)
    elif config.use_two_stage:
        return create_action_prompt(obs_text, thinking_text, config, tokenizer)
    else:
        return create_single_stage_prompt_text(obs_text, config, tokenizer)
