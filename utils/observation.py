"""
Observation-to-text conversion and high-level action helpers.

This file contains two categories of functions:
  1. Observation converters — turn raw env state into natural-language text
  2. High-level actions — callable helpers that map to low-level env actions

Both text mode and compound mode need their own implementations.
Read the game environment file to understand the grid layout, items, agents,
coordinate system, and available low-level actions before implementing.
"""

import random
from typing import Any, Dict, Tuple


# ─────────────────────────────────────────────
# TEXT MODE: observation converter
# ─────────────────────────────────────────────

def parse_observation_to_coords(obs: str, agent_id: int, env) -> str:
    """
    Convert the environment state into a coordinate-based text description
    for text mode (local observation window).

    Args:
        obs: Raw observation string (may be unused — env is the source of truth).
        agent_id: Agent ID.
        env: Game environment instance.

    Returns:
        Natural-language string describing the agent's position and nearby objects/agents.
    """
    # === TODO: implement text-mode observation-to-text ===
    raise NotImplementedError


# ─────────────────────────────────────────────
# COMPOUND MODE: observation converter
# ─────────────────────────────────────────────

def get_observation_description(env: Any, agent_id: int) -> str:
    """
    Generate a natural-language description of the agent's surroundings
    for compound mode.

    Args:
        env: Game environment instance.
        agent_id: Agent ID.

    Returns:
        Natural-language string describing position and visible objects/agents.
    """
    # === TODO: implement compound-mode observation-to-text ===
    raise NotImplementedError


# ─────────────────────────────────────────────
# COMPOUND MODE: global scan helpers
# ─────────────────────────────────────────────
# Add any global-scan or information-gathering helpers here.
# These are NOT actions — they provide extra context to the prompt builder.
# Each should return a dict with at least a 'found' (bool) key.
#
# === TODO: implement scan/info helpers relevant to your game ===


# ─────────────────────────────────────────────
# HIGH-LEVEL ACTIONS (used by compound mode)
# ─────────────────────────────────────────────
# Each high-level action is called via JSON from the LLM:
#   {"action": "<name>", "agent_id": 1, "args": {"key": value, ...}}
#
# Every action function must follow this signature:
#   def action_name(env, agent_id, **kwargs) -> Tuple[str, bool]
# and return:
#   (low_level_action_string, is_done)
#
# === TODO: implement high-level actions relevant to your game ===


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def obs_to_text(obs: str, env, agent_id: int, config) -> str:
    """
    Dispatch observation-to-text conversion based on action_mode.

    Args:
        obs: Raw observation string.
        env: Game environment instance.
        agent_id: Agent ID.
        config: GRPOConfig.

    Returns:
        Natural-language observation string.
    """
    if config.action_mode == "compound":
        return get_observation_description(env, agent_id)
    else:
        return parse_observation_to_coords(obs, agent_id, env)
