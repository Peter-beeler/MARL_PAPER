"""
prompt_template.py — LLM prompt builders for CleanupEnvMove agents.

Provides two functions that return list[dict] with role/content keys:
  - create_thinking_prompt(obs_text, agent_id)  — Stage 1: reason about situation
  - create_single_stage_prompt(obs_text, thinking_response, agent_id) — Stage 2: pick action
"""

from __future__ import annotations

from typing import List, Dict


# ---------------------------------------------------------------------------
# Game context (shared across prompts)
# ---------------------------------------------------------------------------

GAME_CONTEXT = """\
You are playing a cooperative cleanup game on a 15x9 grid.

MAP LAYOUT:
- Land (*) on both sides, river (water) running down the center (3 columns: x=6,7,8).
- Dirt (#) appears on water cells. Apples (a) appear on land cells.

RULES:
- Cleaning dirt from the river enables apple spawning on land.
- Apples only spawn once dirt count drops below the initial amount.
- Eating an apple gives +1.0 reward. Cleaning gives no immediate reward but helps the team.
- You see a 5x3 local window around yourself.

COORDINATES: x: 0 (left) to 14 (right), y: 0 (bottom) to 8 (top)."""


# ---------------------------------------------------------------------------
# Available actions (simple single-word actions)
# ---------------------------------------------------------------------------

ACTION_WORDS = ["up", "down", "left", "right", "eat", "clean", "stay"]

ACTIONS_DESCRIPTION = """\
AVAILABLE ACTIONS (output exactly ONE word):
- up: Move one step up (y+1)
- down: Move one step down (y-1)
- left: Move one step left (x-1)
- right: Move one step right (x+1)
- eat: Eat apple at your current position (only works if apple is here)
- clean: Clean dirt at your current position (only works if dirt is here)
- stay: Do nothing"""


# ---------------------------------------------------------------------------
# Stage 1: Thinking prompt
# ---------------------------------------------------------------------------

def create_thinking_prompt(obs_text: str, agent_id: int) -> List[Dict[str, str]]:
    """Build the Stage-1 prompt asking the agent to reason about the situation.

    Args:
        obs_text: Natural-language observation from observation_to_text().
        agent_id: The agent's integer ID.

    Returns:
        list of dicts with 'role' and 'content' keys.
    """
    system_msg = (
        f"{GAME_CONTEXT}\n\n"
        f"{ACTIONS_DESCRIPTION}\n\n"
        f"You are Agent {agent_id}. Think step-by-step:\n"
        "1. What items do you see at your position and nearby?\n"
        "2. Should you eat/clean something here, or move toward an item?\n"
        "3. If moving, which direction gets you closer to a useful item?\n"
        "4. Cleaning dirt helps everyone by enabling apple spawns."
    )

    user_msg = f"{obs_text}\n\nThink about your best action."

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Stage 2: Action prompt (given thinking)
# ---------------------------------------------------------------------------

def create_single_stage_prompt(
    obs_text: str,
    thinking_response: str,
    agent_id: int,
) -> List[Dict[str, str]]:
    """Build the Stage-2 prompt: given reasoning, output ONE action word.

    Args:
        obs_text: Natural-language observation from observation_to_text().
        thinking_response: The model's Stage-1 reasoning text.
        agent_id: The agent's integer ID.

    Returns:
        list of dicts with 'role' and 'content' keys.
    """
    system_msg = (
        f"{GAME_CONTEXT}\n\n"
        f"{ACTIONS_DESCRIPTION}"
    )

    user_msg = obs_text

    action_instruction = (
        "Based on your reasoning, output ONLY ONE action word from: "
        "up, down, left, right, eat, clean, stay"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": thinking_response},
        {"role": "user", "content": action_instruction},
    ]
