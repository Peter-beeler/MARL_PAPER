"""
prompt_template.py — LLM prompt builders for CleanupEnvMove GRPO training.

Two-stage generation:
  Stage 1 (create_thinking_prompt):  Ask agent to reason about the situation.
  Stage 2 (create_single_stage_prompt): Given the reasoning, output ONE JSON action.

Both functions return List[Dict] with 'role'/'content' keys compatible with
tokenizer.apply_chat_template().
"""

from __future__ import annotations
from typing import List, Dict

# ===========================================================================
# GAME CONTEXT
# ===========================================================================

GAME_CONTEXT = """
GAME: Cooperative Cleanup — a 2D grid-world where agents must clean a river
and harvest apples to score points.

MAP LAYOUT:
- Grid is 15 columns (x: 0-14) wide and 9 rows (y: 0-8 in display coordinates) tall.
- Display coordinates: x=0 is the left edge, y=0 is the BOTTOM edge.
- A 3-column wide river runs vertically through the CENTER of the map (columns 6, 7, 8).
- Everything left of the river (columns 0-5) and right (columns 9-14) is land.

CELL TYPES:
- Land (*): Where apples spawn and agents can walk freely.
- River (x): Where dirt spawns. Agents can walk on the river too.
- Apple (a): A collectible item on land. Eat it for +1.0 reward.
- Dirt (#): Pollutant on the river. Clean it to help apples spawn.

MECHANICS:
- Clean dirt (#) on the river: no immediate reward, but reduces pollution.
  Apples start spawning on land when dirt count drops below the initial dirt count.
  The fewer the remaining dirts, the faster apples appear.
- Eat apple (a) on land: +1.0 reward immediately.
- Agents cannot stack on the same cell (collision = both agents stay).

AGENT ROLES (cooperation is key):
- CLEANER role: Go to the river, find dirt cells, clean them.
  Strategy: navigate to a river cell with dirt, then use 'clean'.
- HARVESTER role: Stay on land, navigate to apple cells, eat them.
  Strategy: navigate toward the nearest apple, then use 'eat'.
- The team needs both cleaners AND harvesters to score well.

COORDINATE SYSTEM (IMPORTANT):
- All coordinates in action calls use DISPLAY coordinates.
- x: 0=left, increases right. y: 0=BOTTOM, increases upward.
- Example: position (7, 4) is in the center of the map.

EPISODE:
- Each episode has up to 30 steps (training) or 200 steps (full game).
- Your score accumulates from eating apples throughout the episode.
- Multiple agents (3 by default) act simultaneously.
""".strip()

AVAILABLE_ACTIONS = """
AVAILABLE ACTIONS (respond with exactly ONE JSON object):

1. move_to — Move one step toward a target cell.
   {"action": "move_to", "agent_id": <id>, "args": {"coord_x": <x>, "coord_y": <y>}}
   Note: Takes ONE step toward target. Call again next turn if not there yet.

2. eat — Eat an apple at your current position.
   {"action": "eat", "agent_id": <id>, "args": {}}
   Use when: you are standing on an apple cell.

3. clean — Clean dirt at your current position.
   {"action": "clean", "agent_id": <id>, "args": {}}
   Use when: you are standing on a dirt cell (river).

4. stay — Wait in place for one step.
   {"action": "stay", "agent_id": <id>, "args": {}}

5. move_up — Move one step upward (increasing y).
   {"action": "move_up", "agent_id": <id>, "args": {}}

6. move_down — Move one step downward (decreasing y).
   {"action": "move_down", "agent_id": <id>, "args": {}}

7. move_left — Move one step left (decreasing x).
   {"action": "move_left", "agent_id": <id>, "args": {}}

8. move_right — Move one step right (increasing x).
   {"action": "move_right", "agent_id": <id>, "args": {}}
""".strip()

RESPONSE_FORMAT = """
RESPONSE FORMAT:
Output EXACTLY ONE JSON object. No explanation, no extra text. Examples:
  {"action": "move_to", "agent_id": 1, "args": {"coord_x": 7, "coord_y": 5}}
  {"action": "eat", "agent_id": 2, "args": {}}
  {"action": "clean", "agent_id": 1, "args": {}}
  {"action": "move_right", "agent_id": 3, "args": {}}
""".strip()

STRATEGY_HINTS = """
STRATEGY TIPS:
- If you are on an APPLE: use 'eat' immediately for +1.0 reward.
- If you are on DIRT (river): use 'clean' to help the team get apples.
- If there are apples nearby on land: navigate to them using 'move_to'.
- If there is a lot of dirt and no apples: prioritize cleaning the river.
- Avoid colliding with other agents (they block your path).
- The river is columns 6-8. Land is columns 0-5 and 9-14.
""".strip()


# ===========================================================================
# PROMPT BUILDERS
# ===========================================================================

def create_thinking_prompt(obs_text: str, agent_id: int) -> List[Dict[str, str]]:
    """
    Build the Stage-1 (thinking/reasoning) prompt for an agent.

    The model is asked to reason about the current situation before
    deciding on an action. Its output will be used in Stage 2.

    Args:
        obs_text : Natural-language observation from observation_to_text().
        agent_id : Integer agent ID (1-indexed).

    Returns:
        List of dicts with 'role' and 'content' keys, suitable for
        tokenizer.apply_chat_template().
    """
    system_content = (
        f"You are Agent {agent_id} in a cooperative grid-world game.\n\n"
        f"{GAME_CONTEXT}\n\n"
        f"{AVAILABLE_ACTIONS}\n\n"
        f"{STRATEGY_HINTS}\n\n"
        "Your task is to think carefully about the current situation and "
        "decide on the best action. Consider: where are the nearest apples "
        "and dirt? Should you clean the river or harvest apples? What is the "
        "most effective next step for the team?"
    )

    user_content = (
        f"CURRENT SITUATION (Agent {agent_id}):\n{obs_text}\n\n"
        "Think step by step about what you should do next. "
        "Consider your position, nearby items, and what will maximize the team's score. "
        "Reason about whether you should be a cleaner or a harvester right now."
    )

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]


def create_single_stage_prompt(
    obs_text: str,
    thinking_response: str,
    agent_id: int,
) -> List[Dict[str, str]]:
    """
    Build the Stage-2 (action selection) prompt for an agent.

    Takes the agent's reasoning from Stage 1 and asks it to commit
    to a single JSON action.

    Args:
        obs_text          : Natural-language observation from observation_to_text().
        thinking_response : The model's Stage-1 reasoning text.
        agent_id          : Integer agent ID (1-indexed).

    Returns:
        List of dicts with 'role' and 'content' keys.
    """
    system_content = (
        f"You are Agent {agent_id} in a cooperative grid-world game.\n\n"
        f"{GAME_CONTEXT}\n\n"
        f"{AVAILABLE_ACTIONS}\n\n"
        f"{RESPONSE_FORMAT}"
    )

    user_content = (
        f"CURRENT SITUATION (Agent {agent_id}):\n{obs_text}\n\n"
        f"Think step by step about what you should do next. "
        f"Consider your position, nearby items, and what will maximize the team's score."
    )

    action_instruction = (
        "Based on your reasoning, output EXACTLY ONE JSON action object. "
        "No explanation — just the JSON.\n"
        f"Remember: you are Agent {agent_id}. "
        "Use the correct agent_id in your JSON.\n"
        f"{RESPONSE_FORMAT}"
    )

    return [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": thinking_response},
        {"role": "user",      "content": action_instruction},
    ]


# ===========================================================================
# LEGACY / CONVENIENCE ALIASES
# ===========================================================================

def create_action_prompt(obs_text: str, thinking_text: str, agent_id: int = 1) -> List[Dict[str, str]]:
    """Alias for create_single_stage_prompt (matches template interface)."""
    return create_single_stage_prompt(obs_text, thinking_text, agent_id)
