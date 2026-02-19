"""
helpers.py — High-level helper functions for CleanupEnvMove agents.

Each function is a building block that takes (env, agent_id, ...) and
executes one or more low-level env.step() calls to accomplish a task.
Functions return bool indicating whether the task succeeded.

Functions are called via JSON:
    {"action": "move_to", "agent_id": 1, "args": {"coord_x": 5, "coord_y": 3}}

Coordinate System:
    - Internal coords: (x, y) where y=0 is the TOP row of the grid
    - Display/user coords: (x, y) where y=0 is the BOTTOM row of the grid
    - env.agents[agent_id] returns internal (x, y)
    - This module uses DISPLAY coordinates in all public interfaces
    - Helper: internal_to_display(y, height) = height - 1 - y
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any


# ===========================================================================
# COORDINATE CONVERSION UTILITIES
# ===========================================================================

def _internal_to_display_y(internal_y: int, height: int) -> int:
    """Convert internal y (row 0=top) to display y (row 0=bottom)."""
    return height - 1 - internal_y


def _display_to_internal_y(display_y: int, height: int) -> int:
    """Convert display y (row 0=bottom) to internal y (row 0=top)."""
    return height - 1 - display_y


def get_agent_display_pos(env, agent_id: int) -> Tuple[int, int]:
    """Return agent position in display coordinates (x, display_y)."""
    x, internal_y = env.agents[agent_id]
    display_y = _internal_to_display_y(internal_y, env.height)
    return (x, display_y)


# ===========================================================================
# OBSERVATION FORMATTING
# ===========================================================================

def observation_to_text(env, agent_id: int) -> str:
    """
    Convert the current environment state into a natural-language description
    for the given agent.

    Returns a string describing:
    - Agent's current position (display coords)
    - Local 5x3 window contents (from env's local render)
    - Nearby visible items: dirts and apples with display coordinates
    - All other agent positions
    - Current scores

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID (1-indexed)

    Returns:
        Natural-language observation string.
    """
    ax, ay_int = env.agents[agent_id]
    ay_disp = _internal_to_display_y(ay_int, env.height)

    lines = []
    lines.append(f"You are Agent {agent_id} at position ({ax}, {ay_disp}).")

    # Terrain at agent's position
    cell_terrain = env.terrain[ay_int][ax]
    terrain_name = "land" if cell_terrain == "*" else "river"
    lines.append(f"You are standing on {terrain_name}.")

    # Item at agent's position
    item_here = env.items[ay_int][ax]
    if item_here == "a":
        lines.append("There is an APPLE at your current position (you can eat it).")
    elif item_here == "#":
        lines.append("There is DIRT at your current position (you can clean it).")

    # Local 5x3 window: collect visible items with display coords
    half_w, half_h = 2, 1
    y0 = max(0, ay_int - half_h)
    y1 = min(env.height - 1, ay_int + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    visible_apples = []
    visible_dirts = []
    visible_agents = []

    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            disp_y = _internal_to_display_y(yy, env.height)
            item = env.items[yy][xx]
            if item == "a":
                visible_apples.append((xx, disp_y))
            elif item == "#":
                visible_dirts.append((xx, disp_y))

    for other_id, (ox, oy_int) in env.agents.items():
        if other_id == agent_id:
            continue
        oy_disp = _internal_to_display_y(oy_int, env.height)
        if x0 <= ox <= x1 and y0 <= oy_int <= y1:
            visible_agents.append((other_id, ox, oy_disp))

    # Items visible in local window
    if visible_apples:
        apple_strs = [f"({x},{y})" for x, y in visible_apples]
        lines.append(f"Visible apples nearby: {', '.join(apple_strs)}.")
    else:
        lines.append("No apples visible in your immediate area.")

    if visible_dirts:
        dirt_strs = [f"({x},{y})" for x, y in visible_dirts]
        lines.append(f"Visible dirt nearby: {', '.join(dirt_strs)}.")
    else:
        lines.append("No dirt visible in your immediate area.")

    if visible_agents:
        agent_strs = [f"Agent {aid} at ({ox},{oy})" for aid, ox, oy in visible_agents]
        lines.append(f"Nearby agents: {', '.join(agent_strs)}.")

    # Global nearest items (beyond local window)
    nearest_dirts = env._find_nearest_items(agent_id, "#", n=3)
    nearest_apples = env._find_nearest_items(agent_id, "a", n=3)

    if nearest_dirts:
        nd_strs = [f"({x},{_internal_to_display_y(y, env.height)})" for x, y in nearest_dirts]
        lines.append(f"Nearest dirt on the map: {', '.join(nd_strs)}.")
    else:
        lines.append("No dirt on the map currently.")

    if nearest_apples:
        na_strs = [f"({x},{_internal_to_display_y(y, env.height)})" for x, y in nearest_apples]
        lines.append(f"Nearest apples on the map: {', '.join(na_strs)}.")
    else:
        lines.append("No apples on the map currently.")

    # All agent positions (full map awareness)
    other_agents_info = []
    for other_id, (ox, oy_int) in env.agents.items():
        if other_id == agent_id:
            continue
        oy_disp = _internal_to_display_y(oy_int, env.height)
        other_agents_info.append(f"Agent {other_id} at ({ox},{oy_disp})")
    if other_agents_info:
        lines.append(f"All agents: {', '.join(other_agents_info)}.")

    # Current scores
    score_strs = [f"Agent {aid}: {score:.1f}" for aid, score in env.scores.items()]
    lines.append(f"Scores: {', '.join(score_strs)}.")

    # Episode progress
    lines.append(f"Step: {env.step_count}/{env.cfg.max_steps}.")

    # River/apple status summary
    dirt_count = env._count_items("#")
    apple_count = env._count_items("a")
    lines.append(
        f"Map status: {dirt_count} dirt on river, {apple_count} apples on land. "
        f"Initial dirt was {env.init_dirt_count}. "
        f"Apples spawn faster when dirt count drops below {env.init_dirt_count}."
    )

    return " ".join(lines)


# ===========================================================================
# HIGH-LEVEL ACTION FUNCTIONS
# ===========================================================================

def move_toward(env, agent_id: int, target_x: int, target_y_display: int) -> str:
    """
    Compute the single best low-level move to get closer to (target_x, target_y_display).

    Args:
        env              : CleanupEnvMove instance
        agent_id         : Integer agent ID
        target_x         : Target x in display coordinates
        target_y_display : Target y in display coordinates (y=0 at bottom)

    Returns:
        One of: "up", "down", "left", "right", "stay"
    """
    ax, ay_int = env.agents[agent_id]
    target_y_int = _display_to_internal_y(target_y_display, env.height)

    dx = target_x - ax
    dy_int = target_y_int - ay_int  # positive = move down in internal coords

    if abs(dx) >= abs(dy_int):
        if dx > 0:
            return "right"
        elif dx < 0:
            return "left"
    else:
        if dy_int > 0:
            return "down"
        elif dy_int < 0:
            return "up"
    return "stay"


def move_to(env, agent_id: int, coord_x: int, coord_y: int) -> bool:
    """
    Move agent one step closer to the target (coord_x, coord_y) in display coordinates.

    This function performs a SINGLE step. Call repeatedly to reach the destination.
    Returns True if the agent was already at the destination (no move needed),
    False if a move was issued.

    The action is applied via env.step() for ALL agents (other agents stay).

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID
        coord_x  : Target x in display coordinates
        coord_y  : Target y in display coordinates (y=0 at bottom)

    Returns:
        True if agent is already at destination, False if move was taken.
    """
    ax, ay_int = env.agents[agent_id]
    ay_disp = _internal_to_display_y(ay_int, env.height)

    if ax == coord_x and ay_disp == coord_y:
        return True

    action = move_toward(env, agent_id, coord_x, coord_y)
    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = action
    env.step(all_actions)
    return False


def eat(env, agent_id: int) -> bool:
    """
    Attempt to eat an apple at the agent's current position.

    Sends 'eat' action for this agent (all others stay).
    Returns True if there was an apple at the agent's position (may have been eaten),
    False otherwise.

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent was on an apple tile, False otherwise.
    """
    ax, ay = env.agents[agent_id]
    has_apple = env.items[ay][ax] == "a"

    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "eat"
    env.step(all_actions)
    return has_apple


def clean(env, agent_id: int) -> bool:
    """
    Attempt to clean dirt at the agent's current position.

    Sends 'clean' action for this agent (all others stay).
    Returns True if there was dirt at the agent's position,
    False otherwise.

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent was on a dirt tile, False otherwise.
    """
    ax, ay = env.agents[agent_id]
    has_dirt = env.items[ay][ax] == "#"

    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "clean"
    env.step(all_actions)
    return has_dirt


def stay(env, agent_id: int) -> bool:
    """
    Do nothing — agent stays in place for one step.

    Sends 'stay' action for this agent (all others also stay).

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        Always True.
    """
    all_actions = {i: "stay" for i in env._agent_ids}
    env.step(all_actions)
    return True


def move_up(env, agent_id: int) -> bool:
    """
    Move agent one step UP (increasing display y).

    In display coordinates, UP = toward higher y values = toward row 0 of internal grid.

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent moved (position changed), False if blocked.
    """
    ax, ay_int = env.agents[agent_id]
    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "up"
    env.step(all_actions)
    new_x, new_y_int = env.agents[agent_id]
    return (new_x, new_y_int) != (ax, ay_int)


def move_down(env, agent_id: int) -> bool:
    """
    Move agent one step DOWN (decreasing display y).

    In display coordinates, DOWN = toward lower y values = toward last row of internal grid.

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent moved (position changed), False if blocked.
    """
    ax, ay_int = env.agents[agent_id]
    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "down"
    env.step(all_actions)
    new_x, new_y_int = env.agents[agent_id]
    return (new_x, new_y_int) != (ax, ay_int)


def move_left(env, agent_id: int) -> bool:
    """
    Move agent one step to the LEFT (decreasing x).

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent moved (position changed), False if blocked.
    """
    ax, ay_int = env.agents[agent_id]
    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "left"
    env.step(all_actions)
    new_x, new_y_int = env.agents[agent_id]
    return (new_x, new_y_int) != (ax, ay_int)


def move_right(env, agent_id: int) -> bool:
    """
    Move agent one step to the RIGHT (increasing x).

    Args:
        env      : CleanupEnvMove instance
        agent_id : Integer agent ID

    Returns:
        True if the agent moved (position changed), False if blocked.
    """
    ax, ay_int = env.agents[agent_id]
    all_actions = {i: "stay" for i in env._agent_ids}
    all_actions[agent_id] = "right"
    env.step(all_actions)
    new_x, new_y_int = env.agents[agent_id]
    return (new_x, new_y_int) != (ax, ay_int)


# ===========================================================================
# DISPATCH TABLE
# ===========================================================================

# Maps action name strings to callable functions
ACTION_REGISTRY: Dict[str, Any] = {
    "move_to":    move_to,
    "eat":        eat,
    "clean":      clean,
    "stay":       stay,
    "move_up":    move_up,
    "move_down":  move_down,
    "move_left":  move_left,
    "move_right": move_right,
}


def dispatch_action(env, action_json: dict) -> bool:
    """
    Execute a helper action from a parsed JSON dict.

    Expected format:
        {
            "action": "move_to",
            "agent_id": 1,
            "args": {"coord_x": 5, "coord_y": 3}
        }

    Args:
        env         : CleanupEnvMove instance
        action_json : Parsed action dictionary

    Returns:
        Result of the called function (bool).

    Raises:
        ValueError if action name is unknown.
    """
    action_name = action_json.get("action", "stay")
    agent_id    = action_json.get("agent_id", 1)
    args        = action_json.get("args", {})

    if action_name not in ACTION_REGISTRY:
        raise ValueError(f"Unknown action: '{action_name}'. Valid: {list(ACTION_REGISTRY.keys())}")

    fn = ACTION_REGISTRY[action_name]
    return fn(env, agent_id, **args)


def parse_action_json(response_text: str) -> Optional[dict]:
    """
    Extract and parse a JSON action dict from model output text.

    Handles nested JSON objects (e.g., args dict with coord_x, coord_y).
    Tries all candidate JSON substrings from last to first.

    Args:
        response_text : Raw model output string.

    Returns:
        Parsed dict with 'action', 'agent_id', 'args' keys, or None on failure.
    """
    import json

    # Find all '{' positions and try to parse from there
    candidates = []
    for i, ch in enumerate(response_text):
        if ch == '{':
            # Find matching closing brace
            depth = 0
            for j in range(i, len(response_text)):
                if response_text[j] == '{':
                    depth += 1
                elif response_text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(response_text[i:j+1])
                        break

    # Try from last candidate first (model output at end)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue

    return None
