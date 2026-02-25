"""
Observation-to-text functions for both text mode (coordinate-based local window)
and compound mode (global scan with nearest-item helpers).
"""

import random
from typing import Any, Dict, Tuple


# ─────────────────────────────────────────────
# TEXT MODE
# ─────────────────────────────────────────────

def parse_observation_to_coords(obs: str, agent_id: int, env) -> str:
    """
    Convert local observation to coordinate-based format (text mode).

    Transforms internal coordinates (y=0 at top) to display coordinates (y=0 at bottom).

    Args:
        obs: Raw observation string (unused - env is used directly).
        agent_id: Agent ID.
        env: CleanupEnvMove instance.

    Returns:
        Natural-language string like "You at (2,3). Dirt at (2,4). Apple at (1,3)."
    """
    agent_pos = env.agents[agent_id]
    ax, ay_internal = agent_pos
    ay_display = (env.height - 1) - ay_internal

    half_w, half_h = 2, 1
    y0 = max(0, ay_internal - half_h)
    y1 = min(env.height - 1, ay_internal + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    dirt_coords = []
    apple_coords = []

    for y_internal in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if env.items[y_internal][x] == '#':
                y_display = (env.height - 1) - y_internal
                dirt_coords.append((x, y_display))
            elif env.items[y_internal][x] == 'a':
                y_display = (env.height - 1) - y_internal
                apple_coords.append((x, y_display))

    other_agents_info = []
    for other_id, (ox, oy_internal) in env.agents.items():
        if other_id == agent_id:
            continue
        if x0 <= ox <= x1 and y0 <= oy_internal <= y1:
            oy_display = (env.height - 1) - oy_internal
            if hasattr(env, 'get_agent_movement'):
                direction, (prev_x, prev_y) = env.get_agent_movement(other_id)
                if direction == "STAYED":
                    other_agents_info.append(f"Agent {other_id} at ({ox},{oy_display}) [stayed]")
                else:
                    other_agents_info.append(
                        f"Agent {other_id} at ({ox},{oy_display}) [moved {direction} from ({prev_x},{prev_y})]"
                    )
            else:
                other_agents_info.append(f"Agent {other_id} at ({ox},{oy_display})")

    obs_text = f"You at ({ax},{ay_display})."
    if dirt_coords:
        obs_text += f" Dirt at {', '.join([f'({x},{y})' for x, y in dirt_coords])}."
    if apple_coords:
        obs_text += f" Apple at {', '.join([f'({x},{y})' for x, y in apple_coords])}."
    if other_agents_info:
        obs_text += " " + " ".join(other_agents_info) + "."
    if not dirt_coords and not apple_coords and not other_agents_info:
        obs_text += " Nothing in your view."

    return obs_text


# ─────────────────────────────────────────────
# COMPOUND MODE: helpers (from archive/helpers.py)
# ─────────────────────────────────────────────

def get_observation_description(env: Any, agent_id: int) -> str:
    """
    Generate a natural-language description of the agent's position and visible objects
    within its local observation window (5x3). Used by compound mode.
    """
    if agent_id not in env.agents:
        return "You are not in the environment."

    ax, ay = env.agents[agent_id]

    half_w, half_h = 2, 1
    x_min = max(0, ax - half_w)
    x_max = min(env.width - 1, ax + half_w)
    y_min = max(0, ay - half_h)
    y_max = min(env.height - 1, ay + half_h)

    visible_objects = []

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            item = env.items[y][x]
            if item == 'a':
                visible_objects.append(f"an apple at ({x}, {y})")
            elif item == '#':
                visible_objects.append(f"dirt at ({x}, {y})")

            for other_id, pos in env.agents.items():
                if other_id != agent_id and pos == (x, y):
                    visible_objects.append(f"agent {other_id} at ({x}, {y})")

    desc = f"You are at ({ax}, {ay})."
    if not visible_objects:
        desc += " You see nothing of interest nearby."
    else:
        if len(visible_objects) == 1:
            desc += f" You see {visible_objects[0]}."
        else:
            joined = ", ".join(visible_objects[:-1]) + f" and {visible_objects[-1]}"
            desc += f" You see {joined}."

    return desc


def find_nearest_apple(env: Any, agent_id: int) -> Dict:
    """Locate the nearest apple ('a') globally. Returns dict with found/coord_x/coord_y/distance."""
    if agent_id not in env.agents:
        return {'found': False}

    ax, ay = env.agents[agent_id]
    nearest_dist = float('inf')
    nearest_pos = None

    for y in range(env.height):
        for x in range(env.width):
            if env.items[y][x] == 'a':
                dist = abs(x - ax) + abs(y - ay)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pos = (x, y)

    if nearest_pos:
        return {'found': True, 'coord_x': nearest_pos[0], 'coord_y': nearest_pos[1], 'distance': nearest_dist}
    return {'found': False}


def find_nearest_dirt(env: Any, agent_id: int) -> Dict:
    """Locate the nearest dirt ('#') globally. Returns dict with found/coord_x/coord_y/distance."""
    if agent_id not in env.agents:
        return {'found': False}

    ax, ay = env.agents[agent_id]
    nearest_dist = float('inf')
    nearest_pos = None

    for y in range(env.height):
        for x in range(env.width):
            if env.items[y][x] == '#':
                dist = abs(x - ax) + abs(y - ay)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pos = (x, y)

    if nearest_pos:
        return {'found': True, 'coord_x': nearest_pos[0], 'coord_y': nearest_pos[1], 'distance': nearest_dist}
    return {'found': False}


def move_to(env: Any, agent_id: int, coord_x: int, coord_y: int) -> Tuple[str, bool]:
    """
    Calculate the next step to move towards a target coordinate.
    Returns (action_string, is_done).
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]
    if curr_x == coord_x and curr_y == coord_y:
        return "stay", True

    if curr_x < coord_x:
        return "right", False
    elif curr_x > coord_x:
        return "left", False
    elif curr_y < coord_y:
        return "down", False
    elif curr_y > coord_y:
        return "up", False

    return "stay", False


def clean_at(env: Any, agent_id: int, coord_x: int, coord_y: int) -> Tuple[str, bool]:
    """
    Move to the target coordinate and clean it.
    Returns (action_string, is_done).
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]
    if (curr_x, curr_y) != (coord_x, coord_y):
        if curr_x < coord_x:
            return "right", False
        elif curr_x > coord_x:
            return "left", False
        elif curr_y < coord_y:
            return "down", False
        elif curr_y > coord_y:
            return "up", False

    item = env.items[curr_y][curr_x]
    if item == '#':
        return "clean", False
    else:
        return "stay", True


def eat_at(env: Any, agent_id: int, coord_x: int, coord_y: int) -> Tuple[str, bool]:
    """
    Move to the target coordinate and eat an apple.
    Returns (action_string, is_done).
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]
    if (curr_x, curr_y) != (coord_x, coord_y):
        if curr_x < coord_x:
            return "right", False
        elif curr_x > coord_x:
            return "left", False
        elif curr_y < coord_y:
            return "down", False
        elif curr_y > coord_y:
            return "up", False

    item = env.items[curr_y][curr_x]
    if item == 'a':
        return "eat", False
    else:
        return "stay", True


def random_explore(env: Any, agent_id: int) -> Tuple[str, bool]:
    """Return a random movement action. is_done is always False."""
    actions = ["up", "down", "left", "right"]
    return random.choice(actions), False


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def obs_to_text(obs: str, env, agent_id: int, config) -> str:
    """
    Dispatch observation-to-text conversion based on action_mode.

    Args:
        obs: Raw observation string.
        env: CleanupEnvMove instance.
        agent_id: Agent ID.
        config: GRPOConfig.

    Returns:
        Natural-language observation string.
    """
    if config.action_mode == "compound":
        return get_observation_description(env, agent_id)
    else:
        return parse_observation_to_coords(obs, agent_id, env)
