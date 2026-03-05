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
    ax, ay = env.agents[agent_id]
    # Use display coordinates: y_display = (height-1) - y_internal
    # so "up" = increasing y, which is intuitive
    h = env.height
    dx, dy = ax, (h - 1) - ay

    parts = [f"You(agent {agent_id}) at ({dx},{dy})."]

    # Scan local 5x3 window
    half_w, half_h = 2, 1
    y0 = max(0, ay - half_h)
    y1 = min(h - 1, ay + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    items_seen = []
    agents_seen = []

    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            if ix == ax and iy == ay:
                # Report item under self
                item = env.items[iy][ix]
                if item == 'a':
                    parts.append(f"Apple HERE at ({ix},{(h-1)-iy}).")
                elif item == '#':
                    parts.append(f"Dirt HERE at ({ix},{(h-1)-iy}).")
                continue
            item = env.items[iy][ix]
            if item == 'a':
                items_seen.append(f"apple({ix},{(h-1)-iy})")
            elif item == '#':
                items_seen.append(f"dirt({ix},{(h-1)-iy})")

    for aid, (ox, oy) in env.agents.items():
        if aid != agent_id and x0 <= ox <= x1 and y0 <= oy <= y1:
            agents_seen.append(f"agent{aid}({ox},{(h-1)-oy})")

    # Report terrain under self
    terrain = env.terrain[ay][ax]
    if terrain == 'x':
        parts.append("On water(river).")
    else:
        parts.append("On land.")

    if items_seen:
        parts.append("Nearby: " + ", ".join(items_seen) + ".")
    if agents_seen:
        parts.append("Other agents: " + ", ".join(agents_seen) + ".")

    return " ".join(parts)


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
    ax, ay = env.agents[agent_id]
    h = env.height
    dx, dy = ax, (h - 1) - ay

    parts = [f"You are agent {agent_id} at ({dx},{dy})."]

    # Terrain
    terrain = env.terrain[ay][ax]
    parts.append(f"Terrain: {'water(river)' if terrain == 'x' else 'land'}.")

    # Item at current position
    item_here = env.items[ay][ax]
    if item_here == 'a':
        parts.append(f"An apple is at your position ({dx},{dy}).")
    elif item_here == '#':
        parts.append(f"Dirt is at your position ({dx},{dy}).")

    # Scan local 5x3 window
    half_w, half_h = 2, 1
    y0 = max(0, ay - half_h)
    y1 = min(h - 1, ay + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    nearby_items = []
    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            if ix == ax and iy == ay:
                continue
            item = env.items[iy][ix]
            if item == 'a':
                nearby_items.append(f"apple at ({ix},{(h-1)-iy})")
            elif item == '#':
                nearby_items.append(f"dirt at ({ix},{(h-1)-iy})")

    if nearby_items:
        parts.append("Nearby: " + ", ".join(nearby_items) + ".")

    nearby_agents = []
    for aid, (ox, oy) in env.agents.items():
        if aid != agent_id and x0 <= ox <= x1 and y0 <= oy <= y1:
            nearby_agents.append(f"agent {aid} at ({ox},{(h-1)-oy})")
    if nearby_agents:
        parts.append("Other agents nearby: " + ", ".join(nearby_agents) + ".")

    return " ".join(parts)


# ─────────────────────────────────────────────
# COMPOUND MODE: global scan helpers
# ─────────────────────────────────────────────
# Add any global-scan or information-gathering helpers here.
# These are NOT actions — they provide extra context to the prompt builder.
# Each should return a dict with at least a 'found' (bool) key.
#
def scan_nearest_dirt(env, agent_id: int) -> dict:
    """Find the nearest dirt on the entire map."""
    ax, ay = env.agents[agent_id]
    h = env.height
    best = None
    best_dist = float('inf')
    for iy in range(h):
        for ix in range(env.width):
            if env.items[iy][ix] == '#':
                dist = abs(ix - ax) + abs(iy - ay)
                if dist < best_dist:
                    best_dist = dist
                    best = (ix, iy)
    if best is None:
        return {'found': False}
    return {'found': True, 'x': best[0], 'y': (h - 1) - best[1], 'distance': best_dist}


def scan_nearest_apple(env, agent_id: int) -> dict:
    """Find the nearest apple on the entire map."""
    ax, ay = env.agents[agent_id]
    h = env.height
    best = None
    best_dist = float('inf')
    for iy in range(h):
        for ix in range(env.width):
            if env.items[iy][ix] == 'a':
                dist = abs(ix - ax) + abs(iy - ay)
                if dist < best_dist:
                    best_dist = dist
                    best = (ix, iy)
    if best is None:
        return {'found': False}
    return {'found': True, 'x': best[0], 'y': (h - 1) - best[1], 'distance': best_dist}


def scan_dirt_count(env, agent_id: int) -> dict:
    """Count total dirt on the map."""
    count = sum(1 for iy in range(env.height) for ix in range(env.width) if env.items[iy][ix] == '#')
    return {'found': count > 0, 'count': count}


def scan_apple_count(env, agent_id: int) -> dict:
    """Count total apples on the map."""
    count = sum(1 for iy in range(env.height) for ix in range(env.width) if env.items[iy][ix] == 'a')
    return {'found': count > 0, 'count': count}


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
def move_to(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Move one step toward target (coord_x, coord_y) in display coords.
    Returns (action, is_done). is_done=True when already at target."""
    coord_x = int(kwargs.get('coord_x', 0))
    coord_y = int(kwargs.get('coord_y', 0))
    ax, ay = env.agents[agent_id]
    h = env.height
    # Convert display y to internal y
    target_iy = (h - 1) - coord_y
    target_ix = coord_x

    if ax == target_ix and ay == target_iy:
        return ("stay", True)

    dx = target_ix - ax
    dy = target_iy - ay

    # Move along the axis with greater distance first
    if abs(dx) >= abs(dy):
        if dx > 0:
            return ("right", False)
        else:
            return ("left", False)
    else:
        if dy < 0:
            return ("up", False)
        else:
            return ("down", False)


def clean_at(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Move toward (coord_x, coord_y) and clean dirt there.
    If already at target, executes 'clean'. Otherwise moves toward it."""
    coord_x = int(kwargs.get('coord_x', 0))
    coord_y = int(kwargs.get('coord_y', 0))
    ax, ay = env.agents[agent_id]
    h = env.height
    target_iy = (h - 1) - coord_y
    target_ix = coord_x

    if ax == target_ix and ay == target_iy:
        return ("clean", True)

    # Move toward target
    dx = target_ix - ax
    dy = target_iy - ay
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left", False)
    else:
        return ("up" if dy < 0 else "down", False)


def eat_at(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Move toward (coord_x, coord_y) and eat apple there.
    If already at target, executes 'eat'. Otherwise moves toward it."""
    coord_x = int(kwargs.get('coord_x', 0))
    coord_y = int(kwargs.get('coord_y', 0))
    ax, ay = env.agents[agent_id]
    h = env.height
    target_iy = (h - 1) - coord_y
    target_ix = coord_x

    if ax == target_ix and ay == target_iy:
        return ("eat", True)

    dx = target_ix - ax
    dy = target_iy - ay
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left", False)
    else:
        return ("up" if dy < 0 else "down", False)


def random_explore(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Take a random movement action for exploration."""
    action = random.choice(["up", "down", "left", "right"])
    return (action, True)


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
