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
from typing import Any, Dict, Optional, Tuple


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


# ─────────────────────────────────────────────
# ROLE SWITCHING: tool actions and role helpers
# ─────────────────────────────────────────────

# Set of action names that are "tool" actions — agent stays in place, gets
# information result that appears in next step's observation.
TOOL_ACTIONS = {"find_nearest_apples", "find_nearest_dirts"}

# Module-level storage for tool results.  Keyed by (id(env), agent_id).
# Written by tool action functions, read/cleared by rollout after generation.
_tool_results: Dict = {}


def find_nearest_apples(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Tool action: scan the map for the 5 nearest apples.
    Agent stays in place. Result is stored in _tool_results for next step."""
    ax, ay = env.agents[agent_id]
    h = env.height
    nearest = env._find_nearest_items(agent_id, "a", n=5)

    if not nearest:
        result = "No apples found on the map."
    else:
        parts = []
        for x, y in nearest:
            disp_y = (h - 1) - y
            dist = abs(x - ax) + abs(y - ay)
            parts.append(f"Apple at ({x},{disp_y}), distance={dist}")
        result = "Nearest apples: " + "; ".join(parts)

    _tool_results[(id(env), agent_id)] = result
    return ("stay", True)


def find_nearest_dirts(env, agent_id: int, **kwargs) -> Tuple[str, bool]:
    """Tool action: scan the map for the 5 nearest dirts.
    Agent stays in place. Result is stored in _tool_results for next step."""
    ax, ay = env.agents[agent_id]
    h = env.height
    nearest = env._find_nearest_items(agent_id, "#", n=5)

    if not nearest:
        result = "No dirt found on the map."
    else:
        parts = []
        for x, y in nearest:
            disp_y = (h - 1) - y
            dist = abs(x - ax) + abs(y - ay)
            parts.append(f"Dirt at ({x},{disp_y}), distance={dist}")
        result = "Nearest dirts: " + "; ".join(parts)

    _tool_results[(id(env), agent_id)] = result
    return ("stay", True)


def pop_tool_result(env, agent_id: int) -> str:
    """Get and clear the tool result for this (env, agent). Returns '' if none."""
    return _tool_results.pop((id(env), agent_id), "")


def get_global_role_context(env) -> str:
    """Build a text summary of global game state for the role-assignment meta-call."""
    dirt_count = env._count_items("#")
    apple_count = env._count_items("a")
    h = env.height

    lines = []
    lines.append(f"Step: {env.step_count}/{env.cfg.max_steps}.")
    lines.append(f"Dirt on river: {dirt_count} (initial: {env.init_dirt_count}).")
    lines.append(f"Apples on land: {apple_count}.")

    for aid in sorted(env.agents.keys()):
        ax, ay = env.agents[aid]
        dy = (h - 1) - ay
        score = env.scores.get(aid, 0.0)
        lines.append(f"Agent {aid}: position ({ax},{dy}), score {score:.1f}.")

    return " ".join(lines)


def get_role_specific_observation(env, agent_id: int, role: str) -> str:
    """
    Role-tailored observation for compound mode.
    - eater: emphasizes apples, de-emphasizes dirt
    - cleaner: emphasizes dirt/river, de-emphasizes apples
    """
    ax, ay = env.agents[agent_id]
    h = env.height
    dx, dy = ax, (h - 1) - ay

    parts = [f"You are agent {agent_id} (role: {role.upper()}) at ({dx},{dy})."]

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

    visible_apples = []
    visible_dirts = []
    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            if ix == ax and iy == ay:
                continue
            item = env.items[iy][ix]
            if item == 'a':
                visible_apples.append((ix, (h - 1) - iy))
            elif item == '#':
                visible_dirts.append((ix, (h - 1) - iy))

    nearby_agents = []
    for aid, (ox, oy) in env.agents.items():
        if aid != agent_id and x0 <= ox <= x1 and y0 <= oy <= y1:
            nearby_agents.append(f"agent {aid} at ({ox},{(h-1)-oy})")

    if role == "eater":
        # Emphasize apples
        if visible_apples:
            apple_strs = [f"({x},{y})" for x, y in visible_apples]
            parts.append(f"[PRIORITY] Visible apples nearby: {', '.join(apple_strs)}.")
        else:
            parts.append("No apples visible in your immediate area.")
        if visible_dirts:
            parts.append(f"({len(visible_dirts)} dirt nearby — not your priority.)")

        # Global nearest apples (key info for eaters)
        nearest = env._find_nearest_items(agent_id, "a", n=5)
        if nearest:
            na_strs = []
            for x, y in nearest:
                disp_y = (h - 1) - y
                dist = abs(x - ax) + abs(y - ay)
                na_strs.append(f"({x},{disp_y}) dist={dist}")
            parts.append(f"[PRIORITY] Nearest apples on map: {', '.join(na_strs)}.")
        else:
            parts.append("No apples on the map currently. Cleaners need to clear more dirt!")
        dirt_count = env._count_items("#")
        parts.append(f"Map dirt: {dirt_count} (initial: {env.init_dirt_count}).")
    else:  # cleaner
        # Emphasize dirts
        if visible_dirts:
            dirt_strs = [f"({x},{y})" for x, y in visible_dirts]
            parts.append(f"[PRIORITY] Visible dirt nearby: {', '.join(dirt_strs)}.")
        else:
            parts.append("No dirt visible in your immediate area.")
        if visible_apples:
            parts.append(f"({len(visible_apples)} apples nearby — not your priority.)")

        # Global nearest dirts (key info for cleaners)
        nearest = env._find_nearest_items(agent_id, "#", n=5)
        if nearest:
            nd_strs = []
            for x, y in nearest:
                disp_y = (h - 1) - y
                dist = abs(x - ax) + abs(y - ay)
                nd_strs.append(f"({x},{disp_y}) dist={dist}")
            parts.append(f"[PRIORITY] Nearest dirt on map: {', '.join(nd_strs)}.")
        else:
            parts.append("No dirt on the map! Great job — river is clean.")
        dirt_count = env._count_items("#")
        apple_count = env._count_items("a")
        parts.append(
            f"Map status: {dirt_count} dirt remaining, {apple_count} apples on land. "
            f"Apples spawn faster when dirt drops below {env.init_dirt_count}."
        )

    if nearby_agents:
        parts.append("Other agents nearby: " + ", ".join(nearby_agents) + ".")

    # All agent positions
    other_agents = []
    for aid, (ox, oy) in env.agents.items():
        if aid != agent_id:
            other_agents.append(f"Agent {aid} at ({ox},{(h-1)-oy})")
    if other_agents:
        parts.append(f"All agents: {', '.join(other_agents)}.")

    # Scores
    score_strs = [f"Agent {aid}: {s:.1f}" for aid, s in env.scores.items()]
    parts.append(f"Scores: {', '.join(score_strs)}.")
    parts.append(f"Step: {env.step_count}/{env.cfg.max_steps}.")

    return " ".join(parts)


def check_action_continuation(env, agent_id: int, last_action: Optional[dict]) -> str:
    """
    Check if a multi-step action (move_to, clean_at, eat_at) hasn't finished.
    Returns a hint string if the agent should continue, or '' if done/N/A.
    """
    if last_action is None:
        return ""
    action_name = last_action.get("action", "")
    if action_name not in ("move_to", "clean_at", "eat_at"):
        return ""
    args = last_action.get("args", {})
    target_x = args.get("coord_x")
    target_y_disp = args.get("coord_y")
    if target_x is None or target_y_disp is None:
        return ""

    ax, ay = env.agents[agent_id]
    h = env.height
    ay_disp = (h - 1) - ay

    try:
        target_x = int(target_x)
        target_y_disp = int(target_y_disp)
    except (ValueError, TypeError):
        return ""

    if ax == target_x and ay_disp == target_y_disp:
        return ""  # arrived

    return (
        f"You chose {action_name}({target_x},{target_y_disp}) last step and haven't "
        f"arrived yet (currently at ({ax},{ay_disp})). Continue with "
        f"{action_name}({target_x},{target_y_disp})."
    )
