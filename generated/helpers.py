"""
helpers.py — High-level action functions for agents in CleanupEnvMove.

Each function is independent and callable via JSON:
    {"action": "move_to", "agent_id": 1, "args": {"coord_x": 5, "coord_y": 3}}

Coordinate system (display coordinates):
    x: 0 (left) to width-1 (right)
    y: 0 (bottom) to height-1 (top)
    Internal grid uses y=0 at top, but these helpers accept display coords
    where y=0 is at the bottom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env_move import CleanupEnvMove


# ---------------------------------------------------------------------------
# Coordinate conversion helpers (internal use only)
# ---------------------------------------------------------------------------

def _display_to_internal(env: "CleanupEnvMove", display_y: int) -> int:
    """Convert display y (0=bottom) to internal y (0=top)."""
    return (env.height - 1) - display_y


def _internal_to_display(env: "CleanupEnvMove", internal_y: int) -> int:
    """Convert internal y (0=top) to display y (0=bottom)."""
    return (env.height - 1) - internal_y


# ---------------------------------------------------------------------------
# observation_to_text
# ---------------------------------------------------------------------------

def observation_to_text(env: "CleanupEnvMove", agent_id: int) -> str:
    """Convert raw observation into a natural-language description.

    Describes the agent's position, the 5x3 local window contents,
    nearby items outside the window, and other agents.

    Args:
        env: The live CleanupEnvMove environment instance.
        agent_id: Integer agent identifier (1-based).

    Returns:
        A human-readable string with coordinates in display format.
    """
    ax, ay = env.agents[agent_id]
    disp_y = _internal_to_display(env, ay)

    # Movement info
    direction, prev_pos = env.get_agent_movement(agent_id)

    parts = [f"You are Agent {agent_id} at position ({ax}, {disp_y})."]

    if direction != "STAYED":
        parts.append(f"You just moved {direction} from ({prev_pos[0]}, {prev_pos[1]}).")

    # Terrain under agent
    terrain = env.terrain[ay][ax]
    parts.append(f"You are standing on {'land' if terrain == '*' else 'water'}.")

    # Local window (5x3)
    half_w, half_h = 2, 1
    y0 = max(0, ay - half_h)
    y1 = min(env.height - 1, ay + half_h)
    x0 = max(0, ax - half_w)
    x1 = min(env.width - 1, ax + half_w)

    local_items = []
    local_agents = []
    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            if ix == ax and iy == ay:
                # Item under agent
                item = env.items[iy][ix]
                if item == 'a':
                    parts.append("There is an apple at your position — use 'eat' to collect it!")
                elif item == '#':
                    parts.append("There is dirt at your position — use 'clean' to remove it!")
                continue
            dy = _internal_to_display(env, iy)
            item = env.items[iy][ix]
            t = env.terrain[iy][ix]
            if item == 'a':
                local_items.append(f"apple at ({ix}, {dy})")
            elif item == '#':
                local_items.append(f"dirt at ({ix}, {dy})")
            # Check for other agents
            for aid, (ox, oy) in env.agents.items():
                if aid != agent_id and ox == ix and oy == iy:
                    local_agents.append(f"Agent {aid} at ({ix}, {dy})")

    if local_items:
        parts.append("In your local view: " + ", ".join(local_items) + ".")
    else:
        parts.append("Your local view is clear — no items nearby.")

    if local_agents:
        parts.append("Nearby agents: " + ", ".join(local_agents) + ".")

    # Nearest items outside local window (global hints)
    nearest_dirt = env._find_nearest_items(agent_id, '#', n=2, exclude_local_window=True)
    nearest_apple = env._find_nearest_items(agent_id, 'a', n=2, exclude_local_window=True)

    if nearest_dirt:
        dirt_strs = [f"({dx}, {_internal_to_display(env, dy)})" for dx, dy in nearest_dirt]
        parts.append("Nearest dirt outside view: " + ", ".join(dirt_strs) + ".")

    if nearest_apple:
        apple_strs = [f"({ax2}, {_internal_to_display(env, ay2)})" for ax2, ay2 in nearest_apple]
        parts.append("Nearest apples outside view: " + ", ".join(apple_strs) + ".")

    # Game state summary
    dirt_count = env._count_items('#')
    apple_count = env._count_items('a')
    parts.append(f"Map status: {dirt_count} dirt, {apple_count} apples on the map. Step {env.step_count}/{env.cfg.max_steps}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# High-level action: move_to
# ---------------------------------------------------------------------------

def move_to(env: "CleanupEnvMove", agent_id: int, coord_x: int, coord_y: int) -> bool:
    """Move agent one step toward the target display coordinate.

    Takes ONE env step moving in the direction that reduces distance to (coord_x, coord_y).
    Movement is one cell at a time (up/down/left/right).

    Args:
        env: The live environment.
        agent_id: The agent's ID.
        coord_x: Target x in display coordinates.
        coord_y: Target y in display coordinates (0=bottom).

    Returns:
        True if a step was taken, False if already at target.
    """
    ax, ay = env.agents[agent_id]
    target_iy = _display_to_internal(env, coord_y)

    dx = coord_x - ax
    dy = target_iy - ay

    if dx == 0 and dy == 0:
        # Already at target
        env.step({agent_id: "stay"})
        return False

    # Choose direction: prefer axis with larger distance
    if abs(dx) >= abs(dy):
        action = "right" if dx > 0 else "left"
    else:
        action = "down" if dy > 0 else "up"

    env.step({agent_id: action})
    return True


# ---------------------------------------------------------------------------
# High-level action: move_up
# ---------------------------------------------------------------------------

def move_up(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Move the agent one step up (increasing display y).

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if the agent moved, False if already at the top edge.
    """
    ax, ay = env.agents[agent_id]
    if ay <= 0:
        return False
    env.step({agent_id: "up"})
    return True


# ---------------------------------------------------------------------------
# High-level action: move_down
# ---------------------------------------------------------------------------

def move_down(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Move the agent one step down (decreasing display y).

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if the agent moved, False if already at the bottom edge.
    """
    ax, ay = env.agents[agent_id]
    if ay >= env.height - 1:
        return False
    env.step({agent_id: "down"})
    return True


# ---------------------------------------------------------------------------
# High-level action: move_left
# ---------------------------------------------------------------------------

def move_left(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Move the agent one step left (decreasing x).

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if the agent moved, False if already at the left edge.
    """
    ax, ay = env.agents[agent_id]
    if ax <= 0:
        return False
    env.step({agent_id: "left"})
    return True


# ---------------------------------------------------------------------------
# High-level action: move_right
# ---------------------------------------------------------------------------

def move_right(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Move the agent one step right (increasing x).

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if the agent moved, False if already at the right edge.
    """
    ax, ay = env.agents[agent_id]
    if ax >= env.width - 1:
        return False
    env.step({agent_id: "right"})
    return True


# ---------------------------------------------------------------------------
# High-level action: eat
# ---------------------------------------------------------------------------

def eat(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Eat an apple at the agent's current position.

    The agent must be standing on a cell with an apple ('a').

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if there was an apple to eat, False otherwise.
    """
    ax, ay = env.agents[agent_id]
    has_apple = env.items[ay][ax] == 'a'
    env.step({agent_id: "eat"})
    return has_apple


# ---------------------------------------------------------------------------
# High-level action: clean
# ---------------------------------------------------------------------------

def clean(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Clean dirt at the agent's current position.

    The agent must be standing on a cell with dirt ('#').

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        True if there was dirt to clean, False otherwise.
    """
    ax, ay = env.agents[agent_id]
    has_dirt = env.items[ay][ax] == '#'
    env.step({agent_id: "clean"})
    return has_dirt


# ---------------------------------------------------------------------------
# High-level action: stay
# ---------------------------------------------------------------------------

def stay(env: "CleanupEnvMove", agent_id: int) -> bool:
    """Do nothing for one step.

    Args:
        env: The live environment.
        agent_id: The agent's ID.

    Returns:
        Always True.
    """
    env.step({agent_id: "stay"})
    return True


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

ACTION_REGISTRY = {
    "move_to": move_to,
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_right": move_right,
    "eat": eat,
    "clean": clean,
    "stay": stay,
}


def dispatch_action(env: "CleanupEnvMove", agent_id: int, action_name: str, args: dict = None) -> bool:
    """Dispatch a named action to the appropriate helper function.

    Called via JSON: {"action": "move_to", "agent_id": 1, "args": {"coord_x": 5, "coord_y": 3}}

    Args:
        env: The live environment.
        agent_id: The agent's ID.
        action_name: Name of the action (must be in ACTION_REGISTRY).
        args: Optional dict of keyword arguments for the action function.

    Returns:
        The boolean result of the action function.
    """
    if args is None:
        args = {}

    func = ACTION_REGISTRY.get(action_name)
    if func is None:
        # Unknown action, default to stay
        return stay(env, agent_id)

    return func(env, agent_id, **args)
