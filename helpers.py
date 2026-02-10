import random
from typing import Tuple, Optional, List

# Action Constants matching env_move.py
ACT_STAY = 0
ACT_EAT = 1
ACT_CLEAN = 2
ACT_UP = 3
ACT_DOWN = 4
ACT_LEFT = 5
ACT_RIGHT = 6

def get_agent_pos(env, agent_id: int) -> Tuple[int, int]:
    """Retrieves the current (x, y) position of an agent."""
    return env.agents[agent_id]

def get_nearest_item(env, agent_id: int, item_char: str) -> Optional[Tuple[int, int]]:
    """
    Finds the nearest item of a specific type ('a' for apple, '#' for dirt).
    Returns (x, y) of the item or None if none exist.
    """
    ax, ay = get_agent_pos(env, agent_id)
    nearest_pos = None
    min_dist = float('inf')

    # Scan the global grid
    for y in range(env.height):
        for x in range(env.width):
            if env.items[y][x] == item_char:
                # Euclidean distance squared is sufficient for comparison
                dist = (x - ax)**2 + (y - ay)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = (x, y)
    
    return nearest_pos

def move_to(env, agent_id: int, target_pos: Tuple[int, int]) -> Tuple[int, bool]:
    """
    Calculates the next move to get closer to target_pos.
    
    Returns:
        (action, arrived): 
        - action: The int action to take (UP, DOWN, LEFT, RIGHT, or STAY).
        - arrived: True if the agent is currently AT the target_pos.
    """
    ax, ay = get_agent_pos(env, agent_id)
    tx, ty = target_pos

    if ax == tx and ay == ty:
        return ACT_STAY, True

    dx = tx - ax
    dy = ty - ay

    # Simple greedy pathfinding: move along the axis with the largest difference
    # This helps minimize "stair-stepping" which can be inefficient in crowded grids
    if abs(dx) > abs(dy):
        return (ACT_RIGHT if dx > 0 else ACT_LEFT), False
    else:
        # Remember: y increases downwards in this env
        return (ACT_DOWN if dy > 0 else ACT_UP), False

def smart_clean_step(env, agent_id: int) -> int:
    """
    High-level behavior: Find dirt, move to it, clean it.
    
    Logic:
    1. If standing on dirt, CLEAN.
    2. If not, find nearest dirt and move towards it.
    3. If no dirt exists, random walk (patrol).
    """
    ax, ay = get_agent_pos(env, agent_id)
    
    # Check if currently on dirt
    if env.items[ay][ax] == '#':
        return ACT_CLEAN

    # Find nearest dirt
    target = get_nearest_item(env, agent_id, '#')
    
    if target:
        action, arrived = move_to(env, agent_id, target)
        if arrived:
            return ACT_CLEAN
        return action
    
    # No dirt found? Random patrol
    return random.choice([ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT])

def smart_forage_step(env, agent_id: int) -> int:
    """
    High-level behavior: Find apple, move to it, eat it.
    
    Logic:
    1. If standing on apple, EAT.
    2. If not, find nearest apple and move towards it.
    3. If no apples exist, default to cleaning behavior (to spawn more apples).
    """
    ax, ay = get_agent_pos(env, agent_id)

    # Check if currently on apple
    if env.items[ay][ax] == 'a':
        return ACT_EAT

    # Find nearest apple
    target = get_nearest_item(env, agent_id, 'a')

    if target:
        action, arrived = move_to(env, agent_id, target)
        if arrived:
            return ACT_EAT
        return action

    # No apples? Help clean up to spawn them
    return smart_clean_step(env, agent_id)

def random_walk(env, agent_id: int) -> int:
    """Returns a random movement action."""
    return random.choice([ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT])