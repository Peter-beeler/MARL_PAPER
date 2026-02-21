import random
from typing import Tuple, Dict, Any, List

def get_observation_description(env: Any, agent_id: int) -> str:
    """
    Generates a natural language description of the agent's global position
    and the objects visible within its local observation window (5x3).
    
    Args:
        env: The CleanupEnvMove instance.
        agent_id: The ID of the agent.
        
    Returns:
        A string description.
    """
    if agent_id not in env.agents:
        return "You are not in the environment."

    ax, ay = env.agents[agent_id]
    
    # Define the local window bounds (radius 2 horizontal, 1 vertical)
    half_w = 2
    half_h = 1
    x_min = max(0, ax - half_w)
    x_max = min(env.width - 1, ax + half_w)
    y_min = max(0, ay - half_h)
    y_max = min(env.height - 1, ay + half_h)

    visible_objects = []

    # Scan for items (apples and dirt)
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            item = env.items[y][x]
            if item == 'a':
                visible_objects.append(f"an apple at ({x}, {y})")
            elif item == '#':
                visible_objects.append(f"dirt at ({x}, {y})")
            
            # Check for other agents
            for other_id, pos in env.agents.items():
                if other_id != agent_id and pos == (x, y):
                    visible_objects.append(f"agent {other_id} at ({x}, {y})")

    # Construct the sentence
    desc = f"You are at ({ax}, {ay})."
    if not visible_objects:
        desc += " You see nothing of interest nearby."
    else:
        # Join with commas and 'and'
        if len(visible_objects) == 1:
            desc += f" You see {visible_objects[0]}."
        else:
            joined = ", ".join(visible_objects[:-1]) + f" and {visible_objects[-1]}"
            desc += f" You see {joined}."

    return desc


def move_to(env: Any, agent_id: int, coord_x: int, coord_y: int) -> Tuple[str, bool]:
    """
    Calculates the next step to move towards a target coordinate.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        coord_x: Target X coordinate.
        coord_y: Target Y coordinate.
        
    Returns:
        Tuple (action_string, is_done). 
        action_string is 'up', 'down', 'left', 'right', or 'stay'.
        is_done is True if the agent is already at the target.
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]

    if curr_x == coord_x and curr_y == coord_y:
        return "stay", True

    # Simple Manhattan pathfinding
    # Prioritize X movement, then Y movement (arbitrary choice)
    print(f"Moving from ({curr_x}, {curr_y}) towards ({coord_x}, {coord_y}).")  # Debug statement
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
    Moves to the target coordinate and cleans it.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        coord_x: Target X coordinate.
        coord_y: Target Y coordinate.
        
    Returns:
        Tuple (action_string, is_done).
        Returns a movement action if not at target.
        Returns 'clean' if at target and dirt exists.
        Returns 'stay' and is_done=True if at target and no dirt exists.
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]
    # If not at target, move there
    if (curr_x, curr_y) != (coord_x, coord_y):
        if curr_x < coord_x:
            return "right", False
        elif curr_x > coord_x:
            return "left", False
        elif curr_y < coord_y:
            return "down", False
        elif curr_y > coord_y:
            return "up", False
    
    # If at target, check for dirt
    item = env.items[curr_y][curr_x]
    if item == '#':
        return "clean", False # Action is clean, task not technically "done" until cleaned
    else:
        # No dirt here, task is done (or impossible)
        return "stay", True


def eat_at(env: Any, agent_id: int, coord_x: int, coord_y: int) -> Tuple[str, bool]:
    """
    Moves to the target coordinate and eats an apple.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        coord_x: Target X coordinate.
        coord_y: Target Y coordinate.
        
    Returns:
        Tuple (action_string, is_done).
        Returns a movement action if not at target.
        Returns 'eat' if at target and apple exists.
        Returns 'stay' and is_done=True if at target and no apple exists.
    """
    if agent_id not in env.agents:
        return "stay", True

    curr_x, curr_y = env.agents[agent_id]

    # If not at target, move there
    if (curr_x, curr_y) != (coord_x, coord_y):
        if curr_x < coord_x:
            return "right", False
        elif curr_x > coord_x:
            return "left", False
        elif curr_y < coord_y:
            return "down", False
        elif curr_y > coord_y:
            return "up", False
            
    # If at target, check for apple
    item = env.items[curr_y][curr_x]
    if item == 'a':
        return "eat", False
    else:
        # No apple here
        return "stay", True


def random_explore(env: Any, agent_id: int) -> Tuple[str, bool]:
    """
    Returns a random valid movement action.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        
    Returns:
        Tuple (action_string, is_done). is_done is always False.
    """
    actions = ["up", "down", "left", "right"]
    return random.choice(actions), False


def find_nearest_dirt(env: Any, agent_id: int) -> Dict[str, Any]:
    """
    Locates the nearest dirt ('#') globally.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        
    Returns:
        Dict with keys 'found' (bool), 'coord_x', 'coord_y', 'distance'.
        If not found, 'found' is False.
    """
    if agent_id not in env.agents:
        return {'found': False}

    ax, ay = env.agents[agent_id]
    nearest_dist = float('inf')
    nearest_pos = None

    for y in range(env.height):
        for x in range(env.width):
            if env.items[y][x] == '#':
                dist = abs(x - ax) + abs(y - ay) # Manhattan distance
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pos = (x, y)
    
    if nearest_pos:
        return {
            'found': True,
            'coord_x': nearest_pos[0],
            'coord_y': nearest_pos[1],
            'distance': nearest_dist
        }
    return {'found': False}


def find_nearest_apple(env: Any, agent_id: int) -> Dict[str, Any]:
    """
    Locates the nearest apple ('a') globally.
    
    Args:
        env: The environment instance.
        agent_id: The agent ID.
        
    Returns:
        Dict with keys 'found' (bool), 'coord_x', 'coord_y', 'distance'.
        If not found, 'found' is False.
    """
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
        return {
            'found': True,
            'coord_x': nearest_pos[0],
            'coord_y': nearest_pos[1],
            'distance': nearest_dist
        }
    return {'found': False}