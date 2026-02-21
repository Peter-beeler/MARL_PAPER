import json
from typing import List, Dict, Any, Optional
import helpers

def get_system_context() -> str:
    """
    Returns the static context describing the game rules and mechanics.
    """
    return (
        "You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+1.0 each). "
        "Rules: Apples only spawn on land when the river is clean (less dirt = more apples). "
        "Cleaning dirt gives no immediate reward but is necessary to enable apple spawning. "
        "You cannot move diagonally. You have high-level functions to navigate and interact."
    )

def get_action_api() -> str:
    """
    Returns the documentation for the available high-level actions and JSON format.
    """
    return """
### AVAILABLE ACTIONS
You must respond with a JSON object calling one of these functions:

1. **move_to(coord_x, coord_y)**
   - Moves you towards the specified coordinate.
   - Use this to get closer to a target.

2. **clean_at(coord_x, coord_y)**
   - Moves to the coordinate and cleans the dirt there.
   - Use this when you decide to help the ecosystem.

3. **eat_at(coord_x, coord_y)**
   - Moves to the coordinate and eats the apple there.
   - Use this to gain rewards.

4. **random_explore()**
   - Moves in a random direction.
   - Use this if you see nothing of interest and have no target.

### RESPONSE FORMAT
You must output **ONLY** a valid JSON object. Do not add markdown formatting like json.
Example:
{
    "action": "eat_at",
    "agent_id": 1,
    "args": {
        "coord_x": 5,
        "coord_y": 3
    }
}
"""

def create_thinking_prompt(env: Any, agent_id: int) -> List[Dict[str, str]]:
    """
    Creates a prompt for the 'Thinking' stage. 
    Injects global knowledge (nearest items) to help the agent reason.
    
    Args:
        env: The CleanupEnvMove instance.
        agent_id: The agent's ID.
        
    Returns:
        A list of chat messages.
    """
    # 1. Get Natural Language Observation
    obs_text = helpers.get_observation_description(env, agent_id)
    
    # 2. Get Global Strategic Info (The "GPS")
    nearest_apple = helpers.find_nearest_apple(env, agent_id)
    nearest_dirt = helpers.find_nearest_dirt(env, agent_id)
    
    strategy_info = []
    if nearest_apple['found']:
        strategy_info.append(f"- Nearest Apple: at ({nearest_apple['coord_x']}, {nearest_apple['coord_y']}), distance {nearest_apple['distance']}.")
    else:
        strategy_info.append("- Nearest Apple: None found.")
        
    if nearest_dirt['found']:
        strategy_info.append(f"- Nearest Dirt: at ({nearest_dirt['coord_x']}, {nearest_dirt['coord_y']}), distance {nearest_dirt['distance']}.")
    else:
        strategy_info.append("- Nearest Dirt: None found.")
        
    strategy_str = "\n".join(strategy_info)

    # 3. Construct Prompt
    system_msg = get_system_context() + "\n\nYour task is to ANALYZE the situation. Do not output JSON yet. Simply reason about what you should do."
    
    user_content = f"""
You are Agent {agent_id}.

### CURRENT OBSERVATION
{obs_text}

### GLOBAL SCAN
{strategy_str}

### DECISION REQUIRED
Analyze the trade-off:
1. Should you eat an apple for immediate reward?
2. Should you clean dirt to ensure future apples spawn (helping the group)?
3. If nothing is nearby, where should you explore?

Provide your reasoning.
"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

    """
    Creates a prompt for the 'Action' stage.
    Takes the previous reasoning and demands a JSON action.
    
    Args:
        env: The CleanupEnvMove instance.
        thinking_response: The output from the previous thinking step.
        agent_id: The agent's ID.
        
    Returns:
        A list of chat messages.
    """
    # Re-generate context to ensure the prompt is self-contained if used in a stateless way,
    # though typically this is appended to the chat history.
    # Here we construct a "final" prompt to force the JSON.
    
    system_msg = get_system_context() + "\n" + get_action_api()
    
    # We reconstruct the user input briefly to remind the model of the context
    obs_text = helpers.get_observation_description(env, agent_id)
    
    user_content = f"""
You are Agent {agent_id}.
Observation: {obs_text}

Your Analysis:
{thinking_response}

Based on your analysis, output the specific JSON action to execute now.
"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]