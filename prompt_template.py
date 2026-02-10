"""
Prompt templates for the Cleanup Environment (env_move.py).
This module generates messages for an LLM agent to make decisions based on
high-level helper functions defined in helpers.py.
"""

import json

def get_system_prompt():
    """
    Returns the system prompt describing the environment, rules, and available high-level actions.
    """
    return """You are an autonomous AI agent playing a cooperative 'Cleanup' game in a 2D grid world.

### ENVIRONMENT & RULES
- **Map Layout**: The world consists of Land ('*') and a central River ('x').
- **Objects**:
  - **Apples ('a')**: Spawn on Land. Eating an apple gives **+1.0 reward**.
  - **Dirt ('#')**: Spawns on Water (River). Cleaning dirt gives **+0.0 immediate reward**.
  - **Agents ('1'..'9')**: You and other agents.
- **The Ecosystem (Crucial)**:
  - Apples auto-generate on land, BUT the spawn rate depends on the cleanliness of the river.
  - If the river is full of dirt, **apples stop spawning entirely**.
  - To maximize rewards, agents must balance cleaning the river (maintenance) and eating apples (harvesting).
  - If everyone eats and no one cleans, the food source will die out.

### AVAILABLE HIGH-LEVEL ACTIONS
You do not control individual steps (up/down/left/right). Instead, you choose a high-level strategy, and the system handles the pathfinding.

1. **`smart_clean_step`**
   - **Behavior**: Automatically scans the global map for the nearest Dirt ('#'). Navigates towards it and cleans it.
   - **When to use**: Use this when you see dirt in the river, or when apples are scarce (indicating the river needs cleaning to trigger spawns).

2. **`smart_forage_step`**
   - **Behavior**: Automatically scans the global map for the nearest Apple ('a'). Navigates towards it and eats it.
   - **When to use**: Use this when apples are visible and available.

3. **`random_walk`**
   - **Behavior**: Takes a random step.
   - **When to use**: Use this if you are stuck, or if there are absolutely no items on the map to search for.

### YOUR GOAL
Maximize your personal score (eating apples) while ensuring the ecosystem survives (cleaning dirt).
"""

def create_thinking_prompt(obs_text: str, agent_id: int):
    """
    Creates the prompt for the first stage of reasoning (Chain of Thought).
    
    Args:
        obs_text (str): The string representation of the agent's local 5x3 observation.
        agent_id (int): The ID of the current agent.
        
    Returns:
        List[dict]: A list of messages for the LLM.
    """
    system_msg = get_system_prompt()
    
    user_msg = f"""You are Agent {agent_id}.
    
Here is your local 5x3 visual observation of the grid (you are in the center):

{obs_text}


**Task**: Analyze the situation.
1. Identify what objects are visible in your vicinity (Dirt '#', Apples 'a', Water 'x', Land '*').
2. Assess the state of the game. Is the river dirty? Are there apples ready to harvest?
3. Decide which high-level strategy (`smart_clean_step` or `smart_forage_step`) is best right now.
   - If you see Dirt, the ecosystem might need maintenance.
   - If you see Apples, you can harvest.
   - If you see nothing, consider what the team needs most.

Provide your reasoning briefly.
"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def create_single_stage_prompt(obs_text: str, thinking_response: str, agent_id: int):
    """
    Creates the prompt for the final action selection.
    
    Args:
        obs_text (str): The local observation string.
        thinking_response (str): The output from the previous reasoning step.
        agent_id (int): The ID of the current agent.
        
    Returns:
        List[dict]: A list of messages for the LLM.
    """
    system_msg = get_system_prompt()
    
    # We include the previous turn's context to ensure consistency
    user_msg_1 = f"You are Agent {agent_id}. Observation:\n{obs_text}\nAnalyze the situation."
    assistant_msg_1 = thinking_response
    
    user_msg_2 = """Based on your analysis, output ONLY the chosen action in valid JSON format.
Do not output any other text.

The valid function names are:
- "smart_clean_step"
- "smart_forage_step"
- "random_walk"

**Output Format:**
json
{
    "action": "smart_clean_step"
}

"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg_1},
        {"role": "assistant", "content": assistant_msg_1},
        {"role": "user", "content": user_msg_2}
    ]