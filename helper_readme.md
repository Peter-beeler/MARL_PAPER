# Cleanup Environment Helpers

This package provides high-level helper functions for the `CleanupEnvMove` environment. These functions allow agents to perform complex tasks (like navigation and interaction) and interpret observations using natural language.

## Coordinate System
The environment uses a grid coordinate system:
- **(0, 0)** is the **Top-Left** corner.
- **X** increases to the Right.
- **Y** increases Downward.

## Usage

These functions are designed to be called by an agent loop. The agent can output a JSON command, which you can map to these functions.

### Observation Helper

#### `get_observation_description(env, agent_id)`
Translates the agent's local grid observation into a natural language string.
- **Returns:** String (e.g., *"You are at (2,3). You see an apple at (2,4) and dirt at (1,3)."*)

### Action Helpers

These functions return the **immediate next action** required to achieve a goal. They should be called in a loop until `is_done` is True.

#### `move_to(env, agent_id, coord_x, coord_y)`
Navigates the agent toward a specific coordinate.
- **Returns:** `(action_str, is_done)`
- **Example:** `('up', False)` or `('stay', True)`

#### `clean_at(env, agent_id, coord_x, coord_y)`
Navigates to the coordinate and performs the 'clean' action if dirt is present.
- **Returns:** `(action_str, is_done)`
- **Logic:** Moves to target -> Cleans if dirt exists -> Returns done if clean.

#### `eat_at(env, agent_id, coord_x, coord_y)`
Navigates to the coordinate and performs the 'eat' action if an apple is present.
- **Returns:** `(action_str, is_done)`

#### `random_explore(env, agent_id)`
Returns a random movement action (up, down, left, right).
- **Returns:** `(action_str, False)`

### Information Helpers

These functions help the agent decide *where* to go.

#### `find_nearest_dirt(env, agent_id)`
Scans the entire grid for the closest dirt.
- **Returns:** `{'found': bool, 'coord_x': int, 'coord_y': int, 'distance': int}`

#### `find_nearest_apple(env, agent_id)`
Scans the entire grid for the closest apple.
- **Returns:** `{'found': bool, 'coord_x': int, 'coord_y': int, 'distance': int}`

## Integration Example

If an LLM agent outputs the following JSON:

json
{
    "action": "move_to",
    "agent_id": 1,
    "args": {
        "coord_x": 5,
        "coord_y": 3
    }
}


You can execute it in Python like this:


import helpers

# ... inside your game loop ...
cmd = agent_response_json
func = getattr(helpers, cmd['action'])
action_str, is_done = func(env, cmd['agent_id'], **cmd['args'])

# Execute the low-level action
obs, rewards, done, info = env.step({cmd['agent_id']: action_str})