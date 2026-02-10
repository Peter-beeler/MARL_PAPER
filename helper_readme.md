# Cleanup Environment Helpers

This module provides high-level abstractions for the `CleanupEnvMove` environment (`env_move.py`). It allows you to control agents using goals (e.g., "go clean", "go eat") rather than calculating raw grid coordinates and action integers manually.

## Installation

Ensure `helpers.py` is in the same directory as `env_move.py`.

## Usage Example

Here is how to run a simulation where Agent 1 focuses on cleaning (to spawn apples) and Agent 2 focuses on eating.


from env_move import CleanupEnvMove
import helpers

env = CleanupEnvMove()
obs = env.reset()

for _ in range(50):
    actions = {}
    
    # Agent 1: The Janitor (Focuses on Dirt)
    actions[1] = helpers.smart_clean_step(env, 1)
    
    # Agent 2: The Forager (Focuses on Apples)
    actions[2] = helpers.smart_forage_step(env, 2)
    
    # Agent 3: Random Walker
    actions[3] = helpers.random_walk(env, 3)

    obs, rewards, done, info = env.step(actions)
    print(env.render())
    
    if done:
        break


## API Reference

### Constants
The file exports action constants matching the environment:
`ACT_STAY`, `ACT_EAT`, `ACT_CLEAN`, `ACT_UP`, `ACT_DOWN`, `ACT_LEFT`, `ACT_RIGHT`.

### Core Functions

#### `move_to(env, agent_id, target_pos)`
Calculates the next directional step to reach a specific (x, y) coordinate.
- **Returns:** `(action_int, arrived_bool)`
- `arrived_bool` is True if the agent is already at the target coordinates.

#### `get_nearest_item(env, agent_id, item_char)`
Scans the global grid to find the closest item of a specific type.
- `item_char`: `'#'` for dirt, `'a'` for apples.
- **Returns:** `(x, y)` tuple or `None` if no items exist.

### High-Level Behaviors

#### `smart_clean_step(env, agent_id)`
Executes a full logic loop for cleaning:
1. If standing on dirt -> **CLEAN**.
2. Else, find nearest dirt -> **MOVE** towards it.
3. If no dirt exists -> **RANDOM WALK**.

#### `smart_forage_step(env, agent_id)`
Executes a full logic loop for eating:
1. If standing on apple -> **EAT**.
2. Else, find nearest apple -> **MOVE** towards it.
3. If no apples exist -> Fallback to **CLEAN** logic (since cleaning spawns apples).