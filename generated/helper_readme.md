# Helpers — CleanupEnvMove High-Level Actions

## Coordinate System

All coordinates use **display format**:
- `x`: 0 (left) to width-1 (right)
- `y`: 0 (bottom) to height-1 (top)

The grid is 15 wide x 9 tall by default.

## Functions

### `observation_to_text(env, agent_id) -> str`

Converts the environment state into a natural-language description for the given agent. Includes:
- Agent position and terrain type
- Last movement direction
- Items in the 5x3 local view (apples and dirt with coordinates)
- Nearest items outside the local view
- Map-wide status (total dirt, total apples, current step)

### `move_to(env, agent_id, coord_x, coord_y) -> bool`

Move one step toward target `(coord_x, coord_y)` in display coordinates. Chooses the axis with the larger gap. Returns `False` if already at target.

**JSON:** `{"action": "move_to", "agent_id": 1, "args": {"coord_x": 5, "coord_y": 3}}`

### `move_up(env, agent_id) -> bool`

Move one step up (increasing display y). Returns `False` if at the top edge.

**JSON:** `{"action": "move_up", "agent_id": 1, "args": {}}`

### `move_down(env, agent_id) -> bool`

Move one step down (decreasing display y). Returns `False` if at the bottom edge.

**JSON:** `{"action": "move_down", "agent_id": 1, "args": {}}`

### `move_left(env, agent_id) -> bool`

Move one step left (decreasing x). Returns `False` if at the left edge.

**JSON:** `{"action": "move_left", "agent_id": 1, "args": {}}`

### `move_right(env, agent_id) -> bool`

Move one step right (increasing x). Returns `False` if at the right edge.

**JSON:** `{"action": "move_right", "agent_id": 1, "args": {}}`

### `eat(env, agent_id) -> bool`

Eat an apple at the agent's current position. Returns `True` if an apple was present, `False` otherwise. Gives +1.0 reward.

**JSON:** `{"action": "eat", "agent_id": 1, "args": {}}`

### `clean(env, agent_id) -> bool`

Clean dirt at the agent's current position. Returns `True` if dirt was present, `False` otherwise. No immediate reward, but cleaning enables apple spawning.

**JSON:** `{"action": "clean", "agent_id": 1, "args": {}}`

### `stay(env, agent_id) -> bool`

Do nothing for one step. Always returns `True`.

**JSON:** `{"action": "stay", "agent_id": 1, "args": {}}`

### `dispatch_action(env, agent_id, action_name, args=None) -> bool`

Routes a named action to the correct function. Used for JSON-based action dispatch.

## Action Registry

| Action      | Args                           | Description                    |
|-------------|--------------------------------|--------------------------------|
| `move_to`   | `coord_x: int, coord_y: int`  | Move one step toward target    |
| `move_up`   | (none)                         | Move one step up               |
| `move_down` | (none)                         | Move one step down             |
| `move_left` | (none)                         | Move one step left             |
| `move_right`| (none)                         | Move one step right            |
| `eat`       | (none)                         | Eat apple at current position  |
| `clean`     | (none)                         | Clean dirt at current position |
| `stay`      | (none)                         | Do nothing                     |

## Important Notes

- Each function calls `env.step()` exactly once, advancing the environment by one tick.
- Functions are independent — no helper calls another helper.
- `move_to` only moves ONE step per call (not pathfinding to completion).
- Apples spawn on land only after dirt count drops below initial count.
- Cleaning gives 0 reward but is necessary for apple spawning.
