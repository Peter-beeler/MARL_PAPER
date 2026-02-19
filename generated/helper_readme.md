# Helper Functions Reference

This document describes all high-level helper functions available for
CleanupEnvMove agents. Functions are called via JSON and automatically
dispatched by `dispatch_action()`.

---

## Coordinate System

The game uses **display coordinates** in all public interfaces:

- **x**: 0 = left edge, increases to the right (0–14 for a 15-wide grid)
- **y**: 0 = **bottom** edge, increases upward (0–8 for a 9-tall grid)

The internal grid stores y with 0 at the top, but helpers handle the conversion.

---

## Observation Text

### `observation_to_text(env, agent_id) -> str`

Converts raw environment state into a natural-language string for the given agent.

**Contains:**
- Agent's current position (display coords)
- Terrain type at agent position (land / river)
- Item at agent's position (apple, dirt, or nothing)
- Visible items in the 5×3 local window with display coordinates
- Visible other agents in the local window
- Nearest 3 dirts and 3 apples on the entire map
- All agent positions (global map awareness)
- Current scores for all agents
- Episode step counter
- Overall map dirt/apple counts and initial dirt count

**Example output:**
```
You are Agent 1 at position (3, 5). You are standing on land.
No apples visible in your immediate area.
Visible dirt nearby: (6, 4), (7, 5).
Nearest dirt on the map: (6, 4), (7, 5), (6, 3).
Nearest apples on the map: (2, 7), (11, 6).
All agents: Agent 2 at (9,5), Agent 3 at (1,2).
Scores: Agent 1: 0.0, Agent 2: 1.0, Agent 3: 0.0.
Step: 12/30.
Map status: 5 dirt on river, 2 apples on land. Initial dirt was 8. Apples spawn faster when dirt count drops below 8.
```

---

## Action Functions

All action functions:
- Take `(env, agent_id, ...)` as first two arguments
- Return `bool` indicating success or completion
- Only affect the target agent (other agents default to `"stay"`)
- Call `env.step()` internally

### `move_to(env, agent_id, coord_x, coord_y) -> bool`

Move agent one step closer to the target position using the optimal direction.

**Args:**
- `coord_x` (int): Target x in display coordinates
- `coord_y` (int): Target y in display coordinates (0 = bottom)

**Returns:** `True` if agent is already at destination, `False` if a move was taken.

**JSON call:**
```json
{"action": "move_to", "agent_id": 1, "args": {"coord_x": 6, "coord_y": 4}}
```

**Note:** This function takes ONE step. To reach a distant target, call it repeatedly over multiple environment steps.

---

### `eat(env, agent_id) -> bool`

Attempt to eat an apple at the agent's current position.

**Returns:** `True` if the agent was standing on an apple tile, `False` otherwise.

**Reward:** +1.0 for successfully eating an apple.

**JSON call:**
```json
{"action": "eat", "agent_id": 1, "args": {}}
```

---

### `clean(env, agent_id) -> bool`

Attempt to clean dirt at the agent's current position.

**Returns:** `True` if the agent was standing on a dirt tile, `False` otherwise.

**Reward:** +0.0 directly, but cleaning lowers the river's dirt count, which causes apples to spawn faster on land.

**JSON call:**
```json
{"action": "clean", "agent_id": 1, "args": {}}
```

---

### `stay(env, agent_id) -> bool`

Do nothing — agent waits for one step.

**Returns:** Always `True`.

**JSON call:**
```json
{"action": "stay", "agent_id": 1, "args": {}}
```

---

### `move_up(env, agent_id) -> bool`

Move one step upward (increasing display y).

**Returns:** `True` if the agent actually moved, `False` if blocked by boundary or another agent.

**JSON call:**
```json
{"action": "move_up", "agent_id": 1, "args": {}}
```

---

### `move_down(env, agent_id) -> bool`

Move one step downward (decreasing display y).

**Returns:** `True` if the agent actually moved, `False` if blocked.

**JSON call:**
```json
{"action": "move_down", "agent_id": 1, "args": {}}
```

---

### `move_left(env, agent_id) -> bool`

Move one step to the left (decreasing x).

**Returns:** `True` if the agent actually moved, `False` if blocked.

**JSON call:**
```json
{"action": "move_left", "agent_id": 1, "args": {}}
```

---

### `move_right(env, agent_id) -> bool`

Move one step to the right (increasing x).

**Returns:** `True` if the agent actually moved, `False` if blocked.

**JSON call:**
```json
{"action": "move_right", "agent_id": 1, "args": {}}
```

---

## Dispatch

### `dispatch_action(env, action_json: dict) -> bool`

Execute any helper function from a parsed JSON dict.

**Input format:**
```json
{
    "action": "<function_name>",
    "agent_id": <int>,
    "args": {<keyword arguments>}
}
```

**Example:**
```python
result = dispatch_action(env, {
    "action": "move_to",
    "agent_id": 2,
    "args": {"coord_x": 7, "coord_y": 4}
})
```

---

### `parse_action_json(response_text: str) -> dict | None`

Extract a JSON action dict from raw model output text.

Returns the parsed dict or `None` if no valid JSON with an `"action"` key is found.

---

## Available Action Names

| Action Name  | Args Required         | Description                                 |
|--------------|-----------------------|---------------------------------------------|
| `move_to`    | `coord_x`, `coord_y`  | Move one step toward (coord_x, coord_y)     |
| `eat`        | _(none)_              | Eat apple at current position               |
| `clean`      | _(none)_              | Clean dirt at current position              |
| `stay`       | _(none)_              | Wait one step                               |
| `move_up`    | _(none)_              | Move up (increase display y)                |
| `move_down`  | _(none)_              | Move down (decrease display y)              |
| `move_left`  | _(none)_              | Move left (decrease x)                      |
| `move_right` | _(none)_              | Move right (increase x)                     |

---

## Game Rules Summary

- **Grid**: 15 wide × 9 tall. A 3-column vertical river band runs through the center.
- **Dirt (#)**: Spawns on river cells. Clean it to enable apple spawning.
- **Apples (a)**: Spawn on land cells. Eat them for +1.0 reward.
- **Apple spawning**: Rate increases as river gets cleaner (dirt count drops below initial count).
- **Collision**: Two agents moving to the same cell both stay put.
- **Episode length**: 200 steps by default (30 in training config).
- **Goal**: Cooperate — some agents clean the river while others harvest apples.
