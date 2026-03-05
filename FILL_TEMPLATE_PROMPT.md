# Task: Implement observation.py, prompts.py, and reward shaping for a GRPO Multi-Agent RL Game

You are given a GRPO-based multi-agent reinforcement learning codebase. Two files need to be implemented — they are currently empty templates with `raise NotImplementedError`. You must also design reward shaping and update TODO markers in other files.

## What you are building

An LLM plays a grid-based game. Each turn, the LLM receives a natural-language description of the game state and outputs an action. There are two action modes:

- **Text mode**: The LLM outputs a single low-level action word (e.g., "up", "left", "eat").
- **Compound mode**: The LLM outputs a JSON object calling a high-level action function (e.g., `{"action": "eat_at", "args": {"coord_x": 5, "coord_y": 3}}`).

Your job is to read the game environment file and implement the game-specific logic for observations, actions, prompts, and reward shaping.

---

## Step 1: Read the game environment

Read the game environment source file carefully. Understand:
- The grid layout (terrain types, coordinate system, dimensions)
- What items/objects exist on the grid and their symbols
- What low-level actions are available (the `ACTIONS` dict and `step()` method)
- How rewards work (what gives points, what doesn't)
- The game mechanics (spawning rules, collision resolution, etc.)
- The observation window (what each agent can see — check `_local_render` and `_observation`)
- Internal vs display coordinate conventions (if any)

The game environment file is: **`env_move.py`** (in the project root).

---

## Step 2: Implement `utils/observation.py`

This file has three sections to implement. **Do NOT modify the `obs_to_text` dispatch function at the bottom** — it is already wired up.

### Section A: Observation converters (2 functions)

1. **`parse_observation_to_coords(obs, agent_id, env) -> str`**
   - For **text mode**. Converts the environment state into a coordinate-based natural-language description.
   - Should describe: the agent's position, nearby items (within the local observation window), and other visible agents.
   - Use the env object directly to get positions, items, etc.
   - Be aware of any coordinate transformations (internal vs display).
   - Example output style: `"You at (2,3). Dirt at (2,4). Apple at (1,3). Agent 2 at (3,3)."`

2. **`get_observation_description(env, agent_id) -> str`**
   - For **compound mode**. Similar to above but can use a different format/style.
   - Example output style: `"You are at (2, 3). You see an apple at (1, 3) and dirt at (2, 4)."`

### Section B: Global scan helpers

Add information-gathering functions that scan the **entire** map (not just the local window). These are called by the prompt builder to give the LLM strategic context.

- Each helper should return a `dict` with at least `{'found': bool}`.
- If found, include coordinates and distance (Manhattan distance).
- Think about what global information would help the LLM make better decisions.
- These are NOT actions — they just provide data.

### Section C: High-level actions

Add action functions for compound mode. Each function:
- Signature: `def action_name(env: Any, agent_id: int, **kwargs) -> Tuple[str, bool]`
- Returns `(low_level_action_string, is_done)` where the action string is one of the env's low-level actions.
- `is_done` is True when the high-level task is complete or impossible.
- Each function translates a high-level goal (e.g., "go to coordinate X,Y and eat") into a single low-level step.
- Functions must be independent — no helper calling another helper.

Think about what high-level actions make sense for your game. Common patterns:
- Moving toward a target coordinate
- Interacting with a specific item at a coordinate
- Random exploration as a fallback

---

## Step 3: Implement `utils/prompts.py`

This file has 4 prompt functions to implement. **Do NOT modify the `build_thinking_prompt` and `build_action_prompt` dispatch functions at the bottom** — they are already wired up.

Each function must:
- Build a `messages` list of `{"role": "system"/"user"/"assistant", "content": "..."}` dicts
- Call `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
- Return the resulting string

The `config` object has these useful fields:
- `config.eat_reward` — reward for the main scoring action (float)
- `config.action_mode` — "text" or "compound"
- `config.use_two_stage` — bool (text mode only)

### Function 1: `create_thinking_prompt(obs_text, agent_id, config, tokenizer)`
- Text mode, stage 1. System message explains game rules, rewards, available actions. User message is the observation. Ask the agent to think briefly about what to do.

### Function 2: `create_action_prompt(obs_text, thinking_text, config, tokenizer)`
- Text mode, stage 2. Same system context. Messages: system → user (observation) → assistant (thinking from stage 1) → user (instruction to output ONE action word).

### Function 3: `create_single_stage_prompt_text(obs_text, config, tokenizer)`
- Text mode, single-stage. Combines thinking and action into one prompt. System message with rules + "output ONE action word". User message is the observation.

### Function 4: `create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)`
- Compound mode. System message explains the game AND documents all the high-level actions you defined in observation.py (function names, args, JSON format). You may call your global scan helpers here (using `env` and `agent_id`) to add strategic info to the user message. Ask the LLM to think briefly then output one JSON action.

### Prompt design tips
- Keep prompts concise — the LLM has limited token budget (256 thinking + 128 action tokens).
- Include reward values from `config` so the LLM understands incentives.
- For compound mode, clearly document the JSON format expected.
- Don't bloat the system message — shorter prompts train faster.

---

## Step 4: Reward shaping — create `env_copy.py`

Copy `env_move.py` to `env_copy.py`. Then modify **only** `env_copy.py` to add reward shaping. **Do NOT modify `env_move.py`** — it stays as the ground truth for evaluation.

The codebase uses `env_copy.py` for training rollouts and `env_move.py` for evaluation. This means shaped rewards guide learning, but evaluation scores reflect true game performance.

### What to do

1. Analyze the game and identify **sub-goals** — intermediate steps that lead toward the main reward but don't directly score points in the original env. Examples of sub-goals in grid games:
   - Moving closer to a valuable target (distance-based shaping)
   - Reaching a strategically useful position
   - Performing a preparatory action that enables future rewards
   - Cooperating with other agents (e.g., not blocking, covering different areas)

2. In `env_copy.py`, modify the `step()` method to add **small bonus rewards** for achieving these sub-goals. Guidelines:
   - Sub-goal rewards should be **much smaller** than the main game rewards (e.g., 0.01–0.1x of the main reward) to avoid dominating the signal.
   - Use **potential-based shaping** where possible (reward the change/improvement, not the absolute state) to avoid reward hacking.
   - Avoid rewarding things the agent can trivially exploit (e.g., don't reward just "being near an item" — the agent would oscillate).
   - Add clear comments explaining each shaped reward and its purpose.
   - Keep the same class name (`CleanupEnvMove`) and identical API so it's a drop-in replacement.

3. Document your reward shaping design: for each sub-goal reward you add, explain:
   - What behavior it encourages
   - Why it helps learning
   - The magnitude and how you chose it

### Important constraints
- The `Config` dataclass, `reset()`, `render()`, `get_state()`, `set_state()`, and `_observation()` must remain unchanged.
- Only modify `step()` (and add private helper methods if needed).
- The shaped rewards are added to the existing reward dict — don't remove original rewards.

---

## Step 5: Wire up `env_copy.py` for training

Make these changes so training uses `env_copy.py` (shaped rewards) while evaluation keeps using `env_move.py` (original rewards):

### `grpo_textgame.py`
- Change the import to load both env classes:
  ```python
  from env_move import CleanupEnvMove as EvalEnvClass, Config as EnvConfigMove
  from env_copy import CleanupEnvMove as TrainEnvClass
  ```
- After `self.env_config = EnvConfigMove(...)`, add:
  ```python
  self.train_env_class = TrainEnvClass
  self.eval_env_class = EvalEnvClass
  ```

### `utils/rollout.py`
- In both `run_episode()` and `run_parallel_episodes()`, replace:
  ```python
  from env_move import CleanupEnvMove
  ```
  with:
  ```python
  EnvClass = getattr(trainer, 'train_env_class', None)
  if EnvClass is None:
      from env_move import CleanupEnvMove as EnvClass
  ```
  and use `EnvClass(trainer.env_config)` instead of `CleanupEnvMove(trainer.env_config)`.

### `utils/eval.py`
- **No changes needed** — it already imports from `env_move` directly, so evaluation uses original rewards.

---

## Step 6: Update TODO markers in other files

These files have `# === TODO ===` comments marking game-specific values. Update them to match your implementations:

### `grpo_textgame.py` (lines ~91-104)
```python
# === TODO: update action_words to match the low-level actions in your game env ===
# === TODO: update helper_functions to match the high-level actions in observation.py ===
self.action_words = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']
self.helper_functions = ['move_to', 'clean_at', 'eat_at', 'random_explore']
```
- `action_words`: must match the low-level action strings from the env's `ACTIONS` dict.
- `helper_functions`: must match the function names you defined in observation.py section C.

### `utils/generation.py` (line ~120)
```python
# === TODO: change this if your game has a different default/fallback action ===
return "stay"
```
- The fallback action when JSON parsing fails. Should be a valid low-level action (usually the "do nothing" action).

---

## Rules

1. **Do NOT modify** `env_move.py` — it is the ground truth for evaluation.
2. **Do NOT modify** the dispatch functions (`obs_to_text`, `build_thinking_prompt`, `build_action_prompt`) — they are already wired correctly.
3. **Do NOT modify** `utils/generation.py` beyond the TODO markers — the action dispatch uses dynamic `getattr()` lookup on `observation.py`, so any function you define there with the right signature `(env, agent_id, **kwargs) -> Tuple[str, bool]` will automatically be callable via JSON.
4. If your observation.py imports anything from the env module, import the class, not the file (e.g., `from env_move import CleanupEnvMove` if needed for type hints).
5. Make sure `action_words` in `grpo_textgame.py` exactly matches the string keys your env accepts.
6. Make sure `helper_functions` in `grpo_textgame.py` exactly matches the function names in observation.py.
7. All observation text should use coordinates consistently — pick one convention and stick with it.
8. `env_copy.py` must have the same class name, same `Config`, and same API as `env_move.py` — only `step()` internals change.

---

## Deliverables

Output the complete implementations of:
1. `utils/observation.py` — full file
2. `utils/prompts.py` — full file
3. `env_copy.py` — full file (copy of env_move.py with reward shaping added)
4. The specific lines to update in `grpo_textgame.py` (action_words and helper_functions lists)
5. The fallback action in `utils/generation.py` (if it needs changing from "stay")
