# GRPO Multi-Agent RL Training Project

## Project Overview

GRPO-based multi-agent reinforcement learning training. An LLM plays a game (`env_move.py`) using high-level helper actions. Your job is to generate the code files and debug them until training runs.

## Source-of-Truth Files 

- `env_move.py` — Game environment implementation
- `grpo_config.py` — Training hyperparameters (`GRPOConfig` dataclass)
- `grpo_template.py` — Template for `train.py` with `# === CUSTOMIZE ===` sections to fill in

## Files You Generate

- `helpers.py` — High-level action functions for agents
- `helper_readme.md` — Documentation for helpers
- `prompt_template.py` — LLM prompt builders (`create_thinking_prompt`, `create_single_stage_prompt`)
- `train.py` — Completed GRPO training loop (filled-in template)
- `launch_training.sh` — figure current linux gpu environments  and write a SLURM job script.

## Generation Pipeline (4 Steps, in order)

### Step 0: Create folder
Create a new folder to store any new files called `generated`

### Step 1: Generate `helpers.py` + `helper_readme.md`

Read `env_move.py`. Then generate:

1. **`helpers.py`** containing:
   - An `observation_to_text(env, agent_id)` function that converts raw observations into natural-language descriptions with coordinates (e.g., "You are at (2,3). You see an apple at (2,4) and a dirt at (1,3).")
   - High-level callable action functions (e.g., `move_to(env, agent_id, coord_x, coord_y) -> bool`, `clean(env, agent_id) -> bool`). Each function abstracts low-level actions (UP, DOWN, etc.) and returns whether the task completed.
   - Functions must be independent — no helper calling another helper. They are building blocks for the agent.
   - Functions are called via JSON: `{"action": "move_to", "agent_id": 1, "args": {"coord_x": 5, "coord_y": 3}}`

2. **`helper_readme.md`** — usage docs for all helpers.

### Step 2: Generate `prompt_template.py`

Read `helpers.py` and `helper_readme.md`. Then generate `prompt_template.py` with:

1. Game context description (map layout, rules, rewards, mechanics from `env_move.py`)
2. Available high-level actions with signatures
3. Response format specification (JSON with action name + args, parseable)
4. Two prompt-building functions that return `list[dict]` with `role`/`content` keys:
   - `create_thinking_prompt(obs_text, agent_id)` — Stage 1: ask agent to reason about the situation
   - `create_single_stage_prompt(obs_text, thinking_response, agent_id)` — Stage 2: given reasoning, output ONE action call

### Step 3: Generate `train.py`

Read `env_move.py`, `helpers.py`, `prompt_template.py`, `grpo_config.py`, and `grpo_template.py`. Then fill in the template:

- Replace every `# === CUSTOMIZE ===` section with real implementations using your helpers and prompts
- Import and instantiate the environment from `env_move.py`
- Use `observation_to_text` from `helpers.py` in `format_observation`
- Use prompt functions from `prompt_template.py` in the prompt builder methods
- Wire up `env.step()` with the action dispatch (parse JSON action → call the right helper function)
- Use `GRPOConfig` field names exactly as defined in `grpo_config.py`
- Keep all `# === GENERIC — DO NOT MODIFY ===` sections unchanged
- Make sure the code is compatible with accelerate for multi-GPU training
- Design a logging system, `train.logs` contains training statistics like loss, eval, lr, gradNorm and so on. `Sample_EP.logs` contains one full sample episode of prompt, thinking_output, action_output and states of environment and do not need to log every episode, to reduce log size, log like every 10 episode.

### Step 4: Generate `launch_training.sh`

Run some linux cmd to check the environment like gpu configs and then generate a SLURM job script that uses `accelerate launch` for training of `train.py`.

## Debug Workflow (after generation)

### Phase 1 — Import Check

Test each generated file imports cleanly:

```
python -c "import helpers"
python -c "import prompt_template"
python -c "import train"
```

Fix any ImportError / SyntaxError / NameError by editing the generated file directly.

### Phase 2 — Runtime Check

Run `launch_training.sh` and config training episode to a small number like 10 and let it execute for a few training steps. Fix any runtime errors (shape mismatches, missing config fields, wrong function signatures, OOM by size of macro batch etc.) and re-run until it completes without crashing.

## Iteration Workflow (after debug)

Set configs for train.py via `grpo_config.py` or cmd parameters(if available) in `launch_training.sh` which you think is suitbale for current gpus and a formal training. Use model `Qwen/Qwen3-4B-Instruct-2507`. Run `launch_training.sh` and monitor the console and `train.logs` and `Sample_EP.logs`. Then, do 2 things: 1) If you see some serious issues in `train.logs`, stop the training and search for a solution then modify files and restart. 2) `Sample_EP.logs` contains agents' inputs and outputs, also environment state transitions. Fix small issues like max_new_tokens is not enough for generation. Then go through it and think about high-level ideas about improvements. For example, multiple agents are targetting the same object during game, one of possible solutions is to divide all agents into 2 different job grounds via prompting. Put all ideas into `future_improvements.md` and DO NOT APPLY now.

## Rules

- Do **NOT** modify `env_move.py`, or `grpo_template.py`
- Do **NOT** add or delete parameters in `grpo_config.py`, but you can change values
- When fixing `train.py`, cross-reference `grpo_config.py` for correct field names and `helpers.py` for correct function signatures
- When fixing `prompt_template.py`, ensure action names match the functions in `helpers.py`
- During debug, edit generated files directly — do not regenerate from scratch unless asked
- I may run this task multiple times, to save claude tokens, for already-generated files, do not regenerate from scratch unless if needed, you could modify it.
- Give me a full log of claude code agent and put it into `claude.logs`, including the steps and actions and reason for actions.
