# GRPO Multi-Agent RL Training — Cleanup

## Project Overview

GRPO-based multi-agent reinforcement learning training. An LLM plays a **Cleanup** game (`env/cleanup.py`) using high-level helper actions. The training pipeline is already implemented and working.

**Game Summary:** A grid with land and a central river. Dirt spawns on the river and blocks apple spawning on land. Agents must clean dirt to enable apples, then eat apples for reward. The dilemma: cleaning benefits everyone but gives no direct reward; eating gives reward but only if someone else cleans.

## Architecture

```
grpo_textgame.py          ← Main trainer (CleanupGameGRPO class)
├── utils/
│   ├── config.py          ← GRPOConfig dataclass
│   ├── args.py            ← CLI argument parser → fills GRPOConfig
│   ├── model_setup.py     ← Accelerator, tokenizer, model, LoRA, optimizer
│   ├── observation.py     ← Obs-to-text, high-level actions, scan helpers, role obs
│   ├── prompts.py         ← Chat-template prompt builders (text/compound/role modes)
│   ├── generation.py      ← Action generation (text two-stage, single-stage, compound)
│   ├── rollout.py         ← Episode rollout, parallel envs, trajectory collection
│   ├── eval.py            ← Evaluation on fixed states, visualization
│   ├── loss.py            ← GRPO/DrGRPO advantages, minibatch iterator, policy loss
│   └── logprob.py         ← Batched sequence log-probability computation
├── env/
│   ├── cleanup.py         ← CleanupEnvMove (vanilla rewards, used for eval)
│   └── cleanup_copy.py    ← CleanupEnvMove (shaped rewards, used for training)
├── grpo_config.py         ← Standalone config with defaults (template)
├── grpo_template.py       ← Reference template showing GRPO algorithm flow
└── launch_textgame.sh     ← SLURM job script
```

### Data Flow

```
CLI args → GRPOConfig → CleanupGameGRPO trainer
  ├── Model: load_base_model → setup_model_for_training (LoRA) → optimizer/scheduler
  ├── Env:   TrainEnvClass(Config) for rollout, EvalEnvClass(Config) for eval
  └── Loop:
      1. run_parallel_episodes(trainer, num_envs)
         ├── env.reset() → obs_dict
         ├── For each step:
         │   ├── generate_actions_multi_env_batch(trainer, env_data, step, model)
         │   │   ├── obs_to_text(obs, env, agent_id, config) → observation string
         │   │   ├── build_action_prompt(obs_text, ...) → chat template string
         │   │   ├── model.generate() → response text
         │   │   └── parse_and_execute_action(response, agent_id, env) → low-level action
         │   └── env.step(actions) → (obs, rewards, done, info)
         └── Returns: trajectories [{prompts, actions, rewards, log_probs, ...}]
      2. compute_advantages(trajectories) → normalized advantages
      3. Inner epochs: flatten → minibatches → compute_loss_on_samples → backward
      4. evaluate(trainer) every eval_interval episodes
```

## Source-of-Truth Files (DO NOT MODIFY)

- `env/cleanup.py` — Game environment
- `env/cleanup_copy.py` — Shaped-reward variant
- `grpo_template.py` — Reference template

## Files You Customize

These files contain **game-specific logic** that must match the environment:

### `utils/observation.py` — Observations & Actions

- `parse_observation_to_coords(obs, agent_id, env)` — Text mode: local 5x3 window → natural language
- `get_observation_description(env, agent_id)` — Compound mode: position + nearby items + agents
- `scan_nearest_dirt(env, agent_id)` / `scan_nearest_apple(env, agent_id)` — Global scan helpers
- `scan_dirt_count(env, agent_id)` / `scan_apple_count(env, agent_id)` — Item count helpers
- **High-level actions** (compound mode, each returns `(low_level_action: str, is_done: bool)`):
  - `move_to(env, agent_id, coord_x, coord_y)` — Move toward target
  - `clean_at(env, agent_id, coord_x, coord_y)` — Move to dirt and clean
  - `eat_at(env, agent_id, coord_x, coord_y)` — Move to apple and eat
  - `random_explore(env, agent_id)` — Random movement
  - `find_nearest_apples(env, agent_id)` — Tool action: scan map (agent stays)
  - `find_nearest_dirts(env, agent_id)` — Tool action: scan map (agent stays)
- `TOOL_ACTIONS` — Set of tool action names (agent stays in place, result in next obs)
- `get_role_specific_observation(env, agent_id, role)` — Role-tailored observation
- `get_global_role_context(env)` — Summary for role-assignment meta-call
- `check_action_continuation(env, agent_id, last_action)` — Multi-step action continuation hints

### `utils/prompts.py` — Prompt Templates

- `create_thinking_prompt(obs_text, agent_id, config, tokenizer)` — Text mode stage 1
- `create_action_prompt(obs_text, thinking_text, config, tokenizer)` — Text mode stage 2
- `create_single_stage_prompt_text(obs_text, config, tokenizer)` — Text single-stage
- `create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)` — Compound mode
- `EATER_COMPOUND_CONTEXT` / `CLEANER_COMPOUND_CONTEXT` — Role-specific prompt templates
- `ROLE_ASSIGNMENT_SYSTEM` — Coordinator system prompt
- `create_role_compound_prompt(obs_text, config, tokenizer, env, agent_id, role)` — Role dispatch
- `create_role_assignment_prompt(global_context, current_roles, num_agents, tokenizer)` — Role assignment
- `build_thinking_prompt()` / `build_action_prompt()` — Dispatch functions

### `grpo_textgame.py` — Trainer Constants

- Line ~40: Env imports (`EvalEnvClass`, `TrainEnvClass`, `EnvConfigMove`)
- `ACTIONS` dict and helper function list in trainer setup
- `_perform_role_assignment()` — Role assignment logic

## Key Game Mechanics

- **Grid:** 15x9, land (`*`) with centered river (`x`, 3 cols wide)
- **Items:** Apples (`a`) on land, dirt (`#`) on water
- **Actions:** `0=stay, 1=eat, 2=clean, 3=up, 4=down, 5=left, 6=right` (7 total)
- **Rewards:** Eat apple = +eat_reward (1.0), clean dirt = +0.0 (shaped in cleanup_copy.py)
- **Spawning:** Apples spawn on land only when dirt count < initial_dirt_count
- **Observations:** 5x3 local window per agent
- **Coordinates:** (x, y) display coords where y increases upward: `display_y = (height-1) - internal_y`

### Env Interface

```python
env.agents: Dict[int, Tuple[int, int]]   # agent_id → (x, y) internal coords
env.items: List[List[str|None]]           # 'a', '#', or None
env.terrain: List[List[str]]              # '*' or 'x'
env.scores: Dict[int, float]
env.init_dirt_count: int
env._count_items(char) → int
env._find_nearest_items(agent_id, char, n=5) → List[(x, y)]
```

## Dynamic Role Switching

### Roles: Eater vs Cleaner

- **Eater** — Finds and eats apples. Actions: `move_to`, `eat_at`, `find_nearest_apples`, `random_explore`
- **Cleaner** — Removes dirt from river. Actions: `move_to`, `clean_at`, `find_nearest_dirts`, `random_explore`

### Assignment Logic

- Considers: dirt count vs initial, apple count, agent positions/scores
- High dirt → more cleaners. Many apples → more eaters. Always >=1 cleaner if dirt remains.
- Output: `{"1": "eater", "2": "cleaner", "3": "eater"}`

### Flow in Code

1. `grpo_textgame.py:_perform_role_assignment(envs, role_states)` called at `role_assignment_interval` steps
2. Calls `utils/observation.py:get_global_role_context(env)` for game summary
3. Calls `utils/prompts.py:create_role_assignment_prompt()` for the meta-call
4. Parses JSON role mapping, updates `role_states[i]['roles']`
5. On each step, `generation.py` checks role and calls `get_role_specific_observation()` + `create_role_compound_prompt()`

## Debug Workflow

### Phase 1 — Import Check
```
python -c "from env.cleanup import CleanupEnvMove, Config"
python -c "from utils.observation import obs_to_text, move_to, clean_at, eat_at"
python -c "from utils.prompts import build_action_prompt, create_role_assignment_prompt"
```

### Phase 2 — Runtime Check
Run `launch_textgame.sh` with small episode count. Fix runtime errors and re-run.

## Iteration Workflow

Use model `Qwen/Qwen3-4B-Instruct-2507`. Monitor `train.logs` and `Sample_EP.logs`. Fix issues, put improvement ideas in `future_improvements.md`.

## Rules

- Do **NOT** modify `env/cleanup.py`, `env/cleanup_copy.py`, or `grpo_template.py`
- Do **NOT** add or delete parameters in `grpo_config.py`, but you can change values
- When fixing `utils/observation.py`, ensure action function signatures match what `generation.py` expects: `fn(env, agent_id, **kwargs) → (str, bool)`
- When fixing `utils/prompts.py`, ensure action names in prompts match functions in `utils/observation.py`
- During debug, edit files directly — do not regenerate from scratch unless asked
- Give me a full log of claude code agent and put it into `claude.logs`
