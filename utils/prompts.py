"""
Prompt builders for both text mode and compound mode.
"""

from typing import Optional
from .observation import find_nearest_apple, find_nearest_dirt


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def _clean_reward_desc(config) -> str:
    """Return a reward description string for cleaning, based on config."""
    if config.clean_reward > 0.0:
        return (
            f"- Cleaning dirt gives +{config.clean_reward} reward AND enables apple spawning "
            f"(less dirt = more apples). "
        )
    return "- Cleaning dirt itself gives NO points, but is necessary to enable apple spawning. "


def _get_action_api_compound() -> str:
    """Return documentation for compound mode high-level actions."""
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
You must output **ONLY** a valid JSON object.
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


# ─────────────────────────────────────────────
# TEXT MODE PROMPTS
# ─────────────────────────────────────────────

def create_thinking_prompt(obs_text: str, agent_id: int, config, tokenizer) -> str:
    """
    Stage 1 prompt (text mode): ask agent to reason about the situation.

    Args:
        obs_text: Pre-formatted coordinate-based observation text.
        agent_id: Agent ID.
        config: GRPOConfig.
        tokenizer: Tokenizer for applying chat template.

    Returns:
        Chat-template formatted prompt string.
    """
    system_msg = (
        f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{config.eat_reward} each). "
        "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
        + _clean_reward_desc(config) +
        "- You can only eat/clean items at your position. "
        "Available actions: "
        "up = move one step up (increase y) "
        "down = move one step down (decrease y) "
        "left = move one step left (decrease x) "
        "right = move one step right (increase x) "
        "clean = clean dirt if on your cell "
        "eat = eat apple if on your cell "
        "stay = stay in current position. "
        "Think about the situation and give your reasoning in short."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": obs_text}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def create_action_prompt(obs_text: str, thinking_text: str, config, tokenizer) -> str:
    """
    Stage 2 prompt (text mode two-stage): given thinking, output ONE action word.

    Args:
        obs_text: Pre-formatted observation text.
        thinking_text: Generated thinking/reasoning from stage 1.
        config: GRPOConfig.
        tokenizer: Tokenizer for applying chat template.

    Returns:
        Chat-template formatted prompt string.
    """
    system_msg = (
        f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{config.eat_reward} each). "
        "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
        + _clean_reward_desc(config) +
        "- You can only eat/clean items at your position. "
        "Available actions: "
        "up = move one step up (increase y) "
        "down = move one step down (decrease y) "
        "left = move one step left (decrease x) "
        "right = move one step right (increase x) "
        "clean = clean dirt if on your cell "
        "eat = eat apple if on your cell "
        "stay = stay in current position "
    )

    action_instruction = (
        "Based on your thinking above, choose your best immediate action and output only ONE action word: "
        "up, down, left, right, clean, eat, or stay."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": obs_text},
        {"role": "assistant", "content": thinking_text},
        {"role": "user", "content": action_instruction}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def create_single_stage_prompt_text(obs_text: str, config, tokenizer) -> str:
    """
    Single-stage prompt (text mode): output ONE action word directly.

    Args:
        obs_text: Pre-formatted observation text.
        config: GRPOConfig.
        tokenizer: Tokenizer for applying chat template.

    Returns:
        Chat-template formatted prompt string.
    """
    clean_line = (
        f"- Cleaning dirt: +{config.clean_reward} reward for you AND enables apple spawning\n"
        if config.clean_reward > 0.0
        else "- Cleaning dirt: no immediate reward, but enables apple spawning\n"
    )
    system_msg = (
        "You are an agent in a multi-agent cleanup game. The game has:\n"
        "- A river with dirt (#) that can be cleaned\n"
        "- Land with apples (a) that can be eaten\n"
        "- Multiple agents (numbered 1, 2, 3, ...) working together\n\n"
        "Actions: up, down, left, right, clean (remove dirt), eat (consume apple), stay.\n\n"
        "Rewards:\n"
        + clean_line +
        f"- Eating apple: +{config.eat_reward} reward for you\n"
        "- Dirt respawns if river isn't clean enough\n\n"
        "Output ONLY ONE action word: up, down, left, right, clean, eat, or stay."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": obs_text}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────
# COMPOUND MODE PROMPTS
# ─────────────────────────────────────────────

def create_single_stage_prompt_compound(
    obs_text: str, config, tokenizer, env=None, agent_id: int = 0
) -> str:
    """
    Single-stage prompt (compound mode): think then output ONE JSON action.

    Args:
        obs_text: Natural-language observation from get_observation_description.
        config: GRPOConfig.
        tokenizer: Tokenizer for applying chat template.
        env: Environment instance (for nearest apple/dirt scan).
        agent_id: Agent ID.

    Returns:
        Chat-template formatted prompt string.
    """
    if config.clean_reward > 0.0:
        clean_desc = (
            f"Cleaning dirt gives +{config.clean_reward} reward AND enables more apple spawning. "
        )
        decision_clean = f"2. Should you clean dirt for +{config.clean_reward} reward AND to ensure future apples spawn?"
    else:
        clean_desc = "Cleaning dirt gives no immediate reward but is necessary to enable apple spawning. "
        decision_clean = "2. Should you clean dirt to ensure future apples spawn (helping the group)?"

    system_context = (
        f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{config.eat_reward} each). "
        + clean_desc
        + "You cannot move diagonally. You have high-level functions to navigate and interact."
    )

    system_content = system_context + "\n\n" + _get_action_api_compound()

    # Global scan info
    scan_section = ""
    if env is not None:
        nearest_apple = find_nearest_apple(env, agent_id)
        nearest_dirt = find_nearest_dirt(env, agent_id)

        strategy_info = []
        if nearest_apple['found']:
            strategy_info.append(
                f"- Nearest Apple: at ({nearest_apple['coord_x']}, {nearest_apple['coord_y']}), "
                f"distance {nearest_apple['distance']}."
            )
        else:
            strategy_info.append("- Nearest Apple: None found.")

        if nearest_dirt['found']:
            strategy_info.append(
                f"- Nearest Dirt: at ({nearest_dirt['coord_x']}, {nearest_dirt['coord_y']}), "
                f"distance {nearest_dirt['distance']}."
            )
        else:
            strategy_info.append("- Nearest Dirt: None found.")

        strategy_str = "\n".join(strategy_info)
        scan_section = f"\n\n### GLOBAL SCAN\n{strategy_str}"

    user_content = (
        f"You are Agent {agent_id}.\n\n"
        f"### CURRENT OBSERVATION\n{obs_text}"
        f"{scan_section}\n\n"
        f"### INSTRUCTIONS\n"
        f"First, briefly think through the trade-off:\n"
        f"1. Should you eat an apple for immediate reward?\n"
        f"{decision_clean}\n"
        f"3. If nothing is nearby, where should you explore?\n\n"
        f"Then think about it and output ONE valid JSON action object."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def build_thinking_prompt(obs_text: str, agent_id: int, config, tokenizer) -> str:
    """Build thinking prompt (text mode only; compound mode uses single-stage)."""
    return create_thinking_prompt(obs_text, agent_id, config, tokenizer)


def build_action_prompt(
    obs_text: str, thinking_text: str, config, tokenizer,
    env=None, agent_id: int = 0
) -> str:
    """
    Build action prompt based on action_mode.
    - text mode (two-stage): create_action_prompt (stage 2)
    - text mode (single-stage): create_single_stage_prompt_text
    - compound mode: create_single_stage_prompt_compound
    """
    if config.action_mode == "compound":
        return create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)
    elif config.use_two_stage:
        return create_action_prompt(obs_text, thinking_text, config, tokenizer)
    else:
        return create_single_stage_prompt_text(obs_text, config, tokenizer)
