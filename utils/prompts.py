"""
Prompt builders for both text mode and compound mode.

This file builds the chat-formatted prompts that are sent to the LLM.
Each function returns a string produced by tokenizer.apply_chat_template().

Read the game environment file to understand game rules, mechanics, rewards,
and available actions before implementing the prompts.
Read observation.py to see what helper actions are available (compound mode)
and what observation text looks like (both modes).
"""

from typing import Optional


# ─────────────────────────────────────────────
# TEXT MODE PROMPTS
# ─────────────────────────────────────────────

def create_thinking_prompt(obs_text: str, agent_id: int, config, tokenizer) -> str:
    """
    Stage 1 prompt (text mode two-stage): ask the agent to reason about the situation.

    Should include:
      - Game context (rules, rewards, mechanics)
      - Available low-level actions
      - Instruction to think/reason about the observation

    Args:
        obs_text: Pre-formatted observation text from observation.py.
        agent_id: Agent ID.
        config: GRPOConfig (has reward values, game settings, etc.).
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    system_msg = (
        f"You play a grid cleanup game as agent {agent_id}. "
        f"Grid has land(*) and a river(water x) in the middle. "
        f"Dirt(#) spawns on water. Apples(a) spawn on land only when dirt decreases below initial amount. "
        f"Clean dirt to enable apple spawning. Eat apples for +{config.eat_reward} reward. "
        f"Actions: up, down, left, right, clean, eat, stay. "
        f"'clean' works only on dirt at your position. 'eat' works only on apple at your position. "
        f"Coordinates: (x,y) where x=left-right, y=bottom-top. "
        f"Think briefly about what to do next."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Observation: {obs_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def create_action_prompt(obs_text: str, thinking_text: str, config, tokenizer) -> str:
    """
    Stage 2 prompt (text mode two-stage): given thinking, output ONE action word.

    Should include:
      - Game context
      - The original observation
      - The thinking output from stage 1 (as assistant message)
      - Instruction to output exactly one action word

    Args:
        obs_text: Pre-formatted observation text.
        thinking_text: Generated thinking/reasoning from stage 1.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    system_msg = (
        f"You play a grid cleanup game. "
        f"Clean dirt on water to enable apple spawning. Eat apples for +{config.eat_reward} reward. "
        f"Actions: up, down, left, right, clean, eat, stay. "
        f"Output exactly ONE action word, nothing else."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Observation: {obs_text}"},
        {"role": "assistant", "content": thinking_text},
        {"role": "user", "content": "Now output exactly ONE action word from: up, down, left, right, clean, eat, stay."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def create_single_stage_prompt_text(obs_text: str, config, tokenizer) -> str:
    """
    Single-stage prompt (text mode): output ONE action word directly.

    Should include:
      - Game context (rules, rewards, mechanics)
      - Available low-level actions
      - Instruction to output exactly one action word

    Args:
        obs_text: Pre-formatted observation text.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().

    Returns:
        Chat-template formatted prompt string.
    """
    system_msg = (
        f"You play a grid cleanup game. "
        f"Grid has land(*) and river(water x). Dirt(#) on water blocks apple spawning. "
        f"Clean dirt to enable apples. Eat apples for +{config.eat_reward} reward. "
        f"Actions: up, down, left, right, clean, eat, stay. "
        f"'clean' works on dirt at your position. 'eat' works on apple at your position. "
        f"Coords: (x,y) where y increases upward. "
        f"Output exactly ONE action word."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Observation: {obs_text}\nOutput ONE action word:"},
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

    Should include:
      - Game context (rules, rewards, mechanics)
      - Documentation of available high-level actions from observation.py
        (function names, arguments, JSON format)
      - Optionally: global scan info from observation.py helpers
      - Instruction to reason briefly then output one JSON action object

    Args:
        obs_text: Natural-language observation from get_observation_description.
        config: GRPOConfig.
        tokenizer: Tokenizer with apply_chat_template().
        env: Environment instance (for calling scan/info helpers).
        agent_id: Agent ID.

    Returns:
        Chat-template formatted prompt string.
    """
    from .observation import scan_nearest_dirt, scan_nearest_apple, scan_dirt_count, scan_apple_count

    system_msg = (
        f"You play a grid cleanup game. "
        f"Grid: {env.width}x{env.height}, land(*) with river(water x) in center. "
        f"Dirt(#) on water blocks apple spawning. Clean dirt to enable apples. "
        f"Eat apples for +{config.eat_reward} reward. Coords: (x,y), y increases upward.\n"
        f"Available actions (output as JSON):\n"
        f'- move_to: move toward target. {{"action":"move_to","args":{{"coord_x":X,"coord_y":Y}}}}\n'
        f'- clean_at: move to dirt and clean it. {{"action":"clean_at","args":{{"coord_x":X,"coord_y":Y}}}}\n'
        f'- eat_at: move to apple and eat it. {{"action":"eat_at","args":{{"coord_x":X,"coord_y":Y}}}}\n'
        f'- random_explore: random move. {{"action":"random_explore","args":{{}}}}\n'
        f"Think briefly, then output ONE JSON action."
    )

    # Add global scan info to user message
    extra = []
    if env is not None:
        dirt_info = scan_nearest_dirt(env, agent_id)
        apple_info = scan_nearest_apple(env, agent_id)
        dc = scan_dirt_count(env, agent_id)
        ac = scan_apple_count(env, agent_id)
        if dirt_info['found']:
            extra.append(f"Nearest dirt: ({dirt_info['x']},{dirt_info['y']}) dist={dirt_info['distance']}")
        if apple_info['found']:
            extra.append(f"Nearest apple: ({apple_info['x']},{apple_info['y']}) dist={apple_info['distance']}")
        extra.append(f"Total dirt: {dc['count']}, Total apples: {ac['count']}")

    user_content = f"Observation: {obs_text}"
    if extra:
        user_content += "\nGlobal info: " + ". ".join(extra) + "."

    messages = [
        {"role": "system", "content": system_msg},
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
