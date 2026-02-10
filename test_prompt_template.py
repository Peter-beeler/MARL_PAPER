"""
Test script for prompt_template.py.

Initializes a CleanupEnvMove environment, generates observations,
and builds prompts using create_thinking_prompt and create_single_stage_prompt
to verify they produce well-formed message lists.
"""

import json
from env_move import CleanupEnvMove, Config
import helpers
import prompt_template


def test_basic_prompt_structure():
    """Verify that both prompt functions return valid message lists."""
    env = CleanupEnvMove()
    obs = env.reset()

    for agent_id in range(1, env.cfg.n_agents + 1):
        obs_text = obs[agent_id]

        # --- Test create_thinking_prompt ---
        thinking_msgs = prompt_template.create_thinking_prompt(obs_text, agent_id)
        assert isinstance(thinking_msgs, list), "create_thinking_prompt should return a list"
        assert len(thinking_msgs) >= 2, "thinking prompt should have at least system + user messages"
        for msg in thinking_msgs:
            assert "role" in msg and "content" in msg, f"Each message must have 'role' and 'content': {msg}"
            assert msg["role"] in ("system", "user", "assistant"), f"Invalid role: {msg['role']}"

        # --- Test create_single_stage_prompt ---
        fake_thinking = "I see dirt nearby and apples are scarce. I should clean."
        single_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, agent_id)
        assert isinstance(single_msgs, list), "create_single_stage_prompt should return a list"
        assert len(single_msgs) >= 2, "single-stage prompt should have at least 2 messages"
        for msg in single_msgs:
            assert "role" in msg and "content" in msg, f"Each message must have 'role' and 'content': {msg}"

        print(f"  Agent {agent_id}: thinking_msgs={len(thinking_msgs)} msgs, "
              f"single_stage_msgs={len(single_msgs)} msgs  [OK]")

    print("[PASS] test_basic_prompt_structure\n")


def test_prompt_contains_actions():
    """Verify that prompts mention the high-level helper action names."""
    env = CleanupEnvMove()
    obs = env.reset()
    obs_text = obs[1]

    expected_actions = ["smart_clean_step", "smart_forage_step", "random_walk"]

    thinking_msgs = prompt_template.create_thinking_prompt(obs_text, 1)
    all_text = " ".join(msg["content"] for msg in thinking_msgs)
    for action_name in expected_actions:
        assert action_name in all_text, f"Thinking prompt missing action: {action_name}"

    fake_thinking = "There is dirt. I should clean."
    single_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, 1)
    all_text = " ".join(msg["content"] for msg in single_msgs)
    for action_name in expected_actions:
        assert action_name in all_text, f"Single-stage prompt missing action: {action_name}"

    print("[PASS] test_prompt_contains_actions\n")


def test_prompt_includes_observation():
    """Verify that the observation text is embedded in the user message."""
    env = CleanupEnvMove()
    obs = env.reset()
    obs_text = obs[1]

    thinking_msgs = prompt_template.create_thinking_prompt(obs_text, 1)
    user_msgs = [m["content"] for m in thinking_msgs if m["role"] == "user"]
    found = any(obs_text in content for content in user_msgs)
    assert found, "Observation text not found in thinking prompt user messages"

    fake_thinking = "I should forage."
    single_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, 1)
    all_text = " ".join(m["content"] for m in single_msgs)
    assert obs_text in all_text or fake_thinking in all_text, \
        "Neither observation nor thinking text found in single-stage prompt"

    print("[PASS] test_prompt_includes_observation\n")


def test_single_stage_includes_thinking():
    """Verify that create_single_stage_prompt includes the thinking response."""
    env = CleanupEnvMove()
    obs = env.reset()
    obs_text = obs[1]

    fake_thinking = "The river has lots of dirt. Cleaning is urgent."
    single_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, 1)

    # The thinking response should appear as an assistant message
    assistant_msgs = [m["content"] for m in single_msgs if m["role"] == "assistant"]
    assert any(fake_thinking in content for content in assistant_msgs), \
        "Thinking response not found in assistant message of single-stage prompt"

    print("[PASS] test_single_stage_includes_thinking\n")


def test_full_episode_loop():
    """Run a short episode using helpers, building prompts at each step."""
    env = CleanupEnvMove(Config(max_steps=10, seed=42))
    obs = env.reset()

    print("  Running 10-step episode with prompt generation...")
    for step in range(10):
        actions = {}
        for agent_id in range(1, env.cfg.n_agents + 1):
            obs_text = obs[agent_id]

            # Stage 1: build thinking prompt
            thinking_msgs = prompt_template.create_thinking_prompt(obs_text, agent_id)
            assert len(thinking_msgs) >= 2

            # Simulate a thinking response
            fake_thinking = "I see dirt nearby, I should clean."

            # Stage 2: build action prompt
            action_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, agent_id)
            assert len(action_msgs) >= 2

            # Use a helper action (instead of calling an LLM)
            actions[agent_id] = helpers.smart_clean_step(env, agent_id)

        obs, rewards, done, info = env.step(actions)
        print(f"    Step {step + 1}: dirt={info['dirt_count']} apples={info['apple_count']} "
              f"rewards={rewards}")
        if done:
            break

    print("[PASS] test_full_episode_loop\n")


def test_print_sample_prompts():
    """Print sample prompts for manual inspection."""
    env = CleanupEnvMove(Config(seed=42))
    obs = env.reset()

    agent_id = 1
    obs_text = obs[agent_id]

    print(f"  Agent {agent_id} observation:\n{obs_text}\n")

    # Thinking prompt
    thinking_msgs = prompt_template.create_thinking_prompt(obs_text, agent_id)
    print("  === THINKING PROMPT ===")
    for msg in thinking_msgs:
        print(f"  [{msg['role'].upper()}]")
        print(f"  {msg['content']}")
        print()

    # Single-stage prompt
    fake_thinking = "I see dirt at nearby cells. The river needs cleaning to spawn apples."
    single_msgs = prompt_template.create_single_stage_prompt(obs_text, fake_thinking, agent_id)
    print("  === SINGLE-STAGE (ACTION) PROMPT ===")
    for msg in single_msgs:
        print(f"  [{msg['role'].upper()}]")
        print(f"  {msg['content']}")
        print()

    print("[PASS] test_print_sample_prompts\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing prompt_template.py")
    print("=" * 60 + "\n")

    test_basic_prompt_structure()
    test_prompt_contains_actions()
    test_prompt_includes_observation()
    test_single_stage_includes_thinking()
    test_full_episode_loop()
    test_print_sample_prompts()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
