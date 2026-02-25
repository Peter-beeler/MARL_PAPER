"""
Evaluation and visualization for both text and compound modes.
"""

import logging
import time
import numpy as np
import torch
from typing import Optional, List

import wandb

logger = logging.getLogger(__name__)


def _generate_eval_states(env_config, config, num_states: int = 20) -> List:
    """Pre-generate fixed initial states for evaluation."""
    from env_move import CleanupEnvMove, Config as EnvConfigMove

    eval_states = []
    for i in range(num_states):
        eval_env = CleanupEnvMove(EnvConfigMove(
            n_agents=config.num_agents,
            max_steps=config.max_env_steps,
            seed=config.seed + 1000 + i,
            eat_reward=config.eat_reward,
            clean_reward=config.clean_reward,
        ))
        eval_env.reset()
        state = eval_env.get_state()
        eval_states.append(state)

    return eval_states


def evaluate(trainer, num_episodes: int = 20, current_episode: int = None):
    """
    Evaluate the trained model on fixed initial states.

    Args:
        trainer: CleanupGameGRPO instance.
        num_episodes: Number of episodes to evaluate.
        current_episode: Current training episode (for wandb logging).

    Returns:
        (avg_reward, std_reward)
    """
    from .rollout import run_episode

    config = trainer.config
    accelerator = trainer.accelerator

    eval_start_time = time.time()
    actual_num_episodes = min(num_episodes, len(trainer.eval_states))

    if accelerator is None or accelerator.is_main_process:
        logger.info(f"\n=== Evaluation ({actual_num_episodes} episodes) ===")

    trainer.model.eval()

    if accelerator is not None:
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index
        local_episode_indices = list(range(process_index, actual_num_episodes, num_processes))
        if accelerator.is_main_process:
            logger.info(f"Splitting {actual_num_episodes} episodes across {num_processes} GPUs (round-robin)")
    else:
        process_index = 0
        local_episode_indices = list(range(actual_num_episodes))

    local_rewards = []
    local_episode_times = []

    for i in local_episode_indices:
        try:
            initial_state = trainer.eval_states[i]
            traj = run_episode(trainer, use_ref_model=False, log_samples=False, initial_state=initial_state)
            local_rewards.append(traj["total_reward"])
            local_episode_times.append(traj["rollout_time"])
            logger.info(
                f"  GPU{process_index} Ep{i + 1}: R={traj['total_reward']:.2f}, "
                f"Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s"
            )
            del traj
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"GPU {process_index}: Evaluation episode {i + 1} failed: {e}")
            continue

    # Gather results from all GPUs
    if accelerator is not None:
        model_device = next(trainer.model.parameters()).device
        local_len = len(local_rewards)
        max_len_tensor = torch.tensor([local_len], dtype=torch.long, device=model_device)
        all_lens = accelerator.gather(max_len_tensor)

        max_len = all_lens.max().item()
        local_rewards_padded = local_rewards + [0.0] * (max_len - len(local_rewards))
        local_times_padded = local_episode_times + [0.0] * (max_len - len(local_episode_times))

        all_rewards = accelerator.gather(
            torch.tensor(local_rewards_padded, dtype=torch.float32, device=model_device)
        )
        all_times = accelerator.gather(
            torch.tensor(local_times_padded, dtype=torch.float32, device=model_device)
        )

        if accelerator.is_main_process:
            rewards = []
            episode_times = []
            for proc_idx in range(accelerator.num_processes):
                start_idx = proc_idx * max_len
                actual_len = all_lens[proc_idx].item()
                rewards.extend(all_rewards.cpu().tolist()[start_idx:start_idx + actual_len])
                episode_times.extend(all_times.cpu().tolist()[start_idx:start_idx + actual_len])
        else:
            rewards = []
            episode_times = []
    else:
        rewards = local_rewards
        episode_times = local_episode_times

    if len(rewards) == 0:
        trainer.model.train()
        return 0.0, 0.0

    if accelerator is None or accelerator.is_main_process:
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        total_eval_time = time.time() - eval_start_time
        avg_episode_time = np.mean(episode_times) if episode_times else 0.0

        logger.info(
            f"\nReward: {avg_reward:.2f}±{std_reward:.2f} [{min(rewards):.2f}, {max(rewards):.2f}]"
        )
        logger.info(f"Evaluation Time: avg={avg_episode_time:.2f}s/episode, total={total_eval_time:.2f}s")

        if config.use_wandb and current_episode is not None:
            wandb.log({
                "eval/reward_mean": avg_reward, "eval/reward_std": std_reward,
                "eval/reward_min": min(rewards), "eval/reward_max": max(rewards),
                "eval/episode_time": avg_episode_time, "eval/total_time": total_eval_time,
                "eval/num_episodes": len(rewards),
            }, step=current_episode)
    else:
        avg_reward = 0.0
        std_reward = 0.0

    trainer.model.train()
    return avg_reward, std_reward


def visualize_rollout(trainer, use_ref_model: bool = False, save_to_file: Optional[str] = None):
    """
    Visualize a single rollout step-by-step (both modes).

    Args:
        trainer: CleanupGameGRPO instance.
        use_ref_model: If True, use reference model.
        save_to_file: Optional filepath to save visualization.
    """
    from env_move import CleanupEnvMove
    from .generation import generate_actions_batch
    from .observation import obs_to_text

    config = trainer.config
    accelerator = trainer.accelerator

    if accelerator is not None and not accelerator.is_main_process:
        return

    rollout_start_time = time.time()

    mode_label = "COMPOUND JSON ACTIONS" if config.action_mode == "compound" else "TEXT ACTIONS"
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAJECTORY VISUALIZATION ({mode_label})")
    logger.info("=" * 80)

    env = CleanupEnvMove(trainer.env_config)
    obs = env.reset()
    initial_dirt_count = sum(row.count('#') for row in env.items)

    if use_ref_model and getattr(trainer, 'ref_model', None) is None:
        logger.warning("Reference model not available. Using current policy.")
        model = trainer.model
    else:
        model = trainer.ref_model if use_ref_model else trainer.model
    model.eval()

    total_reward = 0
    output_lines = []
    step_times = []

    def log_and_save(line):
        logger.info(line)
        output_lines.append(line)

    log_and_save(f"\n{'=' * 80}")
    log_and_save("INITIAL STATE")
    log_and_save(f"{'=' * 80}")
    log_and_save("\nGlobal Grid:")
    for line in env.render().split('\n'):
        log_and_save(f"  {line}")
    log_and_save("")

    for step in range(config.max_env_steps):
        step_start_time = time.time()

        log_and_save(f"\n{'─' * 80}")
        log_and_save(f"STEP {step + 1}/{config.max_env_steps}")
        log_and_save(f"{'─' * 80}")

        actions = {}
        step_info = []

        batch_results = generate_actions_batch(trainer, obs, step, env, model)

        for agent_id in range(1, config.num_agents + 1):
            ax, ay = env.agents[agent_id]

            (action, log_prob, thinking_text, full_response, action_text,
             action_prompt, action_input_ids, action_ids) = batch_results[agent_id]
            actions[agent_id] = action

            obs_nl = obs_to_text(obs[agent_id], env, agent_id, config)

            step_info.append({
                'agent_id': agent_id,
                'position': (ax, ay),
                'obs_nl': obs_nl,
                'thinking': thinking_text.strip() if thinking_text else '',
                'response': full_response.strip() if full_response else '',
                'action_text': action_text.strip() if action_text else '',
                'action_prompt': action_prompt,
                'action': action,
                'log_prob': log_prob.item() if torch.is_tensor(log_prob) else float(log_prob),
            })

        log_and_save("\nAgent Decisions:")
        for info in step_info:
            log_and_save(f"\n  Agent {info['agent_id']} at {info['position']}:")
            log_and_save(f"    Observation: {info['obs_nl']}")

            if config.action_mode == "compound":
                log_and_save(f"\n    Response (thinking + JSON):")
                for line in info['response'].split('\n'):
                    log_and_save(f"    {line}")
                log_and_save(f"\n    --- RAW RESPONSE ---")
                log_and_save(f"    {repr(info['response'])}")
                log_and_save(f"    --- END RAW RESPONSE ---")
                log_and_save(f"    Action (parsed from JSON): {info['action']}")
            else:
                if config.use_two_stage:
                    log_and_save(f"\n    Thinking: '{info['thinking']}'")
                    log_and_save(f"    --- RAW THINKING ---")
                    log_and_save(f"    {repr(info['thinking'])}")
                    log_and_save(f"    --- END RAW THINKING ---")
                    if info['action_prompt']:
                        log_and_save(f"\n    --- STAGE 2 PROMPT ---")
                        for line in info['action_prompt'].split('\n'):
                            log_and_save(f"    {line}")
                        log_and_save(f"    --- END STAGE 2 PROMPT ---")
                    log_and_save(f"    Action text (stage 2): '{info['action_text']}'")
                else:
                    log_and_save(f"\n    Response: '{info['response']}'")
                    log_and_save(f"    --- RAW RESPONSE ---")
                    log_and_save(f"    {repr(info['response'])}")
                    log_and_save(f"    --- END RAW RESPONSE ---")
                log_and_save(f"    Action (parsed): {info['action']}")

            log_and_save(f"    Log Prob: {info['log_prob']:.4f}")

        obs, rewards, done, info = env.step(actions)

        log_and_save("\n  Step Results:")
        step_reward = sum(rewards.values())
        total_reward += step_reward

        for agent_id in range(1, config.num_agents + 1):
            if rewards[agent_id] > 0:
                log_and_save(f"    Agent {agent_id}: +{rewards[agent_id]:.1f} points!")

        log_and_save(f"    Step reward: {step_reward:.1f}")
        log_and_save(f"    Total reward: {total_reward:.1f}")
        log_and_save(f"    Dirt remaining: {info['dirt_count']}")
        log_and_save(f"    Apples available: {info['apple_count']}")

        log_and_save("\n  Grid After Step:")
        for line in env.render().split('\n'):
            log_and_save(f"    {line}")

        step_time = time.time() - step_start_time
        step_times.append(step_time)
        log_and_save(f"\n  Step Time: {step_time:.2f}s")

        if done:
            log_and_save("  Episode ended (max steps reached)")
            break

    total_rollout_time = time.time() - rollout_start_time
    avg_step_time = np.mean(step_times) if step_times else 0.0

    log_and_save(f"\n{'=' * 80}")
    log_and_save("EPISODE SUMMARY")
    log_and_save(f"{'=' * 80}")
    log_and_save(f"Total Reward: {total_reward:.2f}")
    log_and_save(f"Steps Taken: {step + 1}")
    log_and_save(f"Final Scores: {info['scores']}")
    log_and_save(f"Dirt Cleaned: {initial_dirt_count - info['dirt_count']}")
    log_and_save(f"\nTiming:")
    log_and_save(f"  Average Step Time: {avg_step_time:.2f}s")
    log_and_save(f"  Total Rollout Time: {total_rollout_time:.2f}s")
    log_and_save(f"{'=' * 80}\n")

    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"Visualization saved to: {save_to_file}")

    model.train()
