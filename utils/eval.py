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
    """Pre-generate random initial states for evaluation."""
    from env.cleanup import CleanupEnvMove, Config as EnvConfigMove

    eval_states = []
    for i in range(num_states):
        eval_env = CleanupEnvMove(EnvConfigMove(
            n_agents=config.num_agents,
            max_steps=config.max_env_steps,
            seed=None,  # fully random
            eat_reward=config.eat_reward,
        ))
        eval_env.reset()
        state = eval_env.get_state()
        eval_states.append(state)

    return eval_states


def evaluate(trainer, num_episodes: int = 20, current_episode: int = None):
    """
    Evaluate the trained model on fixed initial states using parallel rollout.
    Logs all eval trajectories to file when config.log_trajectory is True.

    Args:
        trainer: CleanupGameGRPO instance.
        num_episodes: Number of episodes to evaluate.
        current_episode: Current training episode (for wandb logging).

    Returns:
        (avg_reward, std_reward)
    """
    from .rollout import run_episode, run_parallel_episodes, log_episode_to_file

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
    local_trajectories = []  # collect all eval trajectories for logging

    # Determine parallel batch size for eval
    parallel_envs = config.parallel_envs if config.parallel_envs > 0 else (
        config.episodes_per_gpu if accelerator is not None else config.episodes_per_update
    )

    # Process local episodes in parallel batches
    idx = 0
    while idx < len(local_episode_indices):
        batch_indices = local_episode_indices[idx:idx + parallel_envs]
        batch_states = [trainer.eval_states[i] for i in batch_indices]

        try:
            batch_trajs = run_parallel_episodes(
                trainer,
                num_envs=len(batch_states),
                use_ref_model=False,
                log_samples=False,
                initial_states=batch_states,
            )
            for j, traj in enumerate(batch_trajs):
                ep_idx = batch_indices[j]
                local_rewards.append(traj["total_reward"])
                local_episode_times.append(traj["rollout_time"])
                local_trajectories.append((ep_idx, traj))
                logger.info(
                    f"  GPU{process_index} Ep{ep_idx + 1}: R={traj['total_reward']:.2f}, "
                    f"Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s"
                )
        except Exception as e:
            logger.warning(f"GPU{process_index}: Parallel eval batch failed: {e}, falling back to sequential")
            for ep_idx in batch_indices:
                try:
                    initial_state = trainer.eval_states[ep_idx]
                    traj = run_episode(trainer, use_ref_model=False, log_samples=False, initial_state=initial_state)
                    local_rewards.append(traj["total_reward"])
                    local_episode_times.append(traj["rollout_time"])
                    local_trajectories.append((ep_idx, traj))
                    logger.info(
                        f"  GPU{process_index} Ep{ep_idx + 1}: R={traj['total_reward']:.2f}, "
                        f"Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s"
                    )
                except Exception as e2:
                    logger.error(f"GPU {process_index}: Evaluation episode {ep_idx + 1} failed: {e2}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        idx += len(batch_indices)

    # Log all eval trajectories to file (main process only)
    if config.log_trajectory and (accelerator is None or accelerator.is_main_process):
        eval_label = f"eval_ep{current_episode}" if current_episode is not None else "eval"
        for ep_idx, traj in sorted(local_trajectories, key=lambda x: x[0]):
            log_episode_to_file(config, traj, group_num=eval_label, episode_idx=ep_idx, accelerator=accelerator)

    # Gather results from all GPUs
    if accelerator is not None:
        # Barrier ensures all ranks finished rollouts before gather
        try:
            accelerator.wait_for_everyone()
        except Exception as e:
            logger.warning(f"Barrier failed during eval gather: {e}")

        try:
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
        except Exception as e:
            logger.warning(f"Gather failed during eval: {e}, using local results only")
            rewards = local_rewards
            episode_times = local_episode_times
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
    from env.cleanup import CleanupEnvMove
    from .generation import generate_actions_multi_env_batch
    from .observation import obs_to_text, get_role_specific_observation, check_action_continuation
    from .rollout import _init_role_states

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

    # ── Role switching setup (same as run_parallel_episodes) ──
    role_interval = getattr(config, 'role_assignment_interval', 0)
    use_roles = (role_interval > 0 and config.action_mode == "compound")

    if use_roles:
        role_states = _init_role_states(1, config.num_agents)
        trainer._role_states = role_states
        if hasattr(trainer, '_perform_role_assignment'):
            trainer._perform_role_assignment([env], role_states)
    else:
        role_states = None
        trainer._role_states = None

    total_reward = 0
    output_lines = []
    step_times = []
    macro_infer_batch = getattr(config, 'macro_infer_batch', 24)

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

    if use_roles and role_states is not None:
        roles_str = ", ".join(
            f"Agent {aid}: {role}" for aid, role in sorted(role_states[0]['roles'].items())
        )
        log_and_save(f"[ROLE ASSIGNMENT] {roles_str}")
        log_and_save("")

    for step in range(config.max_env_steps):
        step_start_time = time.time()

        log_and_save(f"\n{'─' * 80}")
        log_and_save(f"STEP {step + 1}/{config.max_env_steps}")
        log_and_save(f"{'─' * 80}")

        actions = {}
        step_info = []

        # Use multi-env batch generation (handles roles properly)
        active_env_data = [(0, env, obs)]
        all_results = generate_actions_multi_env_batch(
            trainer, active_env_data, step, model, macro_infer_batch
        )
        batch_results = all_results[0]

        for agent_id in range(1, config.num_agents + 1):
            ax, ay = env.agents[agent_id]

            (action, log_prob, thinking_text, full_response, action_text,
             action_prompt, action_input_ids, action_ids) = batch_results[agent_id]
            actions[agent_id] = action

            # Build the obs text matching what was sent to the model
            role = None
            if use_roles and role_states is not None:
                role = role_states[0].get('roles', {}).get(agent_id)
            if role and config.action_mode == "compound":
                obs_nl = get_role_specific_observation(env, agent_id, role)
                tool_res = role_states[0].get('tool_results', {}).get(agent_id, '')
                if tool_res:
                    obs_nl += f"\n[TOOL RESULTS] {tool_res}"
                last_act = role_states[0].get('last_actions', {}).get(agent_id)
                cont_hint = check_action_continuation(env, agent_id, last_act)
                if cont_hint:
                    obs_nl += f"\n[CONTINUE] {cont_hint}"
            else:
                obs_nl = obs_to_text(obs[agent_id], env, agent_id, config)

            step_info.append({
                'agent_id': agent_id,
                'position': (ax, ay),
                'obs_nl': obs_nl,
                'role': role,
                'thinking': thinking_text.strip() if thinking_text else '',
                'response': full_response.strip() if full_response else '',
                'action_text': action_text.strip() if action_text else '',
                'action_prompt': action_prompt,
                'action': action,
                'log_prob': log_prob.item() if torch.is_tensor(log_prob) else float(log_prob),
            })

            # Track last action for continuation hints
            if use_roles and role_states is not None:
                from .rollout import _extract_action_json
                parsed = _extract_action_json(full_response)
                role_states[0]['last_actions'][agent_id] = parsed

        log_and_save("\nAgent Decisions:")
        for info in step_info:
            role_tag = f" [{info['role'].upper()}]" if info.get('role') else ""
            log_and_save(f"\n  Agent {info['agent_id']}{role_tag} at {info['position']}:")
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

        # ── Role reassignment after env.step ──
        if use_roles and role_interval > 0 and (step + 1) % role_interval == 0:
            if hasattr(trainer, '_perform_role_assignment'):
                old_roles = dict(role_states[0]['roles'])
                trainer._perform_role_assignment([env], role_states)
                new_roles = role_states[0]['roles']
                if new_roles != old_roles:
                    roles_str = ", ".join(
                        f"Agent {aid}: {role}" for aid, role in sorted(new_roles.items())
                    )
                    log_and_save(f"\n  >>> [ROLE SWITCH at step {step + 1}] {roles_str}")

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

    # Clean up role state
    trainer._role_states = None

    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"Visualization saved to: {save_to_file}")

    model.train()
