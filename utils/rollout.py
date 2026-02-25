"""
Episode rollout, trajectory logging, and multi-GPU gathering.
"""

import os
import logging
import time
import torch
from typing import Dict, List, Optional

from .generation import generate_actions_batch
from .observation import obs_to_text

logger = logging.getLogger(__name__)


def run_episode(
    trainer,
    use_ref_model: bool = False,
    log_samples: bool = False,
    initial_state: Optional[Dict] = None
) -> Dict:
    """
    Run a single episode and collect trajectory (works for all action modes).

    Args:
        trainer: CleanupGameGRPO instance.
        use_ref_model: If True, use reference model instead of current policy.
        log_samples: If True, log a sample generation to console.
        initial_state: Optional initial state dict (for evaluation).

    Returns:
        Trajectory dict with keys:
            prompts, actions, responses, action_prompts, action_texts,
            rewards, log_probs, agent_ids, observations,
            action_input_ids, action_ids,
            total_reward, final_scores, steps, rollout_time
    """
    from env_move import CleanupEnvMove

    start_time = time.time()
    config = trainer.config

    env = CleanupEnvMove(trainer.env_config)
    if initial_state is not None:
        env.set_state(initial_state)
        obs = env._observation()
    else:
        obs = env.reset()

    trajectory = {
        "prompts": [],
        "actions": [],
        "responses": [],
        "action_prompts": [],
        "action_texts": [],
        "rewards": [],
        "log_probs": [],
        "agent_ids": [],
        "observations": [],
        "action_input_ids": [],
        "action_ids": [],
    }

    if use_ref_model and getattr(trainer, 'ref_model', None) is None:
        logger.warning("Reference model requested but not available. Using current policy.")
        model = trainer.model
    else:
        model = trainer.ref_model if use_ref_model else trainer.model

    total_reward = 0.0

    for step in range(config.max_env_steps):
        actions = {}
        batch_results = generate_actions_batch(trainer, obs, step, env, model)

        for agent_id in range(1, config.num_agents + 1):
            (action, log_prob, thinking_text, full_response, action_text,
             action_prompt, action_input_ids, action_ids) = batch_results[agent_id]

            actions[agent_id] = action

            trajectory["prompts"].append(action_prompt if action_prompt else
                                         _get_stored_prompt(trainer, obs, agent_id, step, env))
            trajectory["actions"].append(action)
            trajectory["responses"].append(full_response)
            trajectory["action_prompts"].append(action_prompt)
            trajectory["action_texts"].append(action_text)
            trajectory["log_probs"].append(log_prob.detach().item() if torch.is_tensor(log_prob) else float(log_prob))
            trajectory["agent_ids"].append(agent_id)
            trajectory["observations"].append(obs[agent_id])
            trajectory["action_input_ids"].append(action_input_ids)
            trajectory["action_ids"].append(action_ids)

            if log_samples and step == 0 and agent_id == 1:
                obs_text = obs_to_text(obs[agent_id], env, agent_id, config)
                if config.action_mode == "compound":
                    logger.info(f"\n  Sample generation (compound):")
                    logger.info(f"    Obs: {obs_text}")
                    logger.info(f"    Response (thinking+JSON): '{full_response[:200]}'")
                    logger.info(f"    → Low-level action: {action}")
                else:
                    logger.info(f"\n  Sample generation (text):")
                    logger.info(f"    Obs: {obs_text}")
                    if thinking_text:
                        logger.info(f"    Thinking: '{thinking_text[:100]}...'")
                    if action_text and action_text != full_response:
                        logger.info(f"    Action text (stage 2): '{action_text}'")
                    logger.info(f"    Action: {action}")

        obs, rewards, done, info = env.step(actions)

        for agent_id in range(1, config.num_agents + 1):
            trajectory["rewards"].append(rewards[agent_id])
            total_reward += rewards[agent_id]

        if done:
            break

    trajectory["total_reward"] = total_reward
    trajectory["final_scores"] = info["scores"]
    trajectory["steps"] = step + 1
    trajectory["rollout_time"] = time.time() - start_time

    if log_samples:
        from .model_setup import log_cuda_memory
        log_cuda_memory("After episode rollout")

    return trajectory


def _get_stored_prompt(trainer, obs, agent_id, step, env):
    """
    Build the prompt to store in trajectory['prompts'].
    For two-stage text mode: store thinking prompt.
    For all other modes: store the single-stage prompt.
    """
    from .observation import obs_to_text
    from .prompts import create_thinking_prompt, create_single_stage_prompt_text, create_single_stage_prompt_compound

    config = trainer.config
    tokenizer = trainer.tokenizer
    obs_text = obs_to_text(obs[agent_id], env, agent_id, config)

    if config.action_mode == "compound":
        return create_single_stage_prompt_compound(obs_text, config, tokenizer, env, agent_id)
    elif config.use_two_stage:
        return create_thinking_prompt(obs_text, agent_id, config, tokenizer)
    else:
        return create_single_stage_prompt_text(obs_text, config, tokenizer)


def log_episode_to_file(config, trajectory: Dict, group_num: int, episode_idx: int, accelerator=None):
    """
    Log a single episode trajectory to a text file for debugging/monitoring.

    Args:
        config: GRPOConfig.
        trajectory: Trajectory dict from run_episode.
        group_num: Current training group number.
        episode_idx: Index of this episode within the group.
        accelerator: Accelerate Accelerator (or None for single-GPU).
    """
    if not config.log_trajectory:
        return
    if accelerator is not None and not accelerator.is_main_process:
        return

    log_path = os.path.join(config.output_dir, config.trajectory_log_file)
    os.makedirs(config.output_dir, exist_ok=True)

    with open(log_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"GROUP {group_num} | EPISODE {episode_idx}\n")
        f.write(f"Total Reward: {trajectory['total_reward']:.2f} | Steps: {trajectory['steps']}\n")
        f.write(f"Final Scores: {trajectory.get('final_scores', 'N/A')}\n")
        f.write("=" * 80 + "\n\n")

        num_agents = config.num_agents
        num_steps = trajectory['steps']

        for step in range(num_steps):
            f.write(f"--- Step {step + 1} ---\n")

            for agent_idx in range(num_agents):
                idx = step * num_agents + agent_idx
                if idx >= len(trajectory['observations']):
                    break

                agent_id = trajectory['agent_ids'][idx]
                obs = trajectory['observations'][idx]
                action = trajectory['actions'][idx]
                reward = trajectory['rewards'][idx]
                response = trajectory['responses'][idx] if idx < len(trajectory['responses']) else "N/A"
                action_text = trajectory['action_texts'][idx] if idx < len(trajectory['action_texts']) else "N/A"

                f.write(f"\n  [Agent {agent_id}]\n")
                f.write(f"    Observation: {obs}\n")

                thinking_part = response[:300] + "..." if len(response) > 300 else response
                f.write(f"    Thinking/Response: {thinking_part}\n")

                if config.action_mode == "text" and config.use_two_stage:
                    f.write(f"    Action text (stage 2): {action_text[:200]}\n")
                elif config.action_mode == "compound":
                    f.write(f"    JSON action: {action_text[:200]}\n")

                f.write(f"    Action: {action}\n")
                f.write(f"    Reward: {reward:.2f}\n")

            f.write("\n")

        f.write(f"\n[Episode Summary]\n")
        f.write(f"  Total Reward: {trajectory['total_reward']:.2f}\n")
        f.write(f"  Rollout Time: {trajectory.get('rollout_time', 0):.2f}s\n\n")


def _gather_trajectories(local_trajectories: List[Dict], accelerator) -> List[Dict]:
    """Gather trajectories from all processes in multi-GPU setup."""
    if accelerator is None:
        return local_trajectories

    all_trajectories = accelerator.gather_for_metrics(local_trajectories)

    if accelerator.is_main_process:
        flattened = []
        for proc_trajs in all_trajectories:
            if isinstance(proc_trajs, list):
                flattened.extend(proc_trajs)
            else:
                flattened.append(proc_trajs)
        return flattened
    else:
        return []
