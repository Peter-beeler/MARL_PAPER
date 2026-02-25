"""
Advantage computation, trajectory flattening, and GRPO loss computation.
"""

import random
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

from .logprob import compute_batch_sequence_log_prob

logger = logging.getLogger(__name__)


def compute_advantages(
    trajectories: List[Dict],
    config,
    accelerator=None,
    model=None
) -> List[Dict]:
    """
    Compute GRPO/DrGRPO advantages with optional multi-GPU global normalization.

    Args:
        trajectories: List of trajectory dicts, each with 'total_reward'.
        config: GRPOConfig instance.
        accelerator: Accelerate Accelerator (or None for single-GPU).
        model: Model (needed to find device for multi-GPU gather).

    Returns:
        Trajectories updated with 'advantage' and 'advantages' keys.
    """
    returns = [traj["total_reward"] for traj in trajectories]
    use_std_norm = (config.loss_type.lower() == "grpo")

    if config.advantage_normalization and len(returns) > 1:
        if accelerator is not None and model is not None:
            model_device = next(model.parameters()).device
            local_returns = torch.tensor(returns, dtype=torch.float32, device=model_device)
            all_returns = accelerator.gather(local_returns)
            global_mean = all_returns.mean().item()

            if use_std_norm:
                global_std = all_returns.std().item() + 1e-8
                normalized_returns = [(r - global_mean) / global_std for r in returns]
                if accelerator.is_main_process:
                    logger.info(f"  GRPO advantage stats: mean={global_mean:.4f}, std={global_std:.4f}, n={len(all_returns)}")
            else:
                normalized_returns = [(r - global_mean) for r in returns]
                if accelerator.is_main_process:
                    logger.info(f"  DrGrpo advantage stats: mean={global_mean:.4f}, n={len(all_returns)}")
        else:
            mean_return = np.mean(returns)
            if use_std_norm:
                std_return = np.std(returns) + 1e-8
                normalized_returns = [(r - mean_return) / std_return for r in returns]
            else:
                normalized_returns = [(r - mean_return) for r in returns]
    else:
        normalized_returns = returns

    normalized_returns = [
        float(np.clip(adv, -config.clip_advantage, config.clip_advantage))
        for adv in normalized_returns
    ]

    for traj, advantage in zip(trajectories, normalized_returns):
        traj["advantage"] = advantage
        traj["advantages"] = [advantage] * len(traj["rewards"])

    return trajectories


def create_minibatch_iterator(trajectories: List[Dict], minibatch_size: int):
    """Yield shuffled mini-batches of trajectories."""
    indices = list(range(len(trajectories)))
    random.shuffle(indices)
    for start_idx in range(0, len(indices), minibatch_size):
        end_idx = min(start_idx + minibatch_size, len(indices))
        yield [trajectories[i] for i in indices[start_idx:end_idx]]


def flatten_trajectories(trajectories: List[Dict]) -> List[Dict]:
    """
    Flatten trajectories into individual (prompt_ids, action_ids, advantage, old_log_prob) samples.

    Returns:
        List of sample dicts with keys:
            prompt_ids, action_ids, advantage, old_log_prob
    """
    all_samples = []
    for traj in trajectories:
        advantage = traj["advantage"]
        for i in range(len(traj["prompts"])):
            action_input_ids = traj["action_input_ids"][i]
            action_ids = traj["action_ids"][i]

            if action_input_ids is None or action_ids is None:
                continue
            if len(action_ids) == 0:
                continue

            all_samples.append({
                "prompt_ids": action_input_ids,
                "action_ids": action_ids,
                "advantage": advantage,
                "old_log_prob": traj["log_probs"][i]
            })

    return all_samples


def compute_loss_on_samples(
    model,
    samples: List[Dict],
    tokenizer,
    config,
    device
) -> Tuple[torch.Tensor, float, int]:
    """
    Compute GRPO loss on a micro-batch of flattened samples.

    Args:
        model: Current policy model (with grad).
        samples: List of sample dicts from flatten_trajectories.
        tokenizer: Tokenizer (for pad_token_id).
        config: GRPOConfig.
        device: Target device.

    Returns:
        (loss tensor, clip_fraction, n_samples)
    """
    if len(samples) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    prompt_ids_list = [s["prompt_ids"] for s in samples]
    action_ids_list = [s["action_ids"] for s in samples]
    advantages_list = [s["advantage"] for s in samples]
    old_log_probs_list = [s["old_log_prob"] for s in samples]

    new_log_probs = compute_batch_sequence_log_prob(
        model=model,
        prompt_input_ids_list=prompt_ids_list,
        generated_ids_list=action_ids_list,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        need_grad=True
    )

    old_log_probs = torch.tensor(old_log_probs_list, device=device, dtype=torch.float32)
    advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    epsilon = config.epsilon
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()

    with torch.no_grad():
        clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
        clip_fraction = clipped_mask.float().mean().item()

    return policy_loss, clip_fraction, len(samples)
