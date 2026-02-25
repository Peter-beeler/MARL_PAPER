"""
Merged GRPOConfig for grpo_textgame.py.
Supports both action_mode="text" and action_mode="compound".
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO training (text and compound modes)."""

    # --- NEW: mode selection ---
    action_mode: str = "text"        # "text" or "compound"
    action_tokens: int = 128         # max new tokens for action/JSON generation stage

    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_length: int = 512
    thinking_tokens: int = 256       # tokens for stage 1 reasoning
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # Text-mode specific
    use_two_stage: bool = True       # two-stage (thinking→action) vs single-stage
    logprob_mode: str = "action+thinking"  # "action" or "action+thinking"

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_8bit: bool = False
    use_4bit: bool = False

    # Training settings
    num_episodes: int = 500
    episodes_per_update: int = 8     # total per update (single-GPU)
    episodes_per_gpu: int = 4        # per GPU (multi-GPU)
    num_agents: int = 5
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = False

    # GRPO specific
    loss_type: str = "grpo"          # "grpo" or "drgrpo"
    gamma: float = 0.99
    epsilon: float = 0.2
    advantage_normalization: bool = True
    clip_advantage: float = 5.0

    # Inner epoch optimization (PPO-style)
    num_inner_epochs: int = 4
    minibatch_size: int = 2
    micro_batch_size: int = 8        # DEPRECATED - use samples_per_micro_batch
    samples_per_micro_batch: int = 2

    # Environment settings
    max_env_steps: int = 50
    eat_reward: float = 1.0
    clean_reward: float = 0.0

    # Checkpoint settings
    output_dir: str = "./grpo_textgame_checkpoints"
    save_steps: int = 50
    log_interval: int = 5
    eval_interval: int = 128
    num_eval_episodes: int = 20

    # Episode trajectory logging
    log_trajectory: bool = True
    trajectory_log_file: str = "episode_trajectories.txt"

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.5

    # Multi-GPU settings
    use_accelerate: bool = False
    use_deepspeed: bool = False      # enable DeepSpeed ZeRO-3 (requires use_accelerate=True)
    mixed_precision: str = "bf16"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "grpo_textgame"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
