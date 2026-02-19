"""
grpo_config.py — Centralized configuration for GRPO training.

How to use:
  1. Edit the fields below to match your model, environment, and hardware.
  2. Import GRPOConfig into your training script:
       from grpo_config import GRPOConfig
  3. Pass it to your trainer:
       trainer = MyGRPOTrainer(GRPOConfig())
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRPOConfig:
    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-1.5B"
    # Max token length for prompt + response (prompt is left-truncated if needed)
    max_length: int = 1024

    # -------------------------------------------------------------------------
    # GENERATION (two-stage: thinking → action)
    # Stage 1: model produces free-form reasoning (thinking_tokens tokens).
    # Stage 2: model reads its reasoning and outputs a single action word.
    # -------------------------------------------------------------------------
    # Number of tokens the model is allowed to "think" before producing an action
    thinking_tokens: int = 384
    # Which tokens count toward the GRPO log-prob:
    #   "action"          → only the action token(s)
    #   "action+thinking" → both thinking and action tokens
    logprob_mode: str = "action+thinking"

    # Sampling parameters
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # -------------------------------------------------------------------------
    # LoRA
    # -------------------------------------------------------------------------
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # None → auto-detect linear layers (recommended). Set explicitly if needed,
    # e.g. ["q_proj", "k_proj", "v_proj", "o_proj"] for Qwen/LLaMA models.
    lora_target_modules: Optional[List[str]] = None

    # Quantization (mutually exclusive; requires bitsandbytes)
    use_8bit: bool = False
    use_4bit: bool = False

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    # Total number of environment episodes to run during training
    num_episodes: int = 1000

    # Single-GPU: how many episodes to collect per update group
    episodes_per_update: int = 8
    # Multi-GPU (accelerate): episodes collected on EACH GPU per group.
    # Total per group = episodes_per_gpu × num_gpus
    episodes_per_gpu: int = 4

    # Number of agents in the environment
    num_agents: int = 3

    learning_rate: float = 1e-5
    warmup_steps: int = 20
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = False

    # -------------------------------------------------------------------------
    # GRPO ALGORITHM
    # -------------------------------------------------------------------------
    # "grpo"   → normalize advantages by std (classic GRPO)
    # "drgrpo" → subtract mean only, no std division (DrGRPO)
    loss_type: str = "grpo"

    # Discount factor for episodic rewards (1.0 = no discounting)
    gamma: float = 0.99

    # PPO-style clipping for the importance-sampling ratio
    epsilon: float = 0.2

    # Whether to normalize advantages across the episode group
    advantage_normalization: bool = True
    # Hard-clip advantages to ±clip_advantage for stability
    clip_advantage: float = 5.0

    # Inner PPO-style epochs (re-use the same episode buffer multiple times)
    num_inner_epochs: int = 4
    # Trajectories sampled per gradient step inside the inner loop
    minibatch_size: int = 2
    # Samples packed into each GPU forward pass for gradient accumulation.
    # Lower = less VRAM; higher = faster.
    samples_per_micro_batch: int = 1

    # -------------------------------------------------------------------------
    # ENVIRONMENT
    # -------------------------------------------------------------------------
    # Maximum steps per episode before forced termination
    max_env_steps: int = 30
    # Seed for reproducibility (environment resets, model init, etc.)
    seed: int = 42

    # -------------------------------------------------------------------------
    # CHECKPOINTING & LOGGING
    # -------------------------------------------------------------------------
    output_dir: str = "./grpo_checkpoints"
    # Save checkpoint every N episodes
    save_steps: int = 50
    # Print a training summary every N groups
    log_interval: int = 5
    # Run evaluation every N episodes (0 = disabled)
    eval_interval: int = 100
    # Number of episodes used during each evaluation pass
    num_eval_episodes: int = 10

    # Log one sampled episode trajectory to a text file per group (useful for debugging)
    log_trajectory: bool = True
    trajectory_log_file: str = "episode_trajectories.txt"

    # -------------------------------------------------------------------------
    # EARLY STOPPING
    # -------------------------------------------------------------------------
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.5

    # -------------------------------------------------------------------------
    # MULTI-GPU (Accelerate)
    # -------------------------------------------------------------------------
    use_accelerate: bool = True
    mixed_precision: str = "bf16"  # "no", "fp16", or "bf16"

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # WANDB
    # -------------------------------------------------------------------------
    use_wandb: bool = False
    wandb_project: str = "grpo_training"
    wandb_entity: Optional[str] = None   # Your wandb username or team name
    wandb_run_name: Optional[str] = None  # Auto-generated if None
