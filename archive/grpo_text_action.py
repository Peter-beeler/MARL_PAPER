"""
GRPO Fine-tuning with Plain Text Actions and Two-Stage Generation (Move Environment)

This version uses plain text action words (up, down, left, right, clean, eat, stay)
instead of special tokens, with a two-stage generation process:

Stage 1: Generate thinking/reasoning (N tokens)
Stage 2: Generate action based on thinking (M tokens, extract action word from output)

Key features:
- No special action tokens (<ST>, <UP>, etc.)
- Two-stage generation: thinking → action
- Action extracted from free-form generation (no logits processor)
- Uses chat template for prompts
- Coordinate-based observations
"""

import os, sys
import json
import torch
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from accelerate import Accelerator
import logging
import warnings
import wandb
from torch.nn.utils.rnn import pad_sequence
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# Import the move environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env_move import CleanupEnvMove, Config as EnvConfigMove

# Suppress the gradient checkpointing + KV cache warning
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def log_cuda_memory(stage: str, device: int = 0):
    """Log CUDA memory usage at a specific stage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        logger.info(f"[CUDA Memory - {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")


class AllowOnlyActionWords(LogitsProcessor):
    """Logits processor that restricts output to only action word tokens."""

    def __init__(self, tokenizer, action_words: List[str]):
        """
        Args:
            tokenizer: The tokenizer
            action_words: List of allowed action words (e.g., ['up', 'down', 'left', 'right'])
        """
        self.tokenizer = tokenizer
        self.action_words = action_words

        # Get token IDs for each action word
        # For each word, get all possible tokenizations (with/without space, case variations)
        self.allowed_token_ids = set()
        for word in action_words:
            # Try different variations
            for variant in [word, word.lower(), word.upper(), word.capitalize(),
                           f" {word}", f" {word.lower()}", f" {word.upper()}", f" {word.capitalize()}"]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                # Only use single token encodings to avoid multi-token words
                if len(tokens) == 1:
                    self.allowed_token_ids.add(tokens[0])

        # Also allow EOS token
        if tokenizer.eos_token_id is not None:
            self.allowed_token_ids.add(tokenizer.eos_token_id)

        self.allowed_tensor = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Mask out all tokens except allowed action tokens."""
        # Lazily create tensor on correct device
        if self.allowed_tensor is None or self.allowed_tensor.device != scores.device:
            self.allowed_tensor = torch.tensor(list(self.allowed_token_ids),
                                              device=scores.device, dtype=torch.long)

        # Create mask: set all to -inf, then unmask allowed tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed_tensor] = scores[:, self.allowed_tensor]

        return mask


@dataclass
class GRPOConfig:
    """Configuration for GRPO training with text actions."""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_length: int = 512
    thinking_tokens: int = 50  # Number of tokens for reasoning in stage 1
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    use_two_stage: bool = True  # Use two-stage generation (thinking + action)
    logprob_mode: str = "action+thinking"  # "action" (only action tokens) or "action+thinking" (both thinking and action tokens)

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    use_8bit: bool = False
    use_4bit: bool = False

    # Training settings
    num_episodes: int = 500
    episodes_per_update: int = 8  # Total episodes per update (will be split across GPUs if using multi-GPU)
    episodes_per_gpu: int = 4  # Episodes per GPU when using multi-GPU (episodes_per_update should be num_gpus * this)
    num_agents: int = 5
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = False

    # GRPO specific
    loss_type: str = "grpo"  # "grpo" or "drgrpo" (DrGrpo: no std normalization in advantages)
    gamma: float = 0.99
    epsilon: float = 0.2  # PPO-style clipping parameter for ratio
    advantage_normalization: bool = True
    clip_advantage: float = 5.0

    # Inner epoch optimization (PPO-style)
    num_inner_epochs: int = 4  # Number of optimization epochs per group (buffer reuse)
    minibatch_size: int = 2  # Number of trajectories to sample per training step

    # DEPRECATED: micro_batch_size is replaced by minibatch_size
    micro_batch_size: int = 8  # DEPRECATED - will be removed in future version

    # Gradient accumulation settings (for memory management)
    samples_per_micro_batch: int = 2  # Number of (prompt, action) samples per micro-batch for gradient accumulation
                                       # Set to 1-2 for A100, adjust based on GPU memory

    # Environment settings
    max_env_steps: int = 50
    eat_reward: float = 1.0    # Reward for eating an apple (passed to env)
    clean_reward: float = 0.0  # Reward for cleaning a dirt tile (passed to env)

    # Checkpoint settings
    output_dir: str = "./grpo_text_action_checkpoints"
    save_steps: int = 50
    log_interval: int = 5
    eval_interval: int = 128  # Evaluate every N training episodes (0 = no mid-training eval)
    num_eval_episodes: int = 20  # Number of episodes to run during evaluation

    # Episode trajectory logging (for debugging/monitoring)
    log_trajectory: bool = True  # Log one random episode per update to file
    trajectory_log_file: str = "episode_trajectories.txt"  # Log file name (in output_dir)

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.5

    # Multi-GPU settings
    use_accelerate: bool = False
    mixed_precision: str = "bf16"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "grpo_text_action"
    wandb_entity: str = None  # Set to your wandb username/team
    wandb_run_name: str = None  # Auto-generated if None


class CleanupGameGRPOText:
    """GRPO trainer for cleanup game using plain text actions."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)

        # Initialize Accelerator for multi-GPU training
        if config.use_accelerate and not (config.use_8bit or config.use_4bit):
            # 1. Define the timeout (e.g., 60 minutes)
            timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

            # 2. Pass it to kwargs_handlers
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
                log_with=None,
                kwargs_handlers=[timeout_kwargs] # <--- Added this line
            )
            self.device = self.accelerator.device
            logger.info(f"Using Accelerator with {self.accelerator.num_processes} processes")
        else:
            self.accelerator = None
            self.device = config.device
            if config.use_8bit or config.use_4bit:
                logger.info(f"Using quantization (8bit={config.use_8bit}, 4bit={config.use_4bit})")

        # Only log on main process
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Loading model: {config.model_name}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set left padding for decoder-only models (required for batch generation)
        self.tokenizer.padding_side = 'left'

        # Define action words (no special tokens needed)
        self.action_words = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']

        # Create logits processor for action-only generation
        self.action_logits_processor = AllowOnlyActionWords(self.tokenizer, self.action_words)

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Action words: {self.action_words}")
            logger.info(f"Two-stage generation: {config.use_two_stage}")
            if config.use_two_stage:
                logger.info(f"Thinking tokens: {config.thinking_tokens}")
                logger.info(f"Log probability mode: {config.logprob_mode}")

        # Load model with quantization if specified
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if config.use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif config.use_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )

        # Move to device if not using quantization or accelerate
        if not (config.use_8bit or config.use_4bit) and (self.accelerator is None):
            base_model = base_model.to(self.device)

        # Apply gradient checkpointing
        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        # Prepare model for training
        if config.use_8bit or config.use_4bit:
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model)

        # Apply LoRA
        if config.use_lora:
            # Auto-detect target modules if not specified
            if config.lora_target_modules is None:
                linear_layers = set()
                for name, module in base_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        layer_name = name.split('.')[-1]
                        linear_layers.add(layer_name)

                exclude_names = {'lm_head', 'embed_tokens', 'wte', 'wpe', 'ln', 'norm'}
                target_modules = [
                    name for name in linear_layers
                    if not any(ex in name.lower() for ex in exclude_names)
                ]

                if not target_modules:
                    if "qwen" in config.model_name.lower():
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    else:
                        target_modules = ["q_proj", "v_proj"]

                if not target_modules:
                    target_modules = list(linear_layers)[:4]
            else:
                target_modules = config.lora_target_modules

            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info(f"Applying LoRA to modules: {target_modules}")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            self.model = get_peft_model(base_model, peft_config)

            if self.accelerator is None or self.accelerator.is_main_process:
                self.model.print_trainable_parameters()
                log_cuda_memory("After LoRA model loaded")
        else:
            self.model = base_model
            if self.accelerator is None or self.accelerator.is_main_process:
                log_cuda_memory("After base model loaded")

        self.model.train()

        # Old model for PPO-style ratio computation (θ_old)
        # This will be a deep copy of the current model, updated at the start of each group
        # During inner epochs, old model stays frozen while current model is updated
        self.old_model = None  # Will be created on first update_old_model() call
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info("GRPO with inner epochs: Old model will be created on first group")

        # Reference model - NOT USED (keeping old code commented for reference)
        self.ref_model = None  # Set to None for standard GRPO
        self.ref_on_cpu = False
        self.ref_device = None

        # Note: If you want to use the reference model for visualization or comparison,
        # uncomment the code below (requires additional GPU memory)
        """
        # Reference model (frozen) - kept on CPU to save GPU memory
        ref_model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if config.use_8bit or config.use_4bit:
            # For quantized models, still need to use GPU
            ref_model_kwargs = model_kwargs.copy()
            self.ref_on_cpu = False
            self.ref_on_secondary_gpu = False
        else:
            # Multi-GPU distributed: each process keeps ref model on its own GPU
            # Single GPU or no accelerator: use CPU to save memory
            if self.accelerator is not None:
                self.ref_on_cpu = False
                self.ref_on_secondary_gpu = False  # Each GPU has its own copy
                if self.accelerator.is_main_process:
                    logger.info(f"Multi-GPU distributed setup ({self.accelerator.num_processes} processes)")
            else:
                # Single GPU: use CPU to save memory
                self.ref_on_cpu = True
                self.ref_on_secondary_gpu = False

        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **ref_model_kwargs
        )

        if self.ref_on_cpu:
            # Keep reference model on CPU
            ref_model = ref_model.to('cpu')
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Reference model placed on CPU to save GPU memory")
                log_cuda_memory("After ref model to CPU")
        elif self.accelerator is not None:
            # Multi-GPU: each process places ref model on its assigned GPU
            ref_model = ref_model.to(self.device)
            logger.info(f"[GPU{self.accelerator.process_index}] Reference model placed on {self.device}")
            log_cuda_memory(f"After ref model to {self.device}")
        elif not (config.use_8bit or config.use_4bit) and (self.accelerator is None):
            ref_model = ref_model.to(self.device)
            log_cuda_memory("After ref model to main GPU")

        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        """

        # Optimizer
        if config.use_lora:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        else:
            trainable_params = self.model.parameters()

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )

        # Learning rate scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return 0.95 ** ((step - self.config.warmup_steps) / 10)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )

        # Prepare for distributed training
        if self.accelerator is not None and not (config.use_8bit or config.use_4bit):
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            logger.info(f"[GPU{self.accelerator.process_index}] Model prepared")
        elif config.use_8bit or config.use_4bit:
            self.device = next(self.model.parameters()).device

        # Environment
        self.env_config = EnvConfigMove(
            n_agents=config.num_agents,
            max_steps=config.max_env_steps,
            seed=config.seed,
            eat_reward=config.eat_reward,
            clean_reward=config.clean_reward,
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.training_step = 0

        # Initialize wandb
        if config.use_wandb and (self.accelerator is None or self.accelerator.is_main_process):
            wandb_config = {
                "model_name": config.model_name,
                "num_agents": config.num_agents,
                "num_episodes": config.num_episodes,
                "episodes_per_update": config.episodes_per_update,
                "learning_rate": config.learning_rate,
                "loss_type": config.loss_type,
                "gamma": config.gamma,
                "epsilon": config.epsilon,
                "max_grad_norm": config.max_grad_norm,
                "temperature": config.temperature,
                "thinking_tokens": config.thinking_tokens,
                "use_two_stage": config.use_two_stage,
                "logprob_mode": config.logprob_mode,
                "use_lora": config.use_lora,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "max_env_steps": config.max_env_steps,
                "seed": config.seed,
            }

            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=wandb_config,
                reinit=True
            )
            logger.info(f"Initialized wandb project: {config.wandb_project}")

        # Pre-generate fixed evaluation states
        self.eval_states = []
        self._generate_eval_states(num_states=20)

    def update_old_model(self):
        """Copy current model weights to old model (θ_old ← θ).

        Called at the beginning of each training group. The old model is used to compute
        log-probs during rollout and stays frozen during all inner optimization epochs.
        """
        import copy

        if self.old_model is None:
            # First time: create old model as deep copy
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Creating old model (first group)...")

            # Deep copy the model
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()

            # Freeze all parameters
            for param in self.old_model.parameters():
                param.requires_grad = False

            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("  ✓ Old model created and frozen")
        else:
            # Subsequent times: just copy weights (faster than deep copy)
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Updating old model with current weights...")

            self.old_model.load_state_dict(self.model.state_dict())
            self.old_model.eval()

            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("  ✓ Old model updated")

    def _generate_eval_states(self, num_states: int = 20):
        """Pre-generate fixed initial states for evaluation."""
        for i in range(num_states):
            eval_env = CleanupEnvMove(EnvConfigMove(
                n_agents=self.config.num_agents,
                max_steps=self.config.max_env_steps,
                seed=self.config.seed + 1000 + i,
                eat_reward=self.config.eat_reward,
                clean_reward=self.config.clean_reward,
            ))
            eval_env.reset()
            state = eval_env.get_state()
            self.eval_states.append(state)

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Generated {num_states} fixed evaluation states")

    def _parse_observation_to_coords(self, obs: str, agent_id: int, env) -> str:
        """Convert local observation to coordinate-based format.

        Transforms internal coordinates (y=0 at top) to display coordinates (y=0 at bottom).
        """
        # Get agent position
        agent_pos = env.agents[agent_id]
        ax, ay_internal = agent_pos

        # Transform y-coordinate: y=0 at bottom instead of top
        ay_display = (env.height - 1) - ay_internal

        # Get local window bounds (in internal coordinates)
        half_w, half_h = 2, 1
        y0 = max(0, ay_internal - half_h)
        y1 = min(env.height - 1, ay_internal + half_h)
        x0 = max(0, ax - half_w)
        x1 = min(env.width - 1, ax + half_w)

        # Find dirt and apples in local window
        dirt_coords = []
        apple_coords = []

        for y_internal in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if env.items[y_internal][x] == '#':
                    # Transform to display coordinates
                    y_display = (env.height - 1) - y_internal
                    dirt_coords.append((x, y_display))
                elif env.items[y_internal][x] == 'a':
                    # Transform to display coordinates
                    y_display = (env.height - 1) - y_internal
                    apple_coords.append((x, y_display))

        # Find other agents in local window and their movement
        other_agents_info = []
        for other_id, (ox, oy_internal) in env.agents.items():
            if other_id == agent_id:
                continue  # Skip self
            # Check if other agent is in local window
            if x0 <= ox <= x1 and y0 <= oy_internal <= y1:
                # Transform to display coordinates
                oy_display = (env.height - 1) - oy_internal
                # Get movement info
                if hasattr(env, 'get_agent_movement'):
                    direction, (prev_x, prev_y) = env.get_agent_movement(other_id)
                    if direction == "STAYED":
                        other_agents_info.append(f"Agent {other_id} at ({ox},{oy_display}) [stayed]")
                    else:
                        other_agents_info.append(f"Agent {other_id} at ({ox},{oy_display}) [moved {direction} from ({prev_x},{prev_y})]")
                else:
                    other_agents_info.append(f"Agent {other_id} at ({ox},{oy_display})")

        # Format observation text with display coordinates
        obs_text = f"You at ({ax},{ay_display})."
        if dirt_coords:
            obs_text += f" Dirt at {', '.join([f'({x},{y})' for x, y in dirt_coords])}."
        if apple_coords:
            obs_text += f" Apple at {', '.join([f'({x},{y})' for x, y in apple_coords])}."
        if other_agents_info:
            obs_text += " " + " ".join(other_agents_info) + "."

        # Add message if nothing is visible
        if not dirt_coords and not apple_coords and not other_agents_info:
            obs_text += " Nothing in your view."

        return obs_text

    def _clean_reward_desc(self) -> str:
        """Return the reward description for cleaning, based on config."""
        if self.config.clean_reward > 0.0:
            return f"- Cleaning dirt gives +{self.config.clean_reward} reward AND enables apple spawning (less dirt = more apples). "
        return "- Cleaning dirt itself gives NO points, but is necessary to enable apple spawning. "

    def create_thinking_prompt(self, obs: str, agent_id: int, step: int, env=None) -> str:
        """Create a prompt for stage 1: thinking/reasoning."""
        # Convert observation to coordinate format
        if env is not None:
            obs_text = self._parse_observation_to_coords(obs, agent_id, env)
        else:
            obs_text = obs

        system_msg = (
            f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{self.config.eat_reward} each). "
            "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
            + self._clean_reward_desc() +
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

        user_msg = obs_text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # Apply chat template
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt_str

    def _create_thinking_prompt_from_text(self, obs_text: str, agent_id: int) -> str:
        """Create thinking prompt from pre-formatted observation text.

        Args:
            obs_text: Pre-formatted observation text
            agent_id: Agent ID

        Returns:
            Chat template formatted prompt string
        """
        system_msg = (
            f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{self.config.eat_reward} each). "
            "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
            + self._clean_reward_desc() +
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

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt_str

    def create_single_stage_prompt(self, obs: str, agent_id: int, step: int, env=None) -> str:
        """Create a prompt for single-stage generation: direct action output only.

        Args:
            obs: Observation (will be converted to coordinate format)
            agent_id: Agent ID
            step: Current step
            env: Environment (optional, for coordinate conversion)
        """
        # Convert observation to coordinate format
        if env is not None:
            obs_text = self._parse_observation_to_coords(obs, agent_id, env)
        else:
            obs_text = obs

        system_msg = (
            f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{self.config.eat_reward} each). "
            "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
            + self._clean_reward_desc() +
            "- You can only eat/clean items at your position. "
            "Available actions: "
            "up = move one step up (increase y) "
            "down = move one step down (decrease y) "
            "left = move one step left (decrease x) "
            "right = move one step right (increase x) "
            "clean = clean dirt if on your cell "
            "eat = eat apple if on your cell "
            "stay = stay in current position. "
            "Output ONLY ONE action word: up, down, left, right, clean, eat, or stay."
        )

        user_msg = obs_text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt_str

    def _create_single_stage_prompt_from_text(self, obs_text: str, agent_id: int) -> str:
        """Create single-stage prompt from pre-formatted observation text.

        Args:
            obs_text: Pre-formatted observation text
            agent_id: Agent ID

        Returns:
            Chat template formatted prompt string
        """
        clean_line = (
            f"- Cleaning dirt: +{self.config.clean_reward} reward for you AND enables apple spawning\n"
            if self.config.clean_reward > 0.0
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
            f"- Eating apple: +{self.config.eat_reward} reward for you\n"
            "- Dirt respawns if river isn't clean enough\n\n"
            "Output ONLY ONE action word: up, down, left, right, clean, eat, or stay."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": obs_text}
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt_str

    def create_action_prompt(self, obs_text: str, thinking_text: str) -> str:
        """Create a prompt for stage 2: action selection based on thinking.

        Args:
            obs_text: The observation text (coordinate format)
            thinking_text: The generated thinking/reasoning text
        """
        system_msg = (
            f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{self.config.eat_reward} each). "
            "Rules: - Apples only spawn on land when the river is clean (less dirt = more apples). "
            + self._clean_reward_desc() +
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

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt_str

    def get_action_from_response(self, response: str) -> str:
        """Extract action word from the end of the response."""
        response = response.strip().lower()

        # Check from the end of the response for action words
        words = response.split()

        # Check last few words for an action
        for i in range(min(5, len(words))):
            word = words[-(i+1)].strip('.,!?;:')
            if word in self.action_words:
                return word

        # If no action found, default to stay
        logger.debug(f"No valid action found in response: '{response}', defaulting to 'stay'")
        return "stay"

    # def compute_sequence_log_prob(self, model, prompt_input_ids: torch.Tensor, generated_ids: torch.Tensor, device: torch.device, pad_token_id: int = None, need_grad = False) -> torch.Tensor:
    #     # Ensure tensors are on the correct device
    #     prompt_ids = prompt_input_ids.to(device)
    #     gen_ids = generated_ids.to(device)

    #     # Flatten to 1D if needed
    #     if prompt_ids.dim() == 2:
    #         prompt_ids = prompt_ids[0]
    #     if gen_ids.dim() == 2:
    #         gen_ids = gen_ids[0]

    #     # --- FIX START: Remove Padding Tokens ---
    #     # If pad_token_id is not provided, try to guess from model config, or assume 0/2 if common
    #     if pad_token_id is None:
    #         # Handle DDP wrapper
    #         actual_model = model.module if hasattr(model, 'module') else model
    #         if hasattr(actual_model.config, 'pad_token_id') and actual_model.config.pad_token_id is not None:
    #             pad_token_id = actual_model.config.pad_token_id

    #     if pad_token_id is not None:
    #         prompt_ids = prompt_ids[prompt_ids != pad_token_id]
    #         # We usually don't need to filter gen_ids, but good safety measure
    #         gen_ids = gen_ids[gen_ids != pad_token_id]
    #     # --- FIX END ---

    #     if gen_ids.size(0) == 0:
    #         return torch.tensor(-10.0, device=device)

    #     # Build full sequence
    #     full_ids = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)  # (1, total_len)

    #     # Create attention mask (all 1s since we removed padding)
    #     attention_mask = torch.ones_like(full_ids)
        
    #     if need_grad:
    #         outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    #         logits = outputs.logits
    #     else:
    #         with torch.no_grad():
    #             # Pass the attention mask explicitly
    #             outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    #             logits = outputs.logits
    #     prompt_len = prompt_ids.size(0)
    #     gen_len = gen_ids.size(0)

    #     # Slicing logic remains the same
    #     if prompt_len > 0:
    #         predicted_logits = logits[:, prompt_len-1:prompt_len-1+gen_len, :]
    #     else:
    #         predicted_logits = logits[:, :gen_len, :]

    #     log_probs = torch.nn.functional.log_softmax(predicted_logits, dim=-1)

    #     gen_ids_batched = gen_ids.unsqueeze(0).unsqueeze(-1)
    #     token_log_probs = torch.gather(log_probs, dim=-1, index=gen_ids_batched).squeeze(-1)

    #     seq_log_prob = token_log_probs.sum(dim=1)
    #     avg_log_prob = seq_log_prob / float(gen_len)

    #     return avg_log_prob.squeeze()
    
    def compute_batch_sequence_log_prob(
        self,
        model,
        prompt_input_ids_list: List[torch.Tensor],
        generated_ids_list: List[torch.Tensor],
        device: torch.device,
        pad_token_id: int = None,
        need_grad: bool = False
    ) -> torch.Tensor:
        
        # --- 1. CONFIGURATION & CLEANUP ---
        if pad_token_id is None:
            actual_model = model.module if hasattr(model, 'module') else model
            if hasattr(actual_model.config, 'pad_token_id'):
                pad_token_id = actual_model.config.pad_token_id
        
        # Fallback if still None (though usually config has it)
        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided or available in model config")

        # --- 2. PREPARE BATCH (CPU side usually fast enough) ---
        full_sequences = []
        prompt_lens = []
        gen_lens = []

        for prompt_ids, gen_ids in zip(prompt_input_ids_list, generated_ids_list):
            # Clean inputs (remove existing padding if any)
            p_ids = prompt_ids.to(device).view(-1)
            p_ids = p_ids[p_ids != pad_token_id]
            
            g_ids = gen_ids.to(device).view(-1)
            g_ids = g_ids[g_ids != pad_token_id]
            
            # Store lengths for masking later
            prompt_lens.append(len(p_ids))
            gen_lens.append(len(g_ids))
            
            # Concatenate: [Prompt, Gen]
            full_sequences.append(torch.cat([p_ids, g_ids]))

        # Create padded batch: Shape [Batch_Size, Max_Seq_Len]
        # pad_sequence defaults to batch_first=False, so we specify True
        batch_input_ids = pad_sequence(full_sequences, batch_first=True, padding_value=pad_token_id)
        
        # Create attention mask (1 for real tokens, 0 for pad)
        attention_mask = (batch_input_ids != pad_token_id).long()

        # --- 3. FORWARD PASS ---
        def forward_pass():
            outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
            return outputs.logits

        if need_grad:
            logits = forward_pass()
        else:
            with torch.no_grad():
                logits = forward_pass()

        # --- 4. CALCULATE LOG PROBS ---
        # Shift so that tokens at [i] predict [i+1]
        # logits shape: [B, Seq_Len, Vocab] -> Slice off last token
        shift_logits = logits[..., :-1, :].contiguous()
        # labels shape: [B, Seq_Len]        -> Slice off first token
        shift_labels = batch_input_ids[..., 1:].contiguous()

        # Calculate CrossEntropy per token (reduction='none')
        # This effectively calculates log_softmax and gathers in one optimized step
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        
        # Shape: [Batch_Size, Seq_Len - 1]
        # We use negative because CrossEntropy is -log(p), we want log(p)
        token_log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.size())

        # --- 5. MASKING (CRITICAL STEP) ---
        # We only want sums for the GENERATED part, not the PROMPT part.
        
        # Create a mask of the same shape as token_log_probs
        # Initialize with 0.0
        action_mask = torch.zeros_like(token_log_probs)

        for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
            # logic: 
            # The first token generated is at index (p_len - 1) in the shifted coordinates
            # It ends at (p_len - 1 + g_len)
            start_idx = max(0, p_len - 1) 
            end_idx = start_idx + g_len
            
            # Ensure we don't go out of bounds (in case of empty gen)
            if g_len > 0:
                action_mask[i, start_idx:end_idx] = 1.0

        # Apply mask
        masked_log_probs = token_log_probs * action_mask

        # Sum per sequence
        seq_log_prob_sum = masked_log_probs.sum(dim=1)
        
        # Optional: If you strictly need Average per token like your original function:
        # tensor_gen_lens = torch.tensor(gen_lens, device=device)
        # return seq_log_prob_sum / tensor_gen_lens

        return seq_log_prob_sum # Returns tensor of shape [Batch_Size]
    def generate_action(self, obs: str, agent_id: int, step: int, env, model, return_prompts: bool = False) -> Tuple[str, torch.Tensor, str, str, str, Optional[Dict[str, str]]]:
        """Generate action from model using two-stage generation.

        Args:
            obs: Observation string
            agent_id: Agent ID
            step: Current step
            env: Environment
            model: Model to use for generation
            return_prompts: If True, return dict with stage 1 and 2 prompts

        Returns:
            action: Action string (e.g., "stay", "up")
            log_prob: Log probability of the action token
            thinking_text: The generated thinking/reasoning text
            full_response: Combined thinking + action
            action_raw: Raw action text from stage 2 (for debugging)
            prompts: Optional dict with 'stage1' and 'stage2' prompts (if return_prompts=True)
        """
        # Determine which device to use
        is_ref_model = (model is self.ref_model)
        if is_ref_model and hasattr(self, 'ref_device'):
            target_device = self.ref_device
        else:
            target_device = self.device

        # Unwrap model if using accelerate
        if self.accelerator is not None and hasattr(model, 'module'):
            gen_model = model.module
        else:
            gen_model = model

        if self.config.use_two_stage:
            # Get observation text in coordinate format
            if env is not None:
                obs_text = self._parse_observation_to_coords(obs, agent_id, env)
            else:
                obs_text = obs

            # STAGE 1: Generate thinking/reasoning
            thinking_prompt = self.create_thinking_prompt(obs, agent_id, step, env)
            thinking_inputs = self.tokenizer(
                thinking_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=False
            ).to(target_device)

            if "attention_mask" not in thinking_inputs:
                thinking_inputs["attention_mask"] = torch.ones_like(thinking_inputs["input_ids"])

            try:
                # Generate thinking tokens
                with torch.no_grad():
                    thinking_outputs = gen_model.generate(
                        **thinking_inputs,
                        max_new_tokens=self.config.thinking_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
             
                    )
            except RuntimeError as e:
                logger.warning(f"Thinking generation failed: {e}, using default action")
                prompts_dict = None if not return_prompts else {'stage1': thinking_prompt, 'stage2': ''}
                return "stay", torch.tensor(0.0, device=target_device), "", "", "", prompts_dict, "", None, None

            # Extract thinking text
            thinking_ids = thinking_outputs.sequences[0][thinking_inputs.input_ids.shape[1]:]
            thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True)

            # STAGE 2: Generate action based on thinking
            action_prompt = self.create_action_prompt(obs_text, thinking_text)
            action_inputs = self.tokenizer(
                action_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=False
            ).to(target_device)

            if "attention_mask" not in action_inputs:
                action_inputs["attention_mask"] = torch.ones_like(action_inputs["input_ids"])

            try:
                # Generate action tokens (no logits processor)
                with torch.no_grad():
                    action_outputs = gen_model.generate(
                        **action_inputs,
                        max_new_tokens=10,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                
                    )
            except RuntimeError as e:
                logger.warning(f"Action generation failed: {e}, using default action")
                prompts_dict = None if not return_prompts else {'stage1': thinking_prompt, 'stage2': action_prompt}
                return "stay", torch.tensor(0.0, device=target_device), thinking_text, thinking_text, "", prompts_dict, action_prompt, None, None

            # Extract action tokens
            action_ids = action_outputs.sequences[0][action_inputs.input_ids.shape[1]:]
            action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
            action = self.get_action_from_response(action_text)

            # Compute log probability using OLD model (for PPO-style ratio computation)
            # This log-prob stays fixed during inner epochs
            try:
                # Use old model if available (after first group), otherwise use current model
                log_prob_model = self.old_model if self.old_model is not None else gen_model

                if self.config.logprob_mode == "action+thinking":
                    # Compute log_prob for both thinking and action tokens
                    # Stage 1: thinking log_prob
                    thinking_log_probs = self.compute_batch_sequence_log_prob(
                        model=log_prob_model,
                        prompt_input_ids_list=[thinking_inputs.input_ids],
                        generated_ids_list=[thinking_ids],
                        device=target_device,
                        pad_token_id=self.tokenizer.pad_token_id,
                        need_grad=False
                    )
                    thinking_log_prob = thinking_log_probs[0]

                    # Stage 2: action log_prob
                    action_log_probs = self.compute_batch_sequence_log_prob(
                        model=log_prob_model,
                        prompt_input_ids_list=[action_inputs.input_ids],
                        generated_ids_list=[action_ids],
                        device=target_device,
                        pad_token_id=self.tokenizer.pad_token_id,
                        need_grad=False
                    )
                    action_log_prob = action_log_probs[0]

                    # Combine both log probabilities
                    log_prob = thinking_log_prob + action_log_prob
                else:
                    # Default: only compute log_prob for action tokens
                    log_probs = self.compute_batch_sequence_log_prob(
                        model=log_prob_model,
                        prompt_input_ids_list=[action_inputs.input_ids],
                        generated_ids_list=[action_ids],
                        device=target_device,
                        pad_token_id=self.tokenizer.pad_token_id,
                        need_grad=False  # No gradients needed for old model
                    )
                    log_prob = log_probs[0]  # Extract single result

                # Debug logging for first few calls
                if not hasattr(self, '_debug_log_count'):
                    self._debug_log_count = 0
                if self._debug_log_count < 3:
                    logger.info(f"[DEBUG] Two-stage generation:")
                    logger.info(f"[DEBUG]   Log prob mode: {self.config.logprob_mode}")
                    logger.info(f"[DEBUG]   Action text: '{action_text}', action: '{action}'")
                    logger.info(f"[DEBUG]   Generated action tokens: {len(action_ids)}")
                    if self.config.logprob_mode == "action+thinking":
                        logger.info(f"[DEBUG]   Generated thinking tokens: {len(thinking_ids)}")
                        logger.info(f"[DEBUG]   Thinking log prob: {thinking_log_prob.item():.6f}")
                        logger.info(f"[DEBUG]   Action log prob: {action_log_prob.item():.6f}")
                        logger.info(f"[DEBUG]   Total log prob: {log_prob.item():.6f}")
                    else:
                        logger.info(f"[DEBUG]   Action log prob: {log_prob.item():.6f}")
                    self._debug_log_count += 1
            except Exception as e:
                logger.warning(f"Log prob calculation failed: {e}")
                import traceback
                traceback.print_exc()
                log_prob = torch.tensor(-10.0, device=target_device)

            full_response = f"{thinking_text} -> {action_text}"
            prompts_dict = None if not return_prompts else {'stage1': thinking_prompt, 'stage2': action_prompt}
            # Return: action, log_prob, thinking_text, full_response, action_text, prompts_dict, action_prompt
            return action, log_prob, thinking_text, full_response, action_text, prompts_dict, action_prompt, action_inputs.input_ids, action_ids


        else:
            # Single-stage generation: thinking + action in one response
            single_stage_prompt = self.create_single_stage_prompt(obs, agent_id, step, env)
            inputs = self.tokenizer(
                single_stage_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=False
            ).to(target_device)

            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            try:
                with torch.no_grad():
                    outputs = gen_model.generate(
                        **inputs,
                        max_new_tokens=10,  # Only need 1-2 tokens for action word
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            except RuntimeError as e:
                logger.warning(f"Generation failed: {e}, using default action")
                prompts_dict = None if not return_prompts else {'stage1': single_stage_prompt, 'stage2': ''}
                return "stay", torch.tensor(0.0, device=target_device), "", "", "", prompts_dict, "", None, None

            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            action = self.get_action_from_response(response)

            # Compute log probability using OLD model (for PPO-style ratio computation)
            try:
                # Use old model if available (after first group), otherwise use current model
                log_prob_model = self.old_model if self.old_model is not None else gen_model

                log_probs = self.compute_batch_sequence_log_prob(
                    model=log_prob_model,
                    prompt_input_ids_list=[inputs.input_ids],
                    generated_ids_list=[generated_ids],
                    device=target_device,
                    pad_token_id=self.tokenizer.pad_token_id,
                    need_grad=False  # No gradients needed for old model
                )
                log_prob = log_probs[0]  # Extract single result

                # Debug logging for first few calls
                if not hasattr(self, '_debug_log_count_single'):
                    self._debug_log_count_single = 0
                if self._debug_log_count_single < 3:
                    logger.info(f"[DEBUG] Single-stage generation:")
                    logger.info(f"[DEBUG]   Response: '{response}', action: '{action}'")
                    logger.info(f"[DEBUG]   Generated tokens: {len(generated_ids)}")
                    logger.info(f"[DEBUG]   Avg log prob per token: {log_prob.item():.6f}")
                    self._debug_log_count_single += 1
            except Exception as e:
                logger.warning(f"Log prob calculation failed: {e}")
                import traceback
                traceback.print_exc()
                log_prob = torch.tensor(-10.0, device=target_device)

            prompts_dict = None if not return_prompts else {'stage1': single_stage_prompt, 'stage2': ''}
            # For single-stage, action_prompt is empty and action_text is the full response
            return action, log_prob, response, response, response, prompts_dict, "", inputs.input_ids, generated_ids

    def generate_actions_batch(self, obs_dict: Dict[int, str], step: int, env, model, use_ref_model: bool = False):
        """Generate actions for all agents in a batch.

        Args:
            obs_dict: Dictionary mapping agent_id -> observation string
            step: Current step
            env: Environment
            model: Model to use for generation
            use_ref_model: Whether using reference model

        Returns:
            Dictionary with keys as agent_ids and values as tuples of:
            (action, log_prob, thinking_text, full_response, action_text, action_prompt, action_input_ids, action_ids)
        """
        # Determine device
        if use_ref_model and hasattr(self, 'ref_device'):
            target_device = self.ref_device
        else:
            target_device = self.device

        # Unwrap model if using accelerate
        if self.accelerator is not None and hasattr(model, 'module'):
            gen_model = model.module
        else:
            gen_model = model

        agent_ids = sorted(obs_dict.keys())
        num_agents = len(agent_ids)

        if self.config.use_two_stage:
            # STAGE 1: Batch generate thinking for all agents

            thinking_prompts = []
            obs_texts = []
            for agent_id in agent_ids:
                obs = obs_dict[agent_id]
                if env is not None:
                    obs_text = self._parse_observation_to_coords(obs, agent_id, env)
                    thinking_prompt = self._create_thinking_prompt_from_text(obs_text, agent_id)
                else:
                    obs_text = obs
                    thinking_prompt = self.create_thinking_prompt(obs, agent_id, step, env)

                obs_texts.append(obs_text)
                thinking_prompts.append(thinking_prompt)

            # Tokenize all thinking prompts
            thinking_inputs = self.tokenizer(
                thinking_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(target_device)

            if "attention_mask" not in thinking_inputs:
                thinking_inputs["attention_mask"] = torch.ones_like(thinking_inputs["input_ids"])

            # Generate thinking for all agents at once
            try:
                with torch.no_grad():
                    thinking_outputs = gen_model.generate(
                        **thinking_inputs,
                        max_new_tokens=self.config.thinking_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
            except RuntimeError as e:
                logger.warning(f"Batch thinking generation failed: {e}, falling back to sequential")
                # Fall back to sequential generation
                results = {}
                for agent_id in agent_ids:
                    action, log_prob, thinking_text, full_response, action_text, prompts_dict, action_prompt, action_input_ids, action_ids = self.generate_action(obs_dict[agent_id], agent_id, step, env, model)
                    results[agent_id] = (action, log_prob, thinking_text, full_response, action_text, action_prompt, action_input_ids, action_ids)
                return results

            # Extract thinking text for each agent
            thinking_texts = []
            for i in range(num_agents):
                thinking_ids = thinking_outputs.sequences[i][thinking_inputs.input_ids[i].shape[0]:]
                thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True)
                thinking_texts.append(thinking_text)

            # STAGE 2: Batch generate actions based on thinking
            action_prompts = []
            for i, agent_id in enumerate(agent_ids):
                action_prompt = self.create_action_prompt(obs_texts[i], thinking_texts[i])
                action_prompts.append(action_prompt)

            # Tokenize all action prompts
            action_inputs = self.tokenizer(
                action_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(target_device)

            if "attention_mask" not in action_inputs:
                action_inputs["attention_mask"] = torch.ones_like(action_inputs["input_ids"])

            # Generate actions for all agents at once
            try:
                with torch.no_grad():
                    action_outputs = gen_model.generate(
                        **action_inputs,
                        max_new_tokens=10,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            except RuntimeError as e:
                logger.warning(f"Batch action generation failed: {e}, falling back to sequential")
                # Fall back to sequential generation
                results = {}
                for agent_id in agent_ids:
                    action, log_prob, thinking_text, full_response, action_text, prompts_dict, action_prompt, action_input_ids, action_ids = self.generate_action(obs_dict[agent_id], agent_id, step, env, model)
                    results[agent_id] = (action, log_prob, thinking_text, full_response, action_text, action_prompt, action_input_ids, action_ids)
                return results

            # Extract action text and compute log probs for each agent
            actions = []
            action_texts = []
            action_ids_list = []
            action_input_ids_list = []

            for i in range(num_agents):
                action_ids = action_outputs.sequences[i][action_inputs.input_ids[i].shape[0]:]
                action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                action = self.get_action_from_response(action_text)

                actions.append(action)
                action_texts.append(action_text)
                action_ids_list.append(action_ids)
                action_input_ids_list.append(action_inputs.input_ids[i])

            # Compute log probabilities in batch
            try:
                log_prob_model = self.old_model if self.old_model is not None else gen_model
                log_probs = self.compute_batch_sequence_log_prob(
                    model=log_prob_model,
                    prompt_input_ids_list=action_input_ids_list,
                    generated_ids_list=action_ids_list,
                    device=target_device,
                    pad_token_id=self.tokenizer.pad_token_id,
                    need_grad=False
                )
            except Exception as e:
                logger.warning(f"Batch log prob calculation failed: {e}")
                log_probs = [torch.tensor(-10.0, device=target_device) for _ in range(num_agents)]

            # Assemble results
            results = {}
            for i, agent_id in enumerate(agent_ids):
                full_response = f"{thinking_texts[i]} -> {action_texts[i]}"
                results[agent_id] = (
                    actions[i],
                    log_probs[i],
                    thinking_texts[i],
                    full_response,
                    action_texts[i],
                    action_prompts[i],
                    action_input_ids_list[i],
                    action_ids_list[i]
                )

            return results

        else:
            # Single-stage: batch all agent prompts

            single_stage_prompts = []
            for agent_id in agent_ids:
                obs = obs_dict[agent_id]

                if env is not None:
                    obs_text = self._parse_observation_to_coords(obs, agent_id, env)
                    single_stage_prompt = self._create_single_stage_prompt_from_text(obs_text, agent_id)
                else:
                    single_stage_prompt = self.create_single_stage_prompt(obs, agent_id, step, env)

                single_stage_prompts.append(single_stage_prompt)

            # Tokenize all prompts
            inputs = self.tokenizer(
                single_stage_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(target_device)

            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            # Generate for all agents at once
            try:
                with torch.no_grad():
                    outputs = gen_model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            except RuntimeError as e:
                logger.warning(f"Batch generation failed: {e}, falling back to sequential")
                # Fall back to sequential generation
                results = {}
                for agent_id in agent_ids:
                    action, log_prob, thinking_text, full_response, action_text, prompts_dict, action_prompt, action_input_ids, action_ids = self.generate_action(obs_dict[agent_id], agent_id, step, env, model)
                    results[agent_id] = (action, log_prob, thinking_text, full_response, action_text, action_prompt, action_input_ids, action_ids)
                return results

            # Extract responses and actions
            responses = []
            actions = []
            generated_ids_list = []
            input_ids_list = []

            for i in range(num_agents):
                generated_ids = outputs.sequences[i][inputs.input_ids[i].shape[0]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                action = self.get_action_from_response(response)

                responses.append(response)
                actions.append(action)
                generated_ids_list.append(generated_ids)
                input_ids_list.append(inputs.input_ids[i])

            # Compute log probabilities in batch
            try:
                log_prob_model = self.old_model if self.old_model is not None else gen_model
                log_probs = self.compute_batch_sequence_log_prob(
                    model=log_prob_model,
                    prompt_input_ids_list=input_ids_list,
                    generated_ids_list=generated_ids_list,
                    device=target_device,
                    pad_token_id=self.tokenizer.pad_token_id,
                    need_grad=False
                )
            except Exception as e:
                logger.warning(f"Batch log prob calculation failed: {e}")
                log_probs = [torch.tensor(-10.0, device=target_device) for _ in range(num_agents)]

            # Assemble results
            results = {}
            for i, agent_id in enumerate(agent_ids):
                results[agent_id] = (
                    actions[i],
                    log_probs[i],
                    responses[i],
                    responses[i],
                    responses[i],
                    "",  # action_prompt empty for single-stage
                    input_ids_list[i],
                    generated_ids_list[i]
                )

            return results

    def run_episode(self, use_ref_model: bool = False, log_samples: bool = False,
                   initial_state: Optional[Dict] = None) -> Dict:
        """Run a single episode and collect trajectory."""
        start_time = time.time()

        env = CleanupEnvMove(self.env_config)
        if initial_state is not None:
            env.set_state(initial_state)
            obs = env._observation()
        else:
            obs = env.reset()

        trajectory = {
            "prompts": [],
            "actions": [],
            "responses": [],
            "action_prompts": [],  # Stage 2 prompt for two-stage generation
            "action_texts": [],     # Action text only (for loss computation)
            "rewards": [],
            "log_probs": [],
            "agent_ids": [],
            "observations": [],
            "action_input_ids": [],
            "action_ids": [],
        }

        # Select model (reference model not available in standard GRPO)
        if use_ref_model and self.ref_model is None:
            logger.warning("Reference model requested but not available (standard GRPO). Using current policy.")
            model = self.model
        else:
            model = self.ref_model if use_ref_model else self.model
        total_reward = 0

        for step in range(self.config.max_env_steps):
            actions = {}

            # Batch generate actions for all agents
            batch_results = self.generate_actions_batch(obs, step, env, model, use_ref_model=use_ref_model)

            # Process results for each agent
            for agent_id in range(1, self.config.num_agents + 1):
                action, log_prob, thinking_text, full_response, action_text, action_prompt, action_input_ids, action_ids = batch_results[agent_id]

                actions[agent_id] = action
                # Store both thinking prompt and full response for training
                thinking_prompt = self.create_thinking_prompt(obs[agent_id], agent_id, step, env)
                trajectory["prompts"].append(thinking_prompt)
                trajectory["actions"].append(action)
                trajectory["responses"].append(full_response)
                trajectory["action_prompts"].append(action_prompt)  # Stage 2 prompt
                trajectory["action_texts"].append(action_text)       # Action text only
                # Store old log prob as a detached scalar (no gradients)
                trajectory["log_probs"].append(log_prob.detach().item())
                trajectory["agent_ids"].append(agent_id)
                trajectory["observations"].append(obs[agent_id])
                trajectory["action_input_ids"].append(action_input_ids)
                trajectory["action_ids"].append(action_ids)

                # Log sample if requested
                if log_samples and step == 0 and agent_id == 1:
                    logger.info(f"\n  Sample generation:")
                    logger.info(f"    Obs: {self._parse_observation_to_coords(obs[agent_id], agent_id, env)}")
                    logger.info(f"    Thinking: '{thinking_text[:100]}...'")
                    if action_text:
                        logger.info(f"    Action text (stage 2): '{action_text}'")
                    logger.info(f"    Action: {action}")

            obs, rewards, done, info = env.step(actions)

            # Store rewards for each agent
            for agent_id in range(1, self.config.num_agents + 1):
                trajectory["rewards"].append(rewards[agent_id])
                total_reward += rewards[agent_id]

            if done:
                break

        trajectory["total_reward"] = total_reward
        trajectory["final_scores"] = info["scores"]
        trajectory["steps"] = step + 1

        # Add timing information
        elapsed_time = time.time() - start_time
        trajectory["rollout_time"] = elapsed_time

        # Log memory after rollout (only for first episode of first group)
        if log_samples:
            log_cuda_memory("After episode rollout")

        return trajectory

    def log_episode_to_file(self, trajectory: Dict, group_num: int, episode_idx: int):
        """Log a single episode trajectory to a text file for debugging/monitoring.

        Args:
            trajectory: The trajectory dict from run_episode
            group_num: Current training group number
            episode_idx: Index of this episode within the group
        """
        if not self.config.log_trajectory:
            return

        # Only log on main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        log_path = os.path.join(self.config.output_dir, self.config.trajectory_log_file)

        with open(log_path, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"GROUP {group_num} | EPISODE {episode_idx}\n")
            f.write(f"Total Reward: {trajectory['total_reward']:.2f} | Steps: {trajectory['steps']}\n")
            f.write(f"Final Scores: {trajectory.get('final_scores', 'N/A')}\n")
            f.write("="*80 + "\n\n")

            num_agents = self.config.num_agents
            num_steps = trajectory['steps']

            # Iterate through steps
            for step in range(num_steps):
                f.write(f"--- Step {step + 1} ---\n")

                for agent_idx in range(num_agents):
                    # Calculate index in flat trajectory arrays
                    idx = step * num_agents + agent_idx
                    if idx >= len(trajectory['observations']):
                        break

                    agent_id = trajectory['agent_ids'][idx]
                    obs = trajectory['observations'][idx]
                    action = trajectory['actions'][idx]
                    reward = trajectory['rewards'][idx]
                    response = trajectory['responses'][idx] if idx < len(trajectory['responses']) else "N/A"

                    f.write(f"\n  [Agent {agent_id}]\n")
                    f.write(f"    Observation: {obs}\n")

                    # Truncate long responses for readability
                    if len(response) > 300:
                        thinking_part = response[:300] + "..."
                    else:
                        thinking_part = response
                    f.write(f"    Thinking: {thinking_part}\n")

                    f.write(f"    Action: {action}\n")
                    f.write(f"    Reward: {reward:.2f}\n")

                f.write("\n")

            f.write(f"\n[Episode Summary]\n")
            f.write(f"  Total Reward: {trajectory['total_reward']:.2f}\n")
            f.write(f"  Rollout Time: {trajectory.get('rollout_time', 0):.2f}s\n")
            f.write("\n")

    def _gather_trajectories(self, local_trajectories: List[Dict]) -> List[Dict]:
        """Gather trajectories from all processes in multi-GPU setup.

        Args:
            local_trajectories: Trajectories collected on this process/GPU

        Returns:
            All trajectories combined from all processes (only on main process)
        """
        if self.accelerator is None:
            return local_trajectories

        # Gather trajectories from all processes
        all_trajectories = self.accelerator.gather_for_metrics(local_trajectories)

        # On main process, flatten the list of lists
        if self.accelerator.is_main_process:
            # all_trajectories is a list with num_processes elements, each being a list of trajectories
            flattened = []
            for proc_trajs in all_trajectories:
                if isinstance(proc_trajs, list):
                    flattened.extend(proc_trajs)
                else:
                    flattened.append(proc_trajs)
            return flattened
        else:
            # Non-main processes return empty list (won't be used)
            return []

    def compute_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        """Compute advantages using group relative policy optimization.

        In multi-GPU setup, advantages are computed globally across all GPUs:
        1. Gather all rewards from all GPUs
        2. Compute global mean (and std for GRPO)
        3. Normalize each local trajectory using global statistics

        For DrGrpo: Only subtract mean, no std normalization
        For GRPO: Subtract mean and divide by std
        """
        returns = [traj["total_reward"] for traj in trajectories]
        use_std_norm = (self.config.loss_type.lower() == "grpo")

        if self.config.advantage_normalization and len(returns) > 1:
            if self.accelerator is not None:
                # Multi-GPU: compute global mean and std across all GPUs
                model_device = next(self.model.parameters()).device
                local_returns = torch.tensor(returns, dtype=torch.float32, device=model_device)

                # Gather all returns from all GPUs
                all_returns = self.accelerator.gather(local_returns)

                # Compute global statistics (on all processes)
                global_mean = all_returns.mean().item()

                if use_std_norm:
                    # GRPO: normalize with std
                    global_std = all_returns.std().item() + 1e-8
                    normalized_returns = [(r - global_mean) / global_std for r in returns]
                    if self.accelerator.is_main_process:
                        logger.info(f"  GRPO advantage stats: mean={global_mean:.4f}, std={global_std:.4f}, n={len(all_returns)}")
                else:
                    # DrGrpo: only center (no std normalization)
                    normalized_returns = [(r - global_mean) for r in returns]
                    if self.accelerator.is_main_process:
                        logger.info(f"  DrGrpo advantage stats: mean={global_mean:.4f}, n={len(all_returns)}")
            else:
                # Single GPU: compute local statistics
                mean_return = np.mean(returns)

                if use_std_norm:
                    # GRPO: normalize with std
                    std_return = np.std(returns) + 1e-8
                    normalized_returns = [(r - mean_return) / std_return for r in returns]
                else:
                    # DrGrpo: only center (no std normalization)
                    normalized_returns = [(r - mean_return) for r in returns]
        else:
            normalized_returns = returns

        # Clip advantages for stability
        normalized_returns = [
            np.clip(adv, -self.config.clip_advantage, self.config.clip_advantage)
            for adv in normalized_returns
        ]

        for traj, advantage in zip(trajectories, normalized_returns):
            traj["advantage"] = advantage
            traj["advantages"] = [advantage] * len(traj["rewards"])

        return trajectories

    # def compute_grpo_loss(self, trajectories: List[Dict]) -> Tuple[torch.Tensor, float, int]:
    #     """Compute standard GRPO loss with PPO-style clipped objective using micro-batching.

    #     Standard GRPO uses PPO-style clipping with importance sampling:
    #     ratio = exp(new_log_probs - old_log_probs)
    #     clipped_ratio = clamp(ratio, 1-epsilon, 1+epsilon)
    #     loss = -E[min(ratio * A, clipped_ratio * A)]

    #     where A is the group-normalized advantage (no entropy bonus).

    #     Instead of processing all samples in one large batch (which can OOM), we:
    #     1. Split samples into micro-batches (e.g., 20 samples each)
    #     2. Compute loss for each micro-batch
    #     3. Call backward() on each micro-batch loss (gradients accumulate)
    #     4. Average gradients at the end
    #     5. Caller does optimizer.step() once

    #     Multi-GPU handling (True Data Parallel):
    #     - Each GPU processes DIFFERENT trajectories (its own local samples)
    #     - Each GPU computes loss and gradients independently on its subset
    #     - Gradients accumulate across micro-batches within each GPU
    #     - Gradient sync happens ONLY on the last micro-batch (via no_sync() context)
    #     - This minimizes communication overhead (1 sync instead of N syncs)
    #     - Gradients are automatically averaged across GPUs by Accelerate
    #     - This avoids duplicate computation - each trajectory processed once across all GPUs
    #     """
    #     log_cuda_memory("Before loss computation")

    #     # Get the actual device of the model (important for multi-GPU)
    #     if self.accelerator is not None:
    #         model_device = next(self.model.parameters()).device
    #     else:
    #         model_device = self.device

    #     # Micro-batch size from config (tune based on GPU memory)
    #     micro_batch_size = self.config.micro_batch_size

    #     # Collect all samples from all trajectories
    #     all_prompts = []
    #     all_responses = []
    #     all_advantages = []
    #     all_old_log_probs = []

    #     for traj in trajectories:
    #         advantage = traj["advantage"]
    #         num_steps = len(traj["prompts"])

    #         for i in range(num_steps):
    #             # For two-stage: use action_prompt and action_text (excludes thinking)
    #             # For single-stage: action_prompt is empty, action_text is full response
    #             action_prompt = traj["action_prompts"][i]
    #             action_text = traj["action_texts"][i]

    #             # If action_prompt is empty (single-stage), use thinking_prompt
    #             if action_prompt == "":
    #                 prompt = traj["prompts"][i]
    #                 response = action_text
    #             else:
    #                 # Two-stage: train only on action generation
    #                 prompt = action_prompt
    #                 response = action_text

    #             all_prompts.append(prompt)
    #             all_responses.append(response)
    #             all_advantages.append(advantage)
    #             all_old_log_probs.append(traj["log_probs"][i])

    #     if len(all_prompts) == 0:
    #         logger.warning("No valid samples for loss computation")
    #         return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0, 0

    #     total_samples = len(all_prompts)

    #     # DEBUG: Log old log probs statistics from rollout
    #     logger.info(f"[DEBUG] Collected {total_samples} samples from {len(trajectories)} trajectories")
    #     logger.info(f"[DEBUG] Old log probs from rollout - min: {min(all_old_log_probs):.4f}, "
    #                 f"max: {max(all_old_log_probs):.4f}, mean: {np.mean(all_old_log_probs):.4f}, "
    #                 f"std: {np.std(all_old_log_probs):.4f}")
    #     logger.info(f"[DEBUG] Sample old log probs (first 5): {[f'{x:.4f}' for x in all_old_log_probs[:5]]}")
    #     n_samples = 0
    #     total_loss = 0.0  # Track total loss for averaging
    #     total_clipped = 0  # Track how many samples were clipped

    #     # Process in micro-batches to avoid OOM
    #     num_micro_batches = (total_samples + micro_batch_size - 1) // micro_batch_size

    #     logger.info(f"Processing {total_samples} samples in {num_micro_batches} micro-batches of size {micro_batch_size}")

    #     for batch_idx in range(num_micro_batches):
    #         logger.info(f"Processing micro-batch {batch_idx}")
    #         start_idx = batch_idx * micro_batch_size
    #         end_idx = min((batch_idx + 1) * micro_batch_size, total_samples)

    #         # Determine if this is the last micro-batch
    #         # Only sync gradients on the last batch to minimize communication overhead
    #         is_last_batch = (batch_idx == num_micro_batches - 1)

    #         batch_prompts = all_prompts[start_idx:end_idx]
    #         batch_responses = all_responses[start_idx:end_idx]
    #         batch_advantages = all_advantages[start_idx:end_idx]
    #         batch_old_log_probs = all_old_log_probs[start_idx:end_idx]

    #         # Tokenize micro-batch prompts and responses
    #         prompt_tokens_list = self.tokenizer(
    #             batch_prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=self.config.max_length,
    #             add_special_tokens=True
    #         )

    #         response_tokens_list = self.tokenizer(
    #             [" " + r for r in batch_responses],
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=self.config.max_length,
    #             add_special_tokens=False
    #         )

    #         # Get prompt lengths for each sample (before padding)
    #         prompt_lengths = (prompt_tokens_list.attention_mask).sum(dim=1)

    #         # Concatenate prompt and response tokens for each sample
    #         batch_size = len(batch_prompts)
    #         max_prompt_len = prompt_tokens_list.input_ids.shape[1]
    #         max_response_len = response_tokens_list.input_ids.shape[1]
    #         max_total_len = max_prompt_len + max_response_len

    #         # Initialize batched tensors
    #         batched_input_ids = torch.full(
    #             (batch_size, max_total_len),
    #             self.tokenizer.pad_token_id,
    #             dtype=torch.long
    #         )
    #         batched_attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long)
    #         batched_response_masks = torch.zeros((batch_size, max_total_len - 1), dtype=torch.float32)

    #         valid_indices = []

    #         # Populate batched tensors
    #         for i in range(batch_size):
    #             # Get actual (non-padded) lengths
    #             prompt_len = prompt_lengths[i].item()
    #             response_len = (response_tokens_list.attention_mask[i]).sum().item()
    #             total_len = prompt_len + response_len

    #             if total_len <= 1:
    #                 continue

    #             # Copy prompt tokens
    #             batched_input_ids[i, :prompt_len] = prompt_tokens_list.input_ids[i, :prompt_len]
    #             batched_attention_mask[i, :prompt_len] = 1

    #             # Copy response tokens
    #             batched_input_ids[i, prompt_len:total_len] = response_tokens_list.input_ids[i, :response_len]
    #             batched_attention_mask[i, prompt_len:total_len] = 1

    #             # Create response mask for loss computation
    #             mask_start = max(0, prompt_len - 1)
    #             mask_end = total_len - 1
    #             if mask_start < mask_end:
    #                 batched_response_masks[i, mask_start:mask_end] = 1.0
    #                 valid_indices.append(i)

    #         if len(valid_indices) == 0:
    #             logger.warning(f"No valid samples in micro-batch {batch_idx+1}")
    #             del prompt_tokens_list, response_tokens_list, prompt_lengths
    #             del batched_input_ids, batched_attention_mask, batched_response_masks
    #             continue

    #         # Filter to only valid samples
    #         valid_indices = torch.tensor(valid_indices, dtype=torch.long)
    #         batched_input_ids = batched_input_ids[valid_indices].to(model_device)
    #         batched_attention_mask = batched_attention_mask[valid_indices].to(model_device)
    #         batched_response_masks = batched_response_masks[valid_indices].to(model_device)
    #         advantages_tensor = torch.tensor(
    #             [batch_advantages[i] for i in valid_indices.tolist()],
    #             device=model_device,
    #             dtype=torch.float32
    #         )

    #         # Free tokenization outputs
    #         del prompt_tokens_list, response_tokens_list, prompt_lengths

    #         # Current policy forward pass
    #         inputs = {
    #             'input_ids': batched_input_ids,
    #             'attention_mask': batched_attention_mask
    #         }

    #         outputs = self.model(**inputs)
    #         logits = outputs.logits
    #         logits = torch.clamp(logits, min=-100, max=100)
    #         log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    #         print(f"new logp logits {logits} ")
    #         print(f"shape {logits.shape} ")
    #         print(f"new logp probs {log_probs} ")
    #         # Get log probs for actual tokens (shape: B x (T-1))
    #         token_log_probs = torch.gather(
    #             log_probs[:, :-1, :],
    #             2,
    #             batched_input_ids[:, 1:].unsqueeze(-1)
    #         ).squeeze(-1)  # B x (T-1)

    #         # Masked token log-probs (zeroed where not response)
    #         masked_log_probs = token_log_probs * batched_response_masks  # B x (T-1)

    #         # Compute sequence length for normalization
    #         seq_lens = batched_response_masks.sum(dim=1)  # B (float)

    #         # SEQUENCE log-prob: Average per token (to match rollout storage)
    #         # During rollout, log probs are stored as averages (sum / length)
    #         # So we need to normalize here too for proper ratio computation
    #         sequence_log_probs_sum = masked_log_probs.sum(dim=1)  # B - cumulative
    #         sequence_log_probs = sequence_log_probs_sum / (seq_lens + 1e-8)  # B - averaged per token

    #         # DEBUG: Log new log probs computation details (first micro-batch only)
    #         if batch_idx == 0:
    #             logger.info(f"[DEBUG] Micro-batch {batch_idx} new log probs computation:")
    #             logger.info(f"[DEBUG]   Batch size: {len(valid_indices)}")
    #             logger.info(f"[DEBUG]   Sequence lengths: {seq_lens.cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Cumulative log probs (sum): {sequence_log_probs_sum.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Averaged log probs (sum/len): {sequence_log_probs.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   New log probs - min: {sequence_log_probs.min().item():.4f}, "
    #                         f"max: {sequence_log_probs.max().item():.4f}, "
    #                         f"mean: {sequence_log_probs.mean().item():.4f}")

    #         # Free some memory from big tensors we won't need on GPU after this
    #         del outputs, logits, log_probs, probs, token_log_probs, masked_log_probs

    #         # ---------- Standard GRPO Loss with PPO-style Clipping ----------
    #         # Get old log probs for valid samples (detached, no gradients)
    #         # These are from the rollout and should be treated as constants
    #         old_log_probs_tensor = torch.tensor(
    #             [batch_old_log_probs[i] for i in valid_indices.tolist()],
    #             device=model_device,
    #             dtype=torch.float32,
    #             requires_grad=False
    #         ).detach()

    #         # Compute importance sampling ratios
    #         # ratio = exp(new_log_prob - old_log_prob) = π_new / π_old
    #         log_ratio = sequence_log_probs - old_log_probs_tensor
    #         ratio = torch.exp(log_ratio)

    #         # DEBUG: Log ratio computation (first micro-batch only)
    #         if batch_idx == 0:
    #             logger.info(f"[DEBUG] Ratio computation for micro-batch {batch_idx}:")
    #             logger.info(f"[DEBUG]   Old log probs (from rollout): {old_log_probs_tensor.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   New log probs (computed): {sequence_log_probs.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Log ratio (new - old): {log_ratio.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Ratio (exp(log_ratio)): {ratio.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Ratio stats - min: {ratio.min().item():.4f}, "
    #                         f"max: {ratio.max().item():.4f}, mean: {ratio.mean().item():.4f}")

    #         # Clip the ratios
    #         epsilon = self.config.epsilon
    #         clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    #         # Normalize advantages (important)
    #         adv = advantages_tensor

    #         # Make sure advantages match device
    #         adv = adv.to(sequence_log_probs.device)

    #         # Compute unclipped and clipped objectives
    #         policy_loss_unclipped = ratio * adv
    #         policy_loss_clipped = clipped_ratio * adv

    #         # Take minimum (pessimistic bound) and negate for gradient ascent
    #         policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()

    #         # Track clipping statistics
    #         clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
    #         num_clipped = clipped_mask.sum().item()
    #         total_clipped += num_clipped

    #         # DEBUG: Log detailed clipping statistics (first micro-batch only)
    #         if batch_idx == 0:
    #             clipped_low = (ratio < 1.0 - epsilon).sum().item()
    #             clipped_high = (ratio > 1.0 + epsilon).sum().item()
    #             logger.info(f"[DEBUG] Clipping statistics for micro-batch {batch_idx}:")
    #             logger.info(f"[DEBUG]   Epsilon: {epsilon}")
    #             logger.info(f"[DEBUG]   Valid range: [{1.0 - epsilon:.3f}, {1.0 + epsilon:.3f}]")
    #             logger.info(f"[DEBUG]   Clipped low (ratio < {1.0 - epsilon:.3f}): {clipped_low}/{len(valid_indices)}")
    #             logger.info(f"[DEBUG]   Clipped high (ratio > {1.0 + epsilon:.3f}): {clipped_high}/{len(valid_indices)}")
    #             logger.info(f"[DEBUG]   Total clipped: {num_clipped}/{len(valid_indices)} ({100*num_clipped/len(valid_indices):.1f}%)")
    #             logger.info(f"[DEBUG]   Advantages: {adv.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Policy loss (unclipped): {policy_loss_unclipped.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Policy loss (clipped): {policy_loss_clipped.detach().cpu().numpy()}")
    #             logger.info(f"[DEBUG]   Final policy loss: {policy_loss.item():.4f}")

    #         # Diagnostics logging
    #         if (batch_idx % 10) == 0:  # every 10 micro-batches or adjust as needed
    #             logger.info(f"[loss debug] avg_logp_per_token={sequence_log_probs.mean().item():.4f} "
    #                         f"mean_ratio={ratio.mean().item():.4f} "
    #                         f"clipped={num_clipped}/{len(valid_indices)} "
    #                         f"avg_seq_len={seq_lens.mean().item():.1f} "
    #                         f"old_avg_logp={old_log_probs_tensor.mean().item():.4f}")

    #         # Final scalar loss for this micro-batch (unscaled)
    #         # Standard GRPO = PPO-clipped policy loss (no entropy bonus)
    #         unscaled_loss = policy_loss

    #         # scale by (len(valid_indices) / total_samples) so gradients average correctly across micro-batches
    #         micro_batch_loss = unscaled_loss * (len(valid_indices) / total_samples)

    #         # Backward on scaled loss
    #         # In multi-GPU: only sync gradients on the last micro-batch (reduces communication overhead)
    #         if self.accelerator is not None and not is_last_batch and hasattr(self.model, 'no_sync'):
    #             # Defer gradient synchronization until last batch
    #             with self.model.no_sync():
    #                 micro_batch_loss.backward()
    #         else:
    #             # Last batch (or single GPU): sync gradients normally
    #             micro_batch_loss.backward()

    #         # accumulate metrics
    #         total_loss += unscaled_loss.item() * len(valid_indices)
    #         n_samples += len(valid_indices)

    #         # cleanup
    #         del batched_input_ids, batched_attention_mask, batched_response_masks
    #         del inputs, sequence_log_probs, unscaled_loss, micro_batch_loss, advantages_tensor
    #         del old_log_probs_tensor, ratio, clipped_ratio, policy_loss_unclipped, policy_loss_clipped

    #         # Periodic cache clearing
    #         if (batch_idx + 1) % 5 == 0:
    #             torch.cuda.empty_cache()
    #         log_cuda_memory(f"After micro-batch {batch_idx}")

    #     if n_samples == 0:
    #         logger.warning("No valid samples processed in any micro-batch")
    #         return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0, 0

    #     # Compute averages
    #     avg_loss = total_loss / n_samples
    #     clip_fraction = total_clipped / n_samples

    #     # DEBUG: Final summary statistics
    #     logger.info(f"[DEBUG] ========== Loss Computation Summary ==========")
    #     logger.info(f"[DEBUG] Total samples processed: {n_samples} across {num_micro_batches} micro-batches")
    #     logger.info(f"[DEBUG] Average loss: {avg_loss:.4f}")
    #     logger.info(f"[DEBUG] Total clipped samples: {total_clipped}/{n_samples} ({100*clip_fraction:.1f}%)")
    #     logger.info(f"[DEBUG] ================================================")

    #     # Note: We don't return a loss tensor with gradients since backward() was already called
    #     # The caller should just call optimizer.step() without calling loss.backward()
    #     # Return the actual mean loss value as a scalar for logging
    #     loss_scalar = torch.tensor(avg_loss, device=model_device, requires_grad=False)

    #     log_cuda_memory("After loss computation")

    #     # Note: Gradients were synchronized across GPUs on the last micro-batch backward()
    #     # The no_sync() context deferred synchronization to minimize communication overhead
    #     # Gradients are now ready for optimizer.step() - no manual all-reduce needed

    #     return loss_scalar, clip_fraction, n_samples

    def create_minibatch_iterator(self, trajectories: List[Dict], minibatch_size: int):
        """Create iterator over mini-batches of trajectories.

        Shuffles trajectories at the start, then yields mini-batches sequentially.
        This is called once per inner epoch.

        Args:
            trajectories: Full buffer of trajectories
            minibatch_size: Number of trajectories per mini-batch

        Yields:
            Mini-batches of trajectories
        """
        import random

        # Shuffle trajectories for this epoch
        indices = list(range(len(trajectories)))
        random.shuffle(indices)

        # Yield mini-batches
        for start_idx in range(0, len(indices), minibatch_size):
            end_idx = min(start_idx + minibatch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            yield [trajectories[i] for i in batch_indices]

    def compute_grpo_loss(self, trajectories: List[Dict]) -> Tuple[torch.Tensor, float, int]:
        """
        Compute GRPO loss on a mini-batch of trajectories.

        This is called during inner optimization epochs. The trajectories parameter
        is a mini-batch sampled from the full buffer.

        Args:
            trajectories: Mini-batch of trajectories (already sampled)

        Returns:
            loss: Loss tensor with gradients (caller should call .backward())
            clip_fraction: Fraction of samples that were clipped
            n_samples: Number of valid samples processed
        """
        # Get device
        if self.accelerator is not None:
            model_device = next(self.model.parameters()).device
        else:
            model_device = self.device

        # Flatten trajectories into lists of (prompt_ids, generated_ids, advantage, old_log_prob)
        all_prompt_ids = []
        all_generated_ids = []
        all_advantages = []
        all_old_log_probs = []

        for traj in trajectories:
            advantage = traj["advantage"]

            num_steps = len(traj["prompts"])
            for i in range(num_steps):
                # Use stored token IDs directly from trajectory buffer
                action_input_ids = traj["action_input_ids"][i]
                action_ids = traj["action_ids"][i]

                # Skip if IDs are None (error case during rollout)
                if action_input_ids is None or action_ids is None:
                    continue

                all_prompt_ids.append(action_input_ids)
                all_generated_ids.append(action_ids)
                all_advantages.append(advantage)
                all_old_log_probs.append(traj["log_probs"][i])

        if len(all_prompt_ids) == 0:
            return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0, 0

        # Compute new log-probs using CURRENT model (batched)
        # Filter out empty sequences first
        valid_prompt_ids = []
        valid_generated_ids = []
        valid_indices = []

        for i in range(len(all_prompt_ids)):
            # Sanity check - ensure we have valid generated tokens
            if len(all_generated_ids[i]) > 0:
                valid_prompt_ids.append(all_prompt_ids[i])
                valid_generated_ids.append(all_generated_ids[i])
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0, 0

        # Compute log-probs for all sequences in one batched call
        new_log_probs = self.compute_batch_sequence_log_prob(
            model=self.model,
            prompt_input_ids_list=valid_prompt_ids,
            generated_ids_list=valid_generated_ids,
            device=model_device,
            pad_token_id=self.tokenizer.pad_token_id,
            need_grad=True  # Gradients needed for training
        )

        # Old log-probs (from old model during rollout, fixed during inner epochs)
        old_log_probs = torch.tensor(
            [all_old_log_probs[i] for i in valid_indices],
            device=model_device,
            dtype=torch.float32
        )

        # Advantages
        advantages = torch.tensor(
            [all_advantages[i] for i in valid_indices],
            device=model_device,
            dtype=torch.float32
        )

        # PPO-style ratio clipping
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        epsilon = self.config.epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        # Loss terms
        loss_unclipped = ratio * advantages
        loss_clipped = clipped_ratio * advantages

        # Maximize objective = Minimize negative loss
        policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()

        # Compute clipping statistics
        with torch.no_grad():
            clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
            clip_fraction = clipped_mask.float().mean().item()

        n_samples = len(valid_indices)

        # Return loss with gradients intact (caller will call .backward())
        return policy_loss, clip_fraction, n_samples

    def flatten_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Flatten a list of trajectories into individual samples.

        Each trajectory contains multiple steps. This method extracts each step
        as a separate sample containing (prompt_ids, action_ids, advantage, old_log_prob).

        Args:
            trajectories: List of trajectory dictionaries

        Returns:
            List of sample dictionaries, each containing:
                - prompt_ids: Input token IDs
                - action_ids: Generated token IDs
                - advantage: Advantage value for this sample
                - old_log_prob: Log probability from old model
        """
        all_samples = []

        for traj in trajectories:
            advantage = traj["advantage"]
            num_steps = len(traj["prompts"])

            for i in range(num_steps):
                # Use stored token IDs directly from trajectory buffer
                action_input_ids = traj["action_input_ids"][i]
                action_ids = traj["action_ids"][i]

                # Skip if IDs are None (error case during rollout)
                if action_input_ids is None or action_ids is None:
                    continue

                # Skip empty sequences
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
        self,
        samples: List[Dict],
        device
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Compute GRPO loss on a micro-batch of flattened samples.

        This method is used for gradient accumulation. It processes a small
        subset of samples to avoid OOM.

        Args:
            samples: List of sample dictionaries (from flatten_trajectories)
            device: Device to put tensors on

        Returns:
            loss: Loss tensor with gradients
            clip_fraction: Fraction of samples that were clipped
            n_samples: Number of valid samples processed
        """
        if len(samples) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

        # Extract data from samples
        prompt_ids_list = [s["prompt_ids"] for s in samples]
        action_ids_list = [s["action_ids"] for s in samples]
        advantages_list = [s["advantage"] for s in samples]
        old_log_probs_list = [s["old_log_prob"] for s in samples]

        # Compute new log-probs using CURRENT model (batched)
        new_log_probs = self.compute_batch_sequence_log_prob(
            model=self.model,
            prompt_input_ids_list=prompt_ids_list,
            generated_ids_list=action_ids_list,
            device=device,
            pad_token_id=self.tokenizer.pad_token_id,
            need_grad=True  # Gradients needed for training
        )
        print(f"[DEBUG] new log probs {new_log_probs} ")
        # Old log-probs (from old model during rollout, fixed during inner epochs)
        old_log_probs = torch.tensor(
            old_log_probs_list,
            device=device,
            dtype=torch.float32
        )

        # Advantages
        advantages = torch.tensor(
            advantages_list,
            device=device,
            dtype=torch.float32
        )

        # PPO-style ratio clipping
        log_ratio = new_log_probs - old_log_probs
        print(f"[DEBUG] log ratio {log_ratio} ")
        ratio = torch.exp(log_ratio)

        epsilon = self.config.epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        # Loss terms
        loss_unclipped = ratio * advantages
        loss_clipped = clipped_ratio * advantages

        # Maximize objective = Minimize negative loss
        policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()

        # Compute clipping statistics
        with torch.no_grad():
            clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
            clip_fraction = clipped_mask.float().mean().item()

        n_samples = len(samples)

        return policy_loss, clip_fraction, n_samples

    def train(self):
        """Main training loop."""
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)

            logger.info("\n" + "="*70)
            logger.info("GRPO Training with Text Actions (Two-Stage)")
            logger.info("="*70)
            logger.info(f"\n[Model] {self.config.model_name}")
            logger.info(f"[Loss Type] {self.config.loss_type.upper()}")
            logger.info(f"[Environment] Move (directional actions)")
            logger.info(f"[Actions] Plain text: {', '.join(self.action_words)}")
            logger.info(f"[Two-Stage] {self.config.use_two_stage}")
            if self.config.use_two_stage:
                logger.info(f"[Thinking Tokens] {self.config.thinking_tokens}")
            logger.info(f"[Agents] {self.config.num_agents}")
            logger.info(f"[Episodes] {self.config.num_episodes}")
            if self.accelerator is not None:
                episodes_per_group = self.config.episodes_per_gpu * self.accelerator.num_processes
                logger.info(
                    f"[Multi-GPU] {self.accelerator.num_processes} GPUs × {self.config.episodes_per_gpu} episodes/GPU "
                    f"= {episodes_per_group} episodes per group"
                )

                # Warn if episodes_per_update doesn't match actual behavior
                if hasattr(self.config, 'episodes_per_update') and self.config.episodes_per_update != episodes_per_group:
                    logger.warning(
                        f"⚠️  episodes_per_update={self.config.episodes_per_update} is deprecated and ignored in multi-GPU mode.\n"
                        f"    Actual episodes per group: {episodes_per_group} "
                        f"({self.config.episodes_per_gpu} per GPU × {self.accelerator.num_processes} GPUs)\n"
                        f"    Consider using only episodes_per_gpu in future."
                    )
            logger.info("="*70 + "\n")

        episode = 0
        best_reward = float('-inf')
        group_num = 0

        while episode < self.config.num_episodes:
            # Store starting episode for this group (for accurate logging)
            group_start_episode = episode

            if self.accelerator is None or self.accelerator.is_main_process:
                # Calculate expected episode range for this group
                if self.accelerator is not None:
                    expected_episodes = self.config.episodes_per_gpu * self.accelerator.num_processes
                else:
                    expected_episodes = self.config.episodes_per_update
                logger.info(f"\n[Group {group_num}] Episodes {episode}-{episode + expected_episodes - 1} (expected)")

            # 📸 Step 1: Update old model (θ_old ← θ)
            # This happens at the start of each group, before rollout
            self.update_old_model()

            # 📊 Step 2: Collect trajectories using current model (but store old model's log-probs)
            # Each GPU processes its own trajectories independently (data parallel)
            # Each GPU processes its own trajectories independently (data parallel)
            if self.accelerator is not None:
                # Multi-GPU: each process collects episodes_per_gpu trajectories
                trajectories = []
                num_local_episodes = self.config.episodes_per_gpu

                for i in range(num_local_episodes):
                    try:
                        log_samples = (i == 0 and group_num % self.config.log_interval == 0 and self.accelerator.is_main_process)

                        traj = self.run_episode(use_ref_model=False, log_samples=log_samples)
                        trajectories.append(traj)

                        if not log_samples:
                            logger.info(f"  [GPU{self.accelerator.process_index}] Ep{i}: R={traj['total_reward']:.2f}, Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s")
                    except Exception as e:
                        logger.error(f"  [GPU{self.accelerator.process_index}] Episode {i} failed: {e}", exc_info=True)
                        continue

                # Data parallel: each GPU will compute loss on its own trajectories
                # Gradients will be automatically synced by Accelerate before optimizer.step()
                # NO GATHER - this avoids duplicate loss computation on all GPUs
            else:
                # Single GPU: collect all trajectories on one device
                trajectories = []
                for i in range(self.config.episodes_per_update):
                    try:
                        log_samples = (i == 0 and group_num % self.config.log_interval == 0)

                        traj = self.run_episode(use_ref_model=False, log_samples=log_samples)
                        trajectories.append(traj)

                        if not log_samples:
                            logger.info(f"  Ep{episode + i}: R={traj['total_reward']:.2f}, Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s")
                    except Exception as e:
                        logger.error(f"Episode {episode + i} failed: {e}", exc_info=True)
                        continue

            # Update episode counter (gather count for multi-GPU)
            if self.accelerator is not None:
                local_count = len(trajectories)
                count_tensor = torch.tensor([local_count], dtype=torch.long, device=next(self.model.parameters()).device)
                all_counts = self.accelerator.gather(count_tensor)
                total_collected = all_counts.sum().item()

                # Debug logging
                if self.accelerator.is_main_process:
                    logger.info(f"[DEBUG Episode Counter] all_counts per GPU: {all_counts.tolist()}, total: {total_collected}")
                    logger.info(f"[DEBUG Episode Counter] Episode counter: {episode} -> {episode + total_collected}")

                episode += total_collected

                # Enhanced logging with actual episodes collected
                if self.accelerator.is_main_process:
                    actual_end = episode - 1
                    expected_episodes = self.config.episodes_per_gpu * self.accelerator.num_processes
                    logger.info(
                        f"  ✓ Collected Episodes {group_start_episode}-{actual_end} "
                        f"({total_collected}/{expected_episodes} episodes, "
                        f"{[count.item() for count in all_counts]} per GPU)"
                    )
            else:
                episode += len(trajectories)

            # Log one random episode trajectory to file for debugging/monitoring
            if len(trajectories) > 0 and self.config.log_trajectory:
                if self.accelerator is None or self.accelerator.is_main_process:
                    import random
                    random_idx = random.randint(0, len(trajectories) - 1)
                    self.log_episode_to_file(trajectories[random_idx], group_num, random_idx)
                    logger.info(f"  📝 Logged episode {random_idx} trajectory to {self.config.trajectory_log_file}")

            if len(trajectories) == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.warning("No valid trajectories collected, skipping update")
                group_num += 1
                continue

            # Compute advantages (on local trajectories with global normalization)
            trajectories = self.compute_advantages(trajectories)

            # Synchronization barrier after advantage computation
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            # Gather statistics for logging (but NOT trajectories for loss)
            # This is just for monitoring - each GPU still computes loss on its own data
            if self.accelerator is not None:
                # Gather rewards and steps from all GPUs for logging
                local_rewards = [t["total_reward"] for t in trajectories]
                local_steps = [t["steps"] for t in trajectories]
                local_times = [t["rollout_time"] for t in trajectories]

                # Convert to tensors and gather
                model_device = next(self.model.parameters()).device
                rewards_tensor = torch.tensor(local_rewards, dtype=torch.float32, device=model_device)
                steps_tensor = torch.tensor(local_steps, dtype=torch.float32, device=model_device)
                times_tensor = torch.tensor(local_times, dtype=torch.float32, device=model_device)

                all_rewards = self.accelerator.gather(rewards_tensor)
                all_steps = self.accelerator.gather(steps_tensor)
                all_times = self.accelerator.gather(times_tensor)

                # Compute statistics from all GPUs (on all processes for consistency)
                all_rewards_list = all_rewards.cpu().tolist()
                all_steps_list = all_steps.cpu().tolist()
                all_times_list = all_times.cpu().tolist()

                avg_reward = np.mean(all_rewards_list)
                avg_steps = np.mean(all_steps_list)
                max_reward = np.max(all_rewards_list)
                min_reward = np.min(all_rewards_list)
                std_reward = np.std(all_rewards_list)
                avg_rollout_time = np.mean(all_times_list)
                total_rollout_time = sum(all_times_list)

                if self.accelerator.is_main_process:
                    self.episode_rewards.append(avg_reward)
                    self.episode_steps.append(avg_steps)

                    logger.info(f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], Steps: {avg_steps:.1f}")
                    logger.info(f"  Rollout Time: avg={avg_rollout_time:.2f}s, total={total_rollout_time:.2f}s")

                    # Log episode metrics to wandb
                    if self.config.use_wandb:
                        wandb.log({
                            "episode": episode,
                            "reward/mean": avg_reward,
                            "reward/std": std_reward,
                            "reward/min": min_reward,
                            "reward/max": max_reward,
                            "episode_steps": avg_steps,
                            "rollout_time/avg": avg_rollout_time,
                            "rollout_time/total": total_rollout_time,
                        }, step=episode)

                    # Action distribution (gather from all GPUs)
                    # For simplicity, just use local actions (each GPU sees similar distribution)
                    all_actions = [t for traj in trajectories for t in traj["actions"]]
                    action_counts = {}
                    for action in all_actions:
                        action_counts[action] = action_counts.get(action, 0) + 1

                    if len(all_actions) > 0:
                        total_actions = sum(action_counts.values())
                        action_dist = " | ".join([
                            f"{action}:{action_counts.get(action, 0)*100//total_actions}%"
                            for action in self.action_words
                        ])
                        logger.info(f"  Actions (GPU0 sample): {action_dist}")

                        # Log action distribution to wandb
                        if self.config.use_wandb:
                            action_dist_dict = {
                                f"actions/{action}": action_counts.get(action, 0) / total_actions
                                for action in self.action_words
                            }
                            wandb.log(action_dist_dict, step=episode)

                    if std_reward < 0.01:
                        logger.warning("  ⚠ Policy collapse detected!")
            else:
                # Single GPU: compute statistics normally
                avg_reward = np.mean([t["total_reward"] for t in trajectories])
                avg_steps = np.mean([t["steps"] for t in trajectories])
                max_reward = np.max([t["total_reward"] for t in trajectories])
                min_reward = np.min([t["total_reward"] for t in trajectories])
                std_reward = np.std([t["total_reward"] for t in trajectories])
                avg_rollout_time = np.mean([t["rollout_time"] for t in trajectories])
                total_rollout_time = sum([t["rollout_time"] for t in trajectories])
                self.episode_rewards.append(avg_reward)
                self.episode_steps.append(avg_steps)

                logger.info(f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], Steps: {avg_steps:.1f}")
                logger.info(f"  Rollout Time: avg={avg_rollout_time:.2f}s, total={total_rollout_time:.2f}s")

                # Log episode metrics to wandb
                if self.config.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "reward/mean": avg_reward,
                        "reward/std": std_reward,
                        "reward/min": min_reward,
                        "reward/max": max_reward,
                        "episode_steps": avg_steps,
                        "rollout_time/avg": avg_rollout_time,
                        "rollout_time/total": total_rollout_time,
                    }, step=episode)

                # Action distribution
                all_actions = [t for traj in trajectories for t in traj["actions"]]
                action_counts = {}
                for action in all_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1

                total_actions = sum(action_counts.values())
                action_dist = " | ".join([
                    f"{action}:{action_counts.get(action, 0)*100//total_actions}%"
                    for action in self.action_words
                ])
                logger.info(f"  Actions: {action_dist}")

                # Log action distribution to wandb
                if self.config.use_wandb:
                    action_dist_dict = {
                        f"actions/{action}": action_counts.get(action, 0) / total_actions
                        for action in self.action_words
                    }
                    wandb.log(action_dist_dict, step=episode)

                if std_reward < 0.01:
                    logger.warning("  ⚠ Policy collapse detected!")

            # ⚙️ Step 4: Inner optimization epochs
            # Train for multiple epochs on the collected buffer
            # Old model stays FIXED during all inner epochs
            try:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info(f"\n  Inner Optimization ({self.config.num_inner_epochs} epochs, minibatch={self.config.minibatch_size}):")

                # Track metrics across all inner epochs
                epoch_losses = []
                epoch_clip_fracs = []
                final_grad_norm = 0.0

                for inner_epoch in range(self.config.num_inner_epochs):
                    # Create mini-batch iterator (shuffles trajectories)
                    minibatch_iterator = self.create_minibatch_iterator(
                        trajectories,
                        self.config.minibatch_size
                    )

                    # Track metrics for this epoch
                    epoch_loss_sum = 0.0
                    epoch_clip_sum = 0.0
                    epoch_samples = 0
                    num_minibatches = 0

                    for minibatch_idx, minibatch in enumerate(minibatch_iterator):
                        self.optimizer.zero_grad()

                        # Get device
                        if self.accelerator is not None:
                            model_device = next(self.model.parameters()).device
                        else:
                            model_device = self.device

                        # 1. FLATTEN MINIBATCH into individual samples
                        all_samples = self.flatten_trajectories(minibatch)
                        total_samples = len(all_samples)

                        if total_samples == 0:
                            continue

                        # 2. CALCULATE NUMBER OF MICRO-BATCHES for gradient accumulation
                        micro_batch_size = self.config.micro_batch_size
                        num_chunks = (total_samples + micro_batch_size - 1) // micro_batch_size

                        # Track metrics across micro-batches
                        total_loss_sum = 0.0
                        total_clip_fraction_sum = 0.0
                        total_n_samples = 0

                        # 3. INNER LOOP: Process micro-batches with gradient accumulation
                        for i in range(0, total_samples, micro_batch_size):
                            # Slice the data for this micro-batch
                            micro_batch = all_samples[i : i + micro_batch_size]

                            # Compute loss on this micro-batch
                            loss, clip_fraction, n_samples = self.compute_loss_on_samples(
                                micro_batch,
                                model_device
                            )

                            if n_samples > 0:
                                # CRITICAL: Normalize loss by number of chunks
                                # This ensures the final gradient is the AVERAGE of the whole minibatch,
                                # not the SUM (which would be like increasing LR by num_chunks)
                                normalized_loss = loss / num_chunks

                                # Accumulate gradients (backward pass)
                                normalized_loss.backward()

                                # Track metrics (using unnormalized loss for logging)
                                total_loss_sum += loss.item() * n_samples
                                total_clip_fraction_sum += clip_fraction * n_samples
                                total_n_samples += n_samples

                        # 4. OPTIMIZER STEP (once per full minibatch, after all micro-batches)
                        if total_n_samples > 0:
                            # Gradient clipping
                            if self.accelerator is not None:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.max_grad_norm
                                )
                            else:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.max_grad_norm
                                )

                            # Optimizer step
                            self.optimizer.step()

                            # Compute average metrics
                            avg_loss = total_loss_sum / total_n_samples
                            avg_clip_fraction = total_clip_fraction_sum / total_n_samples

                            # Track metrics for epoch
                            epoch_loss_sum += total_loss_sum
                            epoch_clip_sum += total_clip_fraction_sum
                            epoch_samples += total_n_samples
                            final_grad_norm = grad_norm
                            num_minibatches += 1

                            # Log first minibatch of first epoch
                            if inner_epoch == 0 and minibatch_idx == 0:
                                if self.accelerator is None or self.accelerator.is_main_process:
                                    logger.info(f"    [Epoch 1/{self.config.num_inner_epochs}, Batch 1] Loss={avg_loss:.4f}, ClipFrac={avg_clip_fraction:.3f}, MicroBatches={num_chunks}")

                    # Compute epoch averages
                    if epoch_samples > 0:
                        epoch_avg_loss = epoch_loss_sum / epoch_samples
                        epoch_avg_clip = epoch_clip_sum / epoch_samples
                        epoch_losses.append(epoch_avg_loss)
                        epoch_clip_fracs.append(epoch_avg_clip)

                        # Log last epoch
                        if inner_epoch == self.config.num_inner_epochs - 1:
                            if self.accelerator is None or self.accelerator.is_main_process:
                                logger.info(f"    [Epoch {inner_epoch+1}/{self.config.num_inner_epochs}] Avg Loss={epoch_avg_loss:.4f}, Avg ClipFrac={epoch_avg_clip:.3f}, Batches={num_minibatches}")

                # Scheduler step (once per group, not per minibatch!)
                if self.training_step >= 0:
                    self.scheduler.step()

                self.training_step += 1

                # Log final metrics
                if len(epoch_losses) > 0:
                    final_loss = epoch_losses[-1]
                    final_clip_frac = epoch_clip_fracs[-1]
                    avg_loss_all_epochs = sum(epoch_losses) / len(epoch_losses)

                    if self.accelerator is None or self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(f"  Final: Loss={final_loss:.4f} (avg={avg_loss_all_epochs:.4f}), ClipFrac={final_clip_frac:.3f}, GradNorm={final_grad_norm:.4f}, LR={current_lr:.2e}")

                        # Log to wandb
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": final_loss,
                                "train/loss_avg_all_epochs": avg_loss_all_epochs,
                                "train/clip_fraction": final_clip_frac,
                                "train/grad_norm": final_grad_norm,
                                "train/learning_rate": current_lr,
                                "train/training_step": self.training_step,
                                "train/num_inner_epochs": self.config.num_inner_epochs,
                            }, step=episode)
                else:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        logger.warning("No valid samples in any epoch, skipping update")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.error(f"Training step failed: {e}", exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                group_num += 1
                continue

            # Track best reward for checkpoint saving
            if avg_reward > best_reward:
                best_reward = avg_reward

            # Save checkpoint
            if (self.accelerator is None or self.accelerator.is_main_process) and \
               (episode % self.config.save_steps == 0 or episode >= self.config.num_episodes):

                if avg_reward >= best_reward:
                    checkpoint_path = os.path.join(self.config.output_dir, "best_model")

                    try:
                        if self.config.use_lora:
                            if self.accelerator is not None:
                                # Unwrap model manually to avoid DeepSpeed import issues
                                if hasattr(self.model, 'module'):
                                    unwrapped_model = self.model.module
                                else:
                                    unwrapped_model = self.model
                            else:
                                unwrapped_model = self.model
                            unwrapped_model.save_pretrained(checkpoint_path)
                        else:
                            if self.accelerator is not None:
                                if hasattr(self.model, 'module'):
                                    self.model.module.save_pretrained(checkpoint_path)
                                else:
                                    self.model.save_pretrained(checkpoint_path)
                            else:
                                self.model.save_pretrained(checkpoint_path)

                        self.tokenizer.save_pretrained(checkpoint_path)
                        logger.info(f"  Saved best model (R={best_reward:.2f})")
                    except Exception as e:
                        logger.warning(f"  Failed to save model: {e}")

                # Save stats
                stats = {
                    "episode": episode,
                    "group": group_num,
                    "rewards": self.episode_rewards,
                    "steps": self.episode_steps,
                    "best_reward": best_reward,
                    "training_step": self.training_step
                }
                with open(os.path.join(self.config.output_dir, "training_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2)

            # Mid-training evaluation (if eval_interval is set)
            if self.config.eval_interval > 0 and episode % self.config.eval_interval == 0 and episode < self.config.num_episodes:
                # ALL processes must participate in evaluation to avoid NCCL deadlock
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info(f"\n=== Mid-Training Evaluation (Episode {episode}) ===")

                # All GPUs must call evaluate() for proper synchronization
                self.evaluate(num_episodes=self.config.num_eval_episodes, current_episode=episode)

                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info("=== Resuming Training ===\n")

            group_num += 1

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"\n=== Training Complete ===")
            logger.info(f"Best reward: {best_reward:.2f}, Total groups: {group_num}\n")

        return self.model

    def evaluate(self, num_episodes: int = 20, current_episode: int = None):
        """Evaluate the trained model.

        When using multi-GPU, evaluation states are split across GPUs in round-robin fashion:
        - GPU 0: episodes 0, 2, 4, 6, ...
        - GPU 1: episodes 1, 3, 5, 7, ...
        - GPU 2: episodes 2, 4, 6, 8, ...
        etc.

        Results are gathered and averaged across all GPUs.

        Args:
            num_episodes: Number of episodes to evaluate
            current_episode: Current training episode (for wandb logging)
        """
        eval_start_time = time.time()
        actual_num_episodes = min(num_episodes, len(self.eval_states))

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"\n=== Evaluation ({actual_num_episodes} episodes) ===")

        self.model.eval()

        # Split episodes across GPUs in round-robin fashion
        if self.accelerator is not None:
            num_processes = self.accelerator.num_processes
            process_index = self.accelerator.process_index

            # Each GPU takes every num_processes-th episode starting from process_index
            # GPU 0: 0, 2, 4, 6, ...
            # GPU 1: 1, 3, 5, 7, ...
            local_episode_indices = list(range(process_index, actual_num_episodes, num_processes))

            if self.accelerator.is_main_process:
                logger.info(f"Splitting {actual_num_episodes} episodes across {num_processes} GPUs (round-robin)")
            logger.info(f"  GPU {process_index}: will evaluate {len(local_episode_indices)} episodes: {local_episode_indices}")
        else:
            # Single GPU: evaluate all episodes
            process_index = 0
            local_episode_indices = list(range(actual_num_episodes))

        # Evaluate episodes assigned to this GPU
        local_rewards = []
        local_episode_times = []

        for i in local_episode_indices:
            try:
                initial_state = self.eval_states[i]
                traj = self.run_episode(use_ref_model=False, log_samples=False, initial_state=initial_state)

                # Extract values we need
                total_reward = traj["total_reward"]
                rollout_time = traj["rollout_time"]
                steps = traj["steps"]

                local_rewards.append(total_reward)
                local_episode_times.append(rollout_time)

                # Log with global episode number (i+1) not local index
                logger.info(f"  GPU{process_index if self.accelerator else 0} Ep{i+1}: R={total_reward:.2f}, Steps={steps}, Time={rollout_time:.2f}s")

                # Delete trajectory to free memory immediately
                del traj

                # Clear CUDA cache periodically to prevent fragmentation
                if torch.cuda.is_available() and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"GPU {process_index if self.accelerator else 0}: Evaluation episode {i+1} failed: {e}")
                continue

        # Gather results from all GPUs
        if self.accelerator is not None:
            # Get the actual device of the model (important for multi-GPU)
            model_device = next(self.model.parameters()).device

            # Get max length across all processes
            local_len = len(local_rewards)
            max_len_tensor = torch.tensor([local_len], dtype=torch.long, device=model_device)
            all_lens = self.accelerator.gather(max_len_tensor)

            if self.accelerator.is_main_process:
                max_len = all_lens.max().item()
            else:
                max_len = local_len  # Will be broadcast

            # Pad to max length
            if len(local_rewards) < max_len:
                local_rewards_padded = local_rewards + [0.0] * (max_len - len(local_rewards))
                local_times_padded = local_episode_times + [0.0] * (max_len - len(local_episode_times))
            else:
                local_rewards_padded = local_rewards
                local_times_padded = local_episode_times

            local_rewards_tensor = torch.tensor(local_rewards_padded, dtype=torch.float32, device=model_device)
            local_times_tensor = torch.tensor(local_times_padded, dtype=torch.float32, device=model_device)

            # Gather all rewards and times from all GPUs
            all_rewards = self.accelerator.gather(local_rewards_tensor)  # Shape: [num_processes * max_len]
            all_times = self.accelerator.gather(local_times_tensor)

            # Convert back to lists (only on main process)
            if self.accelerator.is_main_process:
                # Flatten and remove padding
                all_rewards_list = all_rewards.cpu().tolist()
                all_times_list = all_times.cpu().tolist()

                # Un-pad: only take actual results based on original lengths
                rewards = []
                episode_times = []
                for proc_idx in range(self.accelerator.num_processes):
                    start_idx = proc_idx * max_len
                    actual_len = all_lens[proc_idx].item()
                    rewards.extend(all_rewards_list[start_idx:start_idx + actual_len])
                    episode_times.extend(all_times_list[start_idx:start_idx + actual_len])

                logger.info(f"Gathered {len(rewards)} total results from {self.accelerator.num_processes} GPUs")
            else:
                rewards = []
                episode_times = []
        else:
            rewards = local_rewards
            episode_times = local_episode_times

        if len(rewards) == 0:
            self.model.train()
            return 0.0, 0.0

        # Compute statistics (only on main process)
        if self.accelerator is None or self.accelerator.is_main_process:
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            total_eval_time = time.time() - eval_start_time
            avg_episode_time = np.mean(episode_times) if episode_times else 0.0

            logger.info(f"\nReward: {avg_reward:.2f}±{std_reward:.2f} [{min(rewards):.2f}, {max(rewards):.2f}]")
            logger.info(f"Evaluation Time: avg={avg_episode_time:.2f}s/episode, total={total_eval_time:.2f}s")
            logger.info(f"Total episodes evaluated: {len(rewards)}\n")

            # Log evaluation metrics to wandb
            if self.config.use_wandb and current_episode is not None:
                wandb.log({
                    "eval/reward_mean": avg_reward,
                    "eval/reward_std": std_reward,
                    "eval/reward_min": min(rewards),
                    "eval/reward_max": max(rewards),
                    "eval/episode_time": avg_episode_time,
                    "eval/total_time": total_eval_time,
                    "eval/num_episodes": len(rewards),
                }, step=current_episode)
        else:
            avg_reward = 0.0
            std_reward = 0.0

        self.model.train()
        return avg_reward, std_reward

    def visualize_rollout(self, use_ref_model: bool = False, save_to_file: Optional[str] = None):
        """Visualize a single rollout/trajectory step-by-step.

        Args:
            use_ref_model: If True, use reference model instead of current policy
            save_to_file: Optional filepath to save visualization (e.g., "rollout.txt")
        """
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return  # Only run on main process

        rollout_start_time = time.time()

        logger.info("\n" + "="*80)
        logger.info("TRAJECTORY VISUALIZATION (TEXT ACTIONS)")
        logger.info("="*80)

        # Run one episode with detailed logging
        env = CleanupEnvMove(self.env_config)
        obs = env.reset()
        initial_dirt_count = sum(row.count('#') for row in env.items)

        # Select model (reference model not available in standard GRPO)
        if use_ref_model and self.ref_model is None:
            logger.warning("Reference model requested but not available (standard GRPO). Using current policy.")
            model = self.model
        else:
            model = self.ref_model if use_ref_model else self.model
        model.eval()

        total_reward = 0
        output_lines = []
        step_times = []

        def log_and_save(line):
            """Log to console and save to output buffer."""
            logger.info(line)
            output_lines.append(line)

        log_and_save(f"\n{'='*80}")
        log_and_save(f"INITIAL STATE")
        log_and_save(f"{'='*80}")
        log_and_save("\nGlobal Grid:")
        for line in env.render().split('\n'):
            log_and_save(f"  {line}")
        log_and_save("")

        for step in range(self.config.max_env_steps):
            step_start_time = time.time()

            log_and_save(f"\n{'─'*80}")
            log_and_save(f"STEP {step + 1}/{self.config.max_env_steps}")
            log_and_save(f"{'─'*80}")

            actions = {}
            step_info = []

            # Batch generate actions for all agents
            batch_results = self.generate_actions_batch(obs, step, env, model, use_ref_model=use_ref_model)

            for agent_id in range(1, self.config.num_agents + 1):
                # Get agent position (internal coordinates)
                ax, ay_internal = env.agents[agent_id]
                # Transform to display coordinates (y=0 at bottom)
                ay_display = (env.height - 1) - ay_internal

                # Unpack batch results (8 values - no prompts_dict)
                action, log_prob, thinking_text, full_response, action_raw, action_prompt, action_input_ids, action_ids = batch_results[agent_id]
                actions[agent_id] = action

                # For visualization, reconstruct prompts
                thinking_prompt = self.create_thinking_prompt(obs[agent_id], agent_id, step, env)

                if self.config.use_two_stage:
                    prompts = {
                        'stage1': thinking_prompt,
                        'stage2': action_prompt if action_prompt else ''
                    }
                else:
                    prompts = {
                        'stage1': thinking_prompt,
                        'stage2': ''
                    }

                # Store info for display (using display coordinates)
                step_info.append({
                    'agent_id': agent_id,
                    'position': (ax, ay_display),
                    'thinking': thinking_text.strip(),
                    'response': full_response.strip(),
                    'action_raw': action_raw.strip() if action_raw else '',
                    'action': action,
                    'log_prob': log_prob.item(),
                    'prompts': prompts  # Store prompts for debugging
                })

            # Display agent decisions
            log_and_save("\nAgent Decisions:")
            for info in step_info:
                log_and_save(f"\n  Agent {info['agent_id']} at {info['position']}:")

                # Show coordinate info
                coord_info = self._parse_observation_to_coords(obs[info['agent_id']], info['agent_id'], env)
                log_and_save(f"    {coord_info}")

                # Show prompts if available
                if info.get('prompts'):
                    if self.config.use_two_stage:
                        # Two-stage mode: show thinking prompt
                        log_and_save(f"\n    --- STAGE 1 PROMPT (Thinking) ---")
                        for line in info['prompts']['stage1'].split('\n'):
                            log_and_save(f"    {line}")
                        log_and_save(f"    --- END STAGE 1 PROMPT ---")

                        # Show raw prompt with special tokens
                        log_and_save(f"\n    --- RAW STAGE 1 PROMPT (with special tokens) ---")
                        log_and_save(f"    {repr(info['prompts']['stage1'])}")
                        log_and_save(f"    --- END RAW STAGE 1 PROMPT ---")
                    else:
                        # Single-stage mode: show full prompt
                        log_and_save(f"\n    --- PROMPT ---")
                        for line in info['prompts']['stage1'].split('\n'):
                            log_and_save(f"    {line}")
                        log_and_save(f"    --- END PROMPT ---")

                        # Show raw prompt with special tokens
                        log_and_save(f"\n    --- RAW PROMPT (with special tokens) ---")
                        log_and_save(f"    {repr(info['prompts']['stage1'])}")
                        log_and_save(f"    --- END RAW PROMPT ---")

                if self.config.use_two_stage:
                    # Two-stage mode: show thinking output separately
                    log_and_save(f"\n    Thinking output: '{info['thinking']}'")

                    # Show raw thinking response
                    log_and_save(f"\n    --- RAW THINKING RESPONSE ---")
                    log_and_save(f"    {repr(info['thinking'])}")
                    log_and_save(f"    --- END RAW THINKING RESPONSE ---")

                    # Show stage 2 prompt if available
                    if info.get('prompts') and info['prompts']['stage2']:
                        log_and_save(f"\n    --- STAGE 2 PROMPT (Action) ---")
                        for line in info['prompts']['stage2'].split('\n'):
                            log_and_save(f"    {line}")
                        log_and_save(f"    --- END STAGE 2 PROMPT ---")

                        # Show raw stage 2 prompt with special tokens
                        log_and_save(f"\n    --- RAW STAGE 2 PROMPT (with special tokens) ---")
                        log_and_save(f"    {repr(info['prompts']['stage2'])}")
                        log_and_save(f"    --- END RAW STAGE 2 PROMPT ---")

                    if info['action_raw']:
                        log_and_save(f"\n    Action (stage 2 raw): '{info['action_raw']}'")

                        # Show raw action response
                        log_and_save(f"\n    --- RAW ACTION RESPONSE ---")
                        log_and_save(f"    {repr(info['action_raw'])}")
                        log_and_save(f"    --- END RAW ACTION RESPONSE ---")
                else:
                    # Single-stage mode: show full response
                    log_and_save(f"\n    Response: '{info['response']}'")

                    # Show raw response
                    log_and_save(f"\n    --- RAW RESPONSE ---")
                    log_and_save(f"    {repr(info['response'])}")
                    log_and_save(f"    --- END RAW RESPONSE ---")

                log_and_save(f"    Action (parsed): {info['action']}")
                log_and_save(f"    Log Prob: {info['log_prob']:.4f}")

            # Execute step
            obs, rewards, done, info = env.step(actions)

            # Show results
            log_and_save("\n  Step Results:")
            step_reward = sum(rewards.values())
            total_reward += step_reward

            for agent_id in range(1, self.config.num_agents + 1):
                if rewards[agent_id] > 0:
                    log_and_save(f"    Agent {agent_id}: +{rewards[agent_id]:.1f} points!")

            log_and_save(f"    Step reward: {step_reward:.1f}")
            log_and_save(f"    Total reward: {total_reward:.1f}")
            log_and_save(f"    Dirt remaining: {info['dirt_count']}")
            log_and_save(f"    Apples available: {info['apple_count']}")

            log_and_save("\n  Grid After Step:")
            for line in env.render().split('\n'):
                log_and_save(f"    {line}")

            # Log step time
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            log_and_save(f"\n  Step Time: {step_time:.2f}s")

            if done:
                log_and_save(f"\n  Episode ended (max steps reached)")
                break

        # Final summary
        total_rollout_time = time.time() - rollout_start_time
        avg_step_time = np.mean(step_times) if step_times else 0.0

        log_and_save(f"\n{'='*80}")
        log_and_save(f"EPISODE SUMMARY")
        log_and_save(f"{'='*80}")
        log_and_save(f"Total Reward: {total_reward:.2f}")
        log_and_save(f"Steps Taken: {step + 1}")
        log_and_save(f"Final Scores: {info['scores']}")
        log_and_save(f"Dirt Cleaned: {initial_dirt_count - info['dirt_count']}")
        log_and_save(f"\nTiming:")
        log_and_save(f"  Average Step Time: {avg_step_time:.2f}s")
        log_and_save(f"  Total Rollout Time: {total_rollout_time:.2f}s")
        log_and_save(f"{'='*80}\n")

        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write('\n'.join(output_lines))
            logger.info(f"Visualization saved to: {save_to_file}")

        model.train()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO with Plain Text Actions")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--thinking_tokens", type=int, default=256,
                       help="Number of tokens for thinking/reasoning stage")
    parser.add_argument("--use_two_stage", action="store_true", default=True,
                       help="Use two-stage generation (thinking + action)")
    parser.add_argument("--no_two_stage", action="store_false", dest="use_two_stage",
                       help="Disable two-stage generation")
    parser.add_argument("--logprob_mode", type=str, default="action+thinking", choices=["action", "action+thinking"],
                       help="Log probability mode: 'action' (only action tokens) or 'action+thinking' (both thinking and action tokens)")
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "drgrpo"],
                       help="Loss type: 'grpo' (with std normalization) or 'drgrpo' (no std normalization)")
    parser.add_argument("--num_episodes", type=int, default=800)
    parser.add_argument("--episodes_per_update", type=int, default=8)
    parser.add_argument("--episodes_per_gpu", type=int, default=4,
                       help="Episodes per GPU when using multi-GPU (default: 4)")
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--max_env_steps", type=int, default=20)
    parser.add_argument("--eat_reward", type=float, default=1.0,
                       help="Reward for eating an apple (default: 1.0)")
    parser.add_argument("--clean_reward", type=float, default=0.0,
                       help="Reward for cleaning a dirt tile (default: 0.0)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./grpo_text_action_checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--use_accelerate", action="store_true", default=False,
                       help="Use Accelerate for multi-GPU training")

    # Inner epoch optimization (PPO-style)
    parser.add_argument("--num_inner_epochs", type=int, default=4,
                       help="Number of optimization epochs per group (buffer reuse, default: 4)")
    parser.add_argument("--minibatch_size", type=int, default=8,
                       help="Number of trajectories per mini-batch during training (default: 8)")
    parser.add_argument("--samples_per_micro_batch", type=int, default=2,
                       help="Number of (prompt, action) samples per micro-batch for gradient accumulation (default: 2, use 1-2 for A100)")

    # DEPRECATED: micro_batch_size is replaced by minibatch_size
    parser.add_argument("--micro_batch_size", type=int, default=8,
                       help="DEPRECATED - Use --minibatch_size instead")

    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Visualize a rollout/trajectory (skip training)")
    parser.add_argument("--viz_save_file", type=str, default=None,
                       help="Save visualization to file (e.g., rollout.txt)")
    parser.add_argument("--viz_use_ref", action="store_true", default=False,
                       help="Use reference model for visualization")

    # Sanity check mode
    parser.add_argument("--sanity_check", action="store_true", default=False,
                       help="Run sanity check mode: single agent, fewer episodes to verify basic learning")

    # Evaluation arguments
    parser.add_argument("--skip_pre_eval", action="store_true", default=False,
                       help="Skip pre-training evaluation")
    parser.add_argument("--skip_post_eval", action="store_true", default=False,
                       help="Skip post-training evaluation")
    parser.add_argument("--num_eval_episodes", type=int, default=20,
                       help="Number of episodes for evaluation")
    parser.add_argument("--eval_interval", type=int, default=128,
                       help="Evaluate every N training episodes (0 = no mid-training eval, default: 128)")

    # Episode trajectory logging
    parser.add_argument("--log_trajectory", action="store_true", default=True,
                       help="Log one random episode per update to file (default: True)")
    parser.add_argument("--no_log_trajectory", action="store_false", dest="log_trajectory",
                       help="Disable episode trajectory logging")
    parser.add_argument("--trajectory_log_file", type=str, default="episode_trajectories.txt",
                       help="Trajectory log file name (default: episode_trajectories.txt)")

    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use wandb for logging (default: True)")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                       help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="grpo_text_action",
                       help="Wandb project name (default: grpo_text_action)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/team name (default: None)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (default: auto-generated)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Sanity check mode: single agent, fewer episodes to verify basic learning
    if args.sanity_check:
        logger.info("\n=== SANITY CHECK MODE ===")
        logger.info("Forcing num_agents=1 and num_episodes=50 to verify basic 'Clean -> Wait -> Eat' loop")
        args.num_agents = 1
        args.num_episodes = 50
        # Also use a descriptive wandb run name if not set
        if args.wandb_run_name is None:
            args.wandb_run_name = "sanity_check_single_agent"

    config = GRPOConfig(
        model_name=args.model_name,
        thinking_tokens=args.thinking_tokens,
        use_two_stage=args.use_two_stage,
        logprob_mode=args.logprob_mode,
        loss_type=args.loss_type,
        num_episodes=args.num_episodes,
        episodes_per_update=args.episodes_per_update,
        episodes_per_gpu=args.episodes_per_gpu,
        num_agents=args.num_agents,
        max_env_steps=args.max_env_steps,
        eat_reward=args.eat_reward,
        clean_reward=args.clean_reward,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_lora=args.use_lora,
        use_accelerate=args.use_accelerate,
        num_inner_epochs=args.num_inner_epochs,
        minibatch_size=args.minibatch_size,
        samples_per_micro_batch=args.samples_per_micro_batch,
        micro_batch_size=args.micro_batch_size,  # Keep for backward compatibility
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        log_trajectory=args.log_trajectory,
        trajectory_log_file=args.trajectory_log_file,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    trainer = CleanupGameGRPOText(config)

    # Visualization mode - show one rollout and exit
    if args.visualize:
        logger.info("\n=== VISUALIZATION MODE ===")
        logger.info("Running one rollout with step-by-step visualization\n")
        trainer.visualize_rollout(
            use_ref_model=args.viz_use_ref,
            save_to_file=args.viz_save_file
        )
        logger.info("\nVisualization complete. Exiting (use without --visualize to train).")
        return

    # Normal training mode
    # Evaluate before training
    if not args.skip_pre_eval:
        logger.info("\n=== Pre-Training Evaluation ===")
        trainer.evaluate(num_episodes=args.num_eval_episodes)

    # Train
    model = trainer.train()

    # Evaluate after training
    if not args.skip_post_eval:
        logger.info("\n=== Post-Training Evaluation ===")
        trainer.evaluate(num_episodes=args.num_eval_episodes)

    # Finish wandb run
    if config.use_wandb:
        wandb.finish()
        logger.info("Wandb run finished.")


if __name__ == "__main__":
    main()
