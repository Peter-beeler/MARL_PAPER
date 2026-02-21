"""
GRPO Fine-tuning with Compound High-Level Actions (Move Environment)

Uses high-level JSON actions (move_to, clean_at, eat_at, random_explore) from helpers.py.
The model reasons and outputs ONE JSON action call in a single generation pass.
The JSON is parsed and dispatched to the appropriate helper function, which returns
the low-level env action (up/down/left/right/clean/eat/stay).

Key differences from grpo_text_action.py:
- No AllowOnlyActionWords logits processor (JSON output is unconstrained)
- Observation text produced by helpers.get_observation_description() (global window)
- Prompts built inline with global scan info (nearest apple/dirt)
- Action parsed from JSON → helper call → low-level action for env.step()
"""

import os, sys
import json
import re
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

# Import the move environment (parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env_move import CleanupEnvMove, Config as EnvConfigMove

# Import archive helpers and prompt_template (same directory)
sys.path.insert(0, os.path.dirname(__file__))
import helpers
import prompt_template

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
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info(f"[CUDA Memory - {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")


@dataclass
class GRPOConfig:
    """Configuration for GRPO training with compound (high-level) actions."""
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_length: int = 512
    thinking_tokens: int = 100   # tokens for reasoning stage 1
    action_tokens: int = 80      # tokens for JSON action stage 2 (larger than plain text)
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

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
    episodes_per_update: int = 8
    episodes_per_gpu: int = 4
    num_agents: int = 5
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = False

    # GRPO specific
    loss_type: str = "grpo"  # "grpo" or "drgrpo"
    gamma: float = 0.99
    epsilon: float = 0.2
    advantage_normalization: bool = True
    clip_advantage: float = 5.0

    # Inner epoch optimization (PPO-style)
    num_inner_epochs: int = 4
    minibatch_size: int = 2
    micro_batch_size: int = 8   # DEPRECATED
    samples_per_micro_batch: int = 2

    # Environment settings
    max_env_steps: int = 50
    eat_reward: float = 1.0    # Reward for eating an apple (passed to env)
    clean_reward: float = 0.0  # Reward for cleaning a dirt tile (passed to env)

    # Checkpoint settings
    output_dir: str = "./grpo_compound_checkpoints"
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
    mixed_precision: str = "bf16"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "grpo_compound_actions"
    wandb_entity: str = None
    wandb_run_name: str = None


class CleanupGameGRPOCompound:
    """GRPO trainer for cleanup game using compound (high-level JSON) actions."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)

        # Initialize Accelerator for multi-GPU training
        if config.use_accelerate and not (config.use_8bit or config.use_4bit):
            timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
                log_with=None,
                kwargs_handlers=[timeout_kwargs]
            )
            self.device = self.accelerator.device
            logger.info(f"Using Accelerator with {self.accelerator.num_processes} processes")
        else:
            self.accelerator = None
            self.device = config.device
            if config.use_8bit or config.use_4bit:
                logger.info(f"Using quantization (8bit={config.use_8bit}, 4bit={config.use_4bit})")

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Loading model: {config.model_name}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # Helper function names (instead of plain action words)
        self.helper_functions = ['move_to', 'clean_at', 'eat_at', 'random_explore']
        # Low-level action words that helpers return (for distribution logging)
        self.low_level_actions = ['up', 'down', 'left', 'right', 'clean', 'eat', 'stay']

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Helper functions: {self.helper_functions}")
            logger.info(f"Max new tokens: {config.thinking_tokens + config.action_tokens} (thinking={config.thinking_tokens} + action={config.action_tokens})")

        # Load model
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

        base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        if not (config.use_8bit or config.use_4bit) and (self.accelerator is None):
            base_model = base_model.to(self.device)

        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        if config.use_8bit or config.use_4bit:
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model)

        if config.use_lora:
            if config.lora_target_modules is None:
                linear_layers = set()
                for name, module in base_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_layers.add(name.split('.')[-1])

                exclude_names = {'lm_head', 'embed_tokens', 'wte', 'wpe', 'ln', 'norm'}
                target_modules = [n for n in linear_layers if not any(ex in n.lower() for ex in exclude_names)]

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

        self.old_model = None
        self.ref_model = None
        self.ref_on_cpu = False
        self.ref_device = None

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info("GRPO with inner epochs: Old model will be created on first group")

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

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return 0.95 ** ((step - self.config.warmup_steps) / 10)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

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
                "action_tokens": config.action_tokens,
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

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def update_old_model(self):
        """Copy current model weights to old model (θ_old ← θ)."""
        import copy

        if self.old_model is None:
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("Creating old model (first group)...")
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info("  ✓ Old model created and frozen")
        else:
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

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _system_context(self) -> str:
        """Return the game-context sentence with the correct clean-reward description."""
        if self.config.clean_reward > 0.0:
            clean_desc = (
                f"Cleaning dirt gives +{self.config.clean_reward} reward AND enables more apple spawning. "
            )
        else:
            clean_desc = (
                "Cleaning dirt gives no immediate reward but is necessary to enable apple spawning. "
            )
        return (
            f"You are an agent in a cleanup game. Your goal is to maximize points by eating apples (+{self.config.eat_reward} each). "
            + clean_desc
            + "You cannot move diagonally. You have high-level functions to navigate and interact."
        )

    def create_single_stage_prompt(self, obs: str, agent_id: int, step: int, env=None) -> str:
        """Single-stage prompt: think step by step, then output ONE JSON action."""
        if env is not None:
            obs_text = helpers.get_observation_description(env, agent_id)
            nearest_apple = helpers.find_nearest_apple(env, agent_id)
            nearest_dirt = helpers.find_nearest_dirt(env, agent_id)

            strategy_info = []
            if nearest_apple['found']:
                strategy_info.append(
                    f"- Nearest Apple: at ({nearest_apple['coord_x']}, {nearest_apple['coord_y']}), distance {nearest_apple['distance']}."
                )
            else:
                strategy_info.append("- Nearest Apple: None found.")
            if nearest_dirt['found']:
                strategy_info.append(
                    f"- Nearest Dirt: at ({nearest_dirt['coord_x']}, {nearest_dirt['coord_y']}), distance {nearest_dirt['distance']}."
                )
            else:
                strategy_info.append("- Nearest Dirt: None found.")
            strategy_str = "\n".join(strategy_info)
        else:
            obs_text = obs
            strategy_str = ""

        if self.config.clean_reward > 0.0:
            decision_clean = f"2. Should you clean dirt for +{self.config.clean_reward} reward AND to ensure future apples spawn?"
        else:
            decision_clean = "2. Should you clean dirt to ensure future apples spawn (helping the group)?"

        system_content = self._system_context() + "\n\n" + prompt_template.get_action_api()

        scan_section = f"\n\n### GLOBAL SCAN\n{strategy_str}" if strategy_str else ""

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
            {"role": "user",   "content": user_content},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_and_execute_action(self, response: str, env, agent_id: int) -> str:
        """Parse JSON from model response, call the appropriate helper, return low-level env action.

        Args:
            response: Raw text output from the model.
            env: Current environment instance.
            agent_id: Agent ID.

        Returns:
            Low-level action string accepted by env.step(): up/down/left/right/clean/eat/stay.
        """
        response = response.strip()
        logger.debug(f"Raw model action response: {response}")

        # Step 1: strip markdown code fence if present (```json ... ``` or ``` ... ```)
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Step 2: balanced-brace extraction — find first '{' and walk to matching '}'
            start = response.find('{')
            if start == -1:
                logger.debug(f"No JSON found in response: '{response[:80]}', falling back to random_explore")
                action, _ = helpers.random_explore(env, agent_id)
                return action
            depth = 0
            end = -1
            for i in range(start, len(response)):
                if response[i] == '{':
                    depth += 1
                elif response[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                logger.debug(f"Unbalanced braces in response: '{response[:80]}', falling back to random_explore")
                action, _ = helpers.random_explore(env, agent_id)
                return action
            json_str = response[start:end + 1]

        try:
            data = json.loads(json_str)
            action_name = data.get('action', '')
            args = data.get('args', {})
            if action_name == 'move_to':
                action, _ = helpers.move_to(
                    env, agent_id,
                    int(args.get('coord_x', 0)),
                    int(args.get('coord_y', 0))
                )
            elif action_name == 'clean_at':
                action, _ = helpers.clean_at(
                    env, agent_id,
                    int(args.get('coord_x', 0)),
                    int(args.get('coord_y', 0))
                )
            elif action_name == 'eat_at':
                action, _ = helpers.eat_at(
                    env, agent_id,
                    int(args.get('coord_x', 0)),
                    int(args.get('coord_y', 0))
                )
            elif action_name == 'random_explore':
                action, _ = helpers.random_explore(env, agent_id)
            else:
                logger.debug(f"Unknown action name: '{action_name}', falling back to random_explore")
                action, _ = helpers.random_explore(env, agent_id)

            return action

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Failed to parse JSON action: {e}, falling back to random_explore")
            action, _ = helpers.random_explore(env, agent_id)
            return action

    # ------------------------------------------------------------------
    # Log probability computation
    # ------------------------------------------------------------------

    def compute_batch_sequence_log_prob(
        self,
        model,
        prompt_input_ids_list: List[torch.Tensor],
        generated_ids_list: List[torch.Tensor],
        device: torch.device,
        pad_token_id: int = None,
        need_grad: bool = False
    ) -> torch.Tensor:

        if pad_token_id is None:
            actual_model = model.module if hasattr(model, 'module') else model
            if hasattr(actual_model.config, 'pad_token_id'):
                pad_token_id = actual_model.config.pad_token_id

        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided or available in model config")

        full_sequences = []
        prompt_lens = []
        gen_lens = []

        for prompt_ids, gen_ids in zip(prompt_input_ids_list, generated_ids_list):
            p_ids = prompt_ids.to(device).view(-1)
            p_ids = p_ids[p_ids != pad_token_id]

            g_ids = gen_ids.to(device).view(-1)
            g_ids = g_ids[g_ids != pad_token_id]

            prompt_lens.append(len(p_ids))
            gen_lens.append(len(g_ids))
            full_sequences.append(torch.cat([p_ids, g_ids]))

        batch_input_ids = pad_sequence(full_sequences, batch_first=True, padding_value=pad_token_id)
        attention_mask = (batch_input_ids != pad_token_id).long()

        def forward_pass():
            outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
            return outputs.logits

        if need_grad:
            logits = forward_pass()
        else:
            with torch.no_grad():
                logits = forward_pass()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        token_log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        action_mask = torch.zeros_like(token_log_probs)
        for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
            start_idx = max(0, p_len - 1)
            end_idx = start_idx + g_len
            if g_len > 0:
                action_mask[i, start_idx:end_idx] = 1.0

        masked_log_probs = token_log_probs * action_mask
        seq_log_prob_sum = masked_log_probs.sum(dim=1)

        return seq_log_prob_sum

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def generate_action(
        self, obs: str, agent_id: int, step: int, env, model
    ) -> Tuple:
        """Generate action for one agent (single-stage: think + JSON in one pass).

        Returns:
            (action, log_prob, response, input_ids, generated_ids)
        """
        is_ref_model = (model is self.ref_model)
        target_device = self.ref_device if (is_ref_model and hasattr(self, 'ref_device')) else self.device
        gen_model = model.module if (self.accelerator is not None and hasattr(model, 'module')) else model

        prompt = self.create_single_stage_prompt(obs, agent_id, step, env)
        inputs = self.tokenizer(
            prompt,
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
                    max_new_tokens=self.config.thinking_tokens + self.config.action_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Generation failed: {e}, using default action")
            return "stay", torch.tensor(0.0, device=target_device), "", None, None

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        action = self.parse_and_execute_action(response, env, agent_id)

        try:
            log_prob_model = self.old_model if self.old_model is not None else gen_model
            log_probs = self.compute_batch_sequence_log_prob(
                model=log_prob_model,
                prompt_input_ids_list=[inputs.input_ids],
                generated_ids_list=[generated_ids],
                device=target_device,
                pad_token_id=self.tokenizer.pad_token_id,
                need_grad=False
            )
            log_prob = log_probs[0]
        except Exception as e:
            logger.warning(f"Log prob calculation failed: {e}")
            log_prob = torch.tensor(-10.0, device=target_device)

        return action, log_prob, response, inputs.input_ids, generated_ids

    def generate_actions_batch(
        self, obs_dict: Dict[int, str], step: int, env, model, use_ref_model: bool = False
    ):
        """Generate actions for all agents in a batch.

        Returns dict: agent_id → (action, log_prob, response, input_ids, generated_ids)
        """
        target_device = self.ref_device if (use_ref_model and hasattr(self, 'ref_device')) else self.device
        gen_model = model.module if (self.accelerator is not None and hasattr(model, 'module')) else model

        agent_ids = sorted(obs_dict.keys())
        num_agents = len(agent_ids)

        # Build prompts for all agents
        prompts = []
        for agent_id in agent_ids:
            prompts.append(self.create_single_stage_prompt(obs_dict[agent_id], agent_id, step, env))

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True
        ).to(target_device)

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=self.config.thinking_tokens + self.config.action_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch generation failed: {e}, falling back to sequential")
            results = {}
            for agent_id in agent_ids:
                results[agent_id] = self.generate_action(obs_dict[agent_id], agent_id, step, env, model)
            return results

        responses = []
        actions = []
        generated_ids_list = []
        input_ids_list = []

        for i, agent_id in enumerate(agent_ids):
            gen_ids = outputs.sequences[i][inputs.input_ids[i].shape[0]:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            action = self.parse_and_execute_action(response, env, agent_id)
            responses.append(response)
            actions.append(action)
            generated_ids_list.append(gen_ids)
            input_ids_list.append(inputs.input_ids[i])

        # Batch log prob
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

        results = {}
        for i, agent_id in enumerate(agent_ids):
            results[agent_id] = (
                actions[i],
                log_probs[i],
                responses[i],
                input_ids_list[i],
                generated_ids_list[i]
            )
        return results

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def run_episode(
        self,
        use_ref_model: bool = False,
        log_samples: bool = False,
        initial_state: Optional[Dict] = None
    ) -> Dict:
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
            "action_prompts": [],
            "action_texts": [],
            "rewards": [],
            "log_probs": [],
            "agent_ids": [],
            "observations": [],
            "action_input_ids": [],
            "action_ids": [],
        }

        if use_ref_model and self.ref_model is None:
            logger.warning("Reference model requested but not available. Using current policy.")
            model = self.model
        else:
            model = self.ref_model if use_ref_model else self.model

        total_reward = 0

        for step in range(self.config.max_env_steps):
            actions = {}
            batch_results = self.generate_actions_batch(obs, step, env, model, use_ref_model=use_ref_model)

            for agent_id in range(1, self.config.num_agents + 1):
                action, log_prob, response, action_input_ids, action_ids = batch_results[agent_id]

                actions[agent_id] = action

                # Reconstruct prompt for storage (env state not yet stepped)
                stored_prompt = self.create_single_stage_prompt(obs[agent_id], agent_id, step, env)
                trajectory["prompts"].append(stored_prompt)
                trajectory["actions"].append(action)
                trajectory["responses"].append(response)
                trajectory["action_prompts"].append("")
                trajectory["action_texts"].append(response)
                trajectory["log_probs"].append(log_prob.detach().item())
                trajectory["agent_ids"].append(agent_id)
                trajectory["observations"].append(obs[agent_id])
                trajectory["action_input_ids"].append(action_input_ids)
                trajectory["action_ids"].append(action_ids)

                if log_samples and step == 0 and agent_id == 1:
                    obs_text = helpers.get_observation_description(env, agent_id)
                    logger.info(f"\n  Sample generation (compound):")
                    logger.info(f"    Obs: {obs_text}")
                    logger.info(f"    Response (thinking+JSON): '{response[:200]}'")
                    logger.info(f"    → Low-level action: {action}")

            obs, rewards, done, info = env.step(actions)

            for agent_id in range(1, self.config.num_agents + 1):
                trajectory["rewards"].append(rewards[agent_id])
                total_reward += rewards[agent_id]

            if done:
                break

        trajectory["total_reward"] = total_reward
        trajectory["final_scores"] = info["scores"]
        trajectory["steps"] = step + 1
        trajectory["rollout_time"] = time.time() - start_time

        if log_samples:
            log_cuda_memory("After episode rollout")

        return trajectory

    def log_episode_to_file(self, trajectory: Dict, group_num: int, episode_idx: int):
        """Log a single episode trajectory to file."""
        if not self.config.log_trajectory:
            return
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
                    f.write(f"    Thinking: {thinking_part}\n")
                    f.write(f"    JSON action: {action_text[:200]}\n")
                    f.write(f"    Low-level action: {action}\n")
                    f.write(f"    Reward: {reward:.2f}\n")

                f.write("\n")

            f.write(f"\n[Episode Summary]\n")
            f.write(f"  Total Reward: {trajectory['total_reward']:.2f}\n")
            f.write(f"  Rollout Time: {trajectory.get('rollout_time', 0):.2f}s\n\n")

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        """Compute GRPO/DrGRPO advantages with optional multi-GPU global normalization."""
        returns = [traj["total_reward"] for traj in trajectories]
        use_std_norm = (self.config.loss_type.lower() == "grpo")

        if self.config.advantage_normalization and len(returns) > 1:
            if self.accelerator is not None:
                model_device = next(self.model.parameters()).device
                local_returns = torch.tensor(returns, dtype=torch.float32, device=model_device)
                all_returns = self.accelerator.gather(local_returns)
                global_mean = all_returns.mean().item()

                if use_std_norm:
                    global_std = all_returns.std().item() + 1e-8
                    normalized_returns = [(r - global_mean) / global_std for r in returns]
                else:
                    normalized_returns = [(r - global_mean) for r in returns]
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
            np.clip(adv, -self.config.clip_advantage, self.config.clip_advantage)
            for adv in normalized_returns
        ]

        for traj, advantage in zip(trajectories, normalized_returns):
            traj["advantage"] = advantage
            traj["advantages"] = [advantage] * len(traj["rewards"])

        return trajectories

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def create_minibatch_iterator(self, trajectories: List[Dict], minibatch_size: int):
        """Yield shuffled mini-batches of trajectories."""
        import random
        indices = list(range(len(trajectories)))
        random.shuffle(indices)
        for start_idx in range(0, len(indices), minibatch_size):
            end_idx = min(start_idx + minibatch_size, len(indices))
            yield [trajectories[i] for i in indices[start_idx:end_idx]]

    def flatten_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """Flatten trajectories into individual (prompt_ids, action_ids, advantage, old_log_prob) samples."""
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
        self,
        samples: List[Dict],
        device
    ) -> Tuple[torch.Tensor, float, int]:
        """Compute GRPO loss on a micro-batch of flattened samples."""
        if len(samples) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

        prompt_ids_list = [s["prompt_ids"] for s in samples]
        action_ids_list = [s["action_ids"] for s in samples]
        advantages_list = [s["advantage"] for s in samples]
        old_log_probs_list = [s["old_log_prob"] for s in samples]

        new_log_probs = self.compute_batch_sequence_log_prob(
            model=self.model,
            prompt_input_ids_list=prompt_ids_list,
            generated_ids_list=action_ids_list,
            device=device,
            pad_token_id=self.tokenizer.pad_token_id,
            need_grad=True
        )

        old_log_probs = torch.tensor(old_log_probs_list, device=device, dtype=torch.float32)
        advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)

        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        epsilon = self.config.epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        loss_unclipped = ratio * advantages
        loss_clipped = clipped_ratio * advantages
        policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()

        with torch.no_grad():
            clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
            clip_fraction = clipped_mask.float().mean().item()

        return policy_loss, clip_fraction, len(samples)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop."""
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)

            logger.info("\n" + "="*70)
            logger.info("GRPO Training with Compound High-Level Actions (Single-Stage)")
            logger.info("="*70)
            logger.info(f"\n[Model] {self.config.model_name}")
            logger.info(f"[Loss Type] {self.config.loss_type.upper()}")
            logger.info(f"[Environment] Move (directional actions via helpers)")
            logger.info(f"[Helpers] {self.helper_functions}")
            logger.info(f"[Max New Tokens] {self.config.thinking_tokens + self.config.action_tokens} (thinking={self.config.thinking_tokens} + action={self.config.action_tokens})")
            logger.info(f"[Agents] {self.config.num_agents}")
            logger.info(f"[Episodes] {self.config.num_episodes}")
            if self.accelerator is not None:
                eps_per_group = self.config.episodes_per_gpu * self.accelerator.num_processes
                logger.info(f"[Multi-GPU] {self.accelerator.num_processes} GPUs × {self.config.episodes_per_gpu} eps/GPU = {eps_per_group} per group")
            logger.info("="*70 + "\n")

        episode = 0
        best_reward = float('-inf')
        group_num = 0

        while episode < self.config.num_episodes:
            group_start_episode = episode

            if self.accelerator is None or self.accelerator.is_main_process:
                if self.accelerator is not None:
                    expected_episodes = self.config.episodes_per_gpu * self.accelerator.num_processes
                else:
                    expected_episodes = self.config.episodes_per_update
                logger.info(f"\n[Group {group_num}] Episodes {episode}-{episode + expected_episodes - 1} (expected)")

            self.update_old_model()

            if self.accelerator is not None:
                trajectories = []
                for i in range(self.config.episodes_per_gpu):
                    try:
                        log_samples = (i == 0 and group_num % self.config.log_interval == 0 and self.accelerator.is_main_process)
                        traj = self.run_episode(use_ref_model=False, log_samples=log_samples)
                        trajectories.append(traj)
                        if not log_samples:
                            logger.info(f"  [GPU{self.accelerator.process_index}] Ep{i}: R={traj['total_reward']:.2f}, Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s")
                    except Exception as e:
                        logger.error(f"  [GPU{self.accelerator.process_index}] Episode {i} failed: {e}", exc_info=True)
                        continue
            else:
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

            # Update episode counter
            if self.accelerator is not None:
                local_count = len(trajectories)
                count_tensor = torch.tensor([local_count], dtype=torch.long, device=next(self.model.parameters()).device)
                all_counts = self.accelerator.gather(count_tensor)
                total_collected = all_counts.sum().item()
                episode += total_collected
                if self.accelerator.is_main_process:
                    logger.info(f"  ✓ Collected {total_collected} episodes ({[c.item() for c in all_counts]} per GPU)")
            else:
                episode += len(trajectories)

            if len(trajectories) > 0 and self.config.log_trajectory:
                if self.accelerator is None or self.accelerator.is_main_process:
                    import random as _random
                    random_idx = _random.randint(0, len(trajectories) - 1)
                    self.log_episode_to_file(trajectories[random_idx], group_num, random_idx)
                    logger.info(f"  📝 Logged episode {random_idx} trajectory")

            if len(trajectories) == 0:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.warning("No valid trajectories collected, skipping update")
                group_num += 1
                continue

            trajectories = self.compute_advantages(trajectories)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            # Initialize here so all ranks (including non-main) always have a value
            avg_reward = 0.0
            std_reward = 0.0

            # Gather stats for logging
            if self.accelerator is not None:
                local_rewards = [t["total_reward"] for t in trajectories]
                local_steps = [t["steps"] for t in trajectories]
                local_times = [t["rollout_time"] for t in trajectories]

                model_device = next(self.model.parameters()).device
                all_rewards = self.accelerator.gather(torch.tensor(local_rewards, dtype=torch.float32, device=model_device))
                all_steps = self.accelerator.gather(torch.tensor(local_steps, dtype=torch.float32, device=model_device))
                all_times = self.accelerator.gather(torch.tensor(local_times, dtype=torch.float32, device=model_device))

                if self.accelerator.is_main_process:
                    avg_reward = all_rewards.mean().item()
                    std_reward = all_rewards.std().item()
                    max_reward = all_rewards.max().item()
                    min_reward = all_rewards.min().item()
                    avg_steps = all_steps.mean().item()
                    avg_rollout_time = all_times.mean().item()
                    total_rollout_time = all_times.sum().item()

                    self.episode_rewards.append(avg_reward)
                    self.episode_steps.append(avg_steps)
                    logger.info(f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], Steps: {avg_steps:.1f}")

                    if self.config.use_wandb:
                        wandb.log({
                            "episode": episode, "reward/mean": avg_reward, "reward/std": std_reward,
                            "reward/min": min_reward, "reward/max": max_reward,
                            "episode_steps": avg_steps, "rollout_time/avg": avg_rollout_time,
                        }, step=episode)

                    # Low-level action distribution
                    all_actions = [a for traj in trajectories for a in traj["actions"]]
                    action_counts = {}
                    for a in all_actions:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    if all_actions:
                        total_a = len(all_actions)
                        dist_str = " | ".join([f"{a}:{action_counts.get(a, 0)*100//total_a}%" for a in self.low_level_actions])
                        logger.info(f"  Low-level actions (GPU0 sample): {dist_str}")
            else:
                avg_reward = np.mean([t["total_reward"] for t in trajectories])
                std_reward = np.std([t["total_reward"] for t in trajectories])
                max_reward = np.max([t["total_reward"] for t in trajectories])
                min_reward = np.min([t["total_reward"] for t in trajectories])
                avg_steps = np.mean([t["steps"] for t in trajectories])
                avg_rollout_time = np.mean([t["rollout_time"] for t in trajectories])
                total_rollout_time = sum([t["rollout_time"] for t in trajectories])

                self.episode_rewards.append(avg_reward)
                self.episode_steps.append(avg_steps)
                logger.info(f"  Reward: {avg_reward:.2f}±{std_reward:.2f} [{min_reward:.2f}, {max_reward:.2f}], Steps: {avg_steps:.1f}")
                logger.info(f"  Rollout Time: avg={avg_rollout_time:.2f}s, total={total_rollout_time:.2f}s")

                if self.config.use_wandb:
                    wandb.log({
                        "episode": episode, "reward/mean": avg_reward, "reward/std": std_reward,
                        "reward/min": min_reward, "reward/max": max_reward,
                        "episode_steps": avg_steps,
                    }, step=episode)

                all_actions = [a for traj in trajectories for a in traj["actions"]]
                action_counts = {}
                for a in all_actions:
                    action_counts[a] = action_counts.get(a, 0) + 1
                if all_actions:
                    total_a = len(all_actions)
                    dist_str = " | ".join([f"{a}:{action_counts.get(a, 0)*100//total_a}%" for a in self.low_level_actions])
                    logger.info(f"  Low-level actions: {dist_str}")

            # Inner optimization epochs
            try:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info(f"\n  Inner Optimization ({self.config.num_inner_epochs} epochs, minibatch={self.config.minibatch_size}):")

                epoch_losses = []
                epoch_clip_fracs = []
                final_grad_norm = 0.0

                for inner_epoch in range(self.config.num_inner_epochs):
                    minibatch_iterator = self.create_minibatch_iterator(trajectories, self.config.minibatch_size)
                    epoch_loss_sum = 0.0
                    epoch_clip_sum = 0.0
                    epoch_samples = 0
                    num_minibatches = 0

                    for minibatch_idx, minibatch in enumerate(minibatch_iterator):
                        self.optimizer.zero_grad()
                        model_device = next(self.model.parameters()).device if self.accelerator is not None else self.device

                        all_samples = self.flatten_trajectories(minibatch)
                        total_samples = len(all_samples)
                        if total_samples == 0:
                            continue

                        micro_batch_size = self.config.micro_batch_size
                        num_chunks = (total_samples + micro_batch_size - 1) // micro_batch_size

                        total_loss_sum = 0.0
                        total_clip_sum = 0.0
                        total_n_samples = 0

                        for i in range(0, total_samples, micro_batch_size):
                            micro_batch = all_samples[i: i + micro_batch_size]
                            loss, clip_fraction, n_samples = self.compute_loss_on_samples(micro_batch, model_device)

                            if n_samples > 0:
                                normalized_loss = loss / num_chunks
                                normalized_loss.backward()
                                total_loss_sum += loss.item() * n_samples
                                total_clip_sum += clip_fraction * n_samples
                                total_n_samples += n_samples

                        if total_n_samples > 0:
                            if self.accelerator is not None:
                                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            else:
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                            self.optimizer.step()

                            avg_loss = total_loss_sum / total_n_samples
                            avg_clip = total_clip_sum / total_n_samples
                            epoch_loss_sum += total_loss_sum
                            epoch_clip_sum += total_clip_sum
                            epoch_samples += total_n_samples
                            final_grad_norm = grad_norm
                            num_minibatches += 1

                            if inner_epoch == 0 and minibatch_idx == 0:
                                if self.accelerator is None or self.accelerator.is_main_process:
                                    logger.info(f"    [Epoch 1/{self.config.num_inner_epochs}, Batch 1] Loss={avg_loss:.4f}, ClipFrac={avg_clip:.3f}")

                    if epoch_samples > 0:
                        epoch_avg_loss = epoch_loss_sum / epoch_samples
                        epoch_avg_clip = epoch_clip_sum / epoch_samples
                        epoch_losses.append(epoch_avg_loss)
                        epoch_clip_fracs.append(epoch_avg_clip)

                        if inner_epoch == self.config.num_inner_epochs - 1:
                            if self.accelerator is None or self.accelerator.is_main_process:
                                logger.info(f"    [Epoch {inner_epoch+1}/{self.config.num_inner_epochs}] Avg Loss={epoch_avg_loss:.4f}, ClipFrac={epoch_avg_clip:.3f}")

                if self.training_step >= 0:
                    self.scheduler.step()
                self.training_step += 1

                if len(epoch_losses) > 0:
                    final_loss = epoch_losses[-1]
                    final_clip_frac = epoch_clip_fracs[-1]
                    avg_loss_all_epochs = sum(epoch_losses) / len(epoch_losses)

                    if self.accelerator is None or self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(f"  Final: Loss={final_loss:.4f} (avg={avg_loss_all_epochs:.4f}), ClipFrac={final_clip_frac:.3f}, GradNorm={final_grad_norm:.4f}, LR={current_lr:.2e}")

                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": final_loss,
                                "train/clip_fraction": final_clip_frac,
                                "train/grad_norm": final_grad_norm,
                                "train/learning_rate": current_lr,
                                "train/training_step": self.training_step,
                            }, step=episode)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.error(f"Training step failed: {e}", exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                group_num += 1
                continue

            if avg_reward > best_reward:
                best_reward = avg_reward

            # Checkpoint
            if (self.accelerator is None or self.accelerator.is_main_process) and \
               (episode % self.config.save_steps == 0 or episode >= self.config.num_episodes):

                if avg_reward >= best_reward:
                    checkpoint_path = os.path.join(self.config.output_dir, "best_model")
                    try:
                        if self.config.use_lora:
                            unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
                            unwrapped.save_pretrained(checkpoint_path)
                        else:
                            m = self.model.module if hasattr(self.model, 'module') else self.model
                            m.save_pretrained(checkpoint_path)
                        self.tokenizer.save_pretrained(checkpoint_path)
                        logger.info(f"  Saved best model (R={best_reward:.2f})")
                    except Exception as e:
                        logger.warning(f"  Failed to save model: {e}")

                stats = {
                    "episode": episode, "group": group_num,
                    "rewards": self.episode_rewards, "steps": self.episode_steps,
                    "best_reward": best_reward, "training_step": self.training_step
                }
                with open(os.path.join(self.config.output_dir, "training_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2)

            # Mid-training evaluation
            if self.config.eval_interval > 0 and episode % self.config.eval_interval == 0 and episode < self.config.num_episodes:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info(f"\n=== Mid-Training Evaluation (Episode {episode}) ===")
                self.evaluate(num_episodes=self.config.num_eval_episodes, current_episode=episode)
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.info("=== Resuming Training ===\n")

            group_num += 1

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"\n=== Training Complete ===")
            logger.info(f"Best reward: {best_reward:.2f}, Total groups: {group_num}\n")

        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 20, current_episode: int = None):
        """Evaluate current policy on fixed initial states."""
        eval_start_time = time.time()
        actual_num_episodes = min(num_episodes, len(self.eval_states))

        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"\n=== Evaluation ({actual_num_episodes} episodes) ===")

        self.model.eval()

        if self.accelerator is not None:
            num_processes = self.accelerator.num_processes
            process_index = self.accelerator.process_index
            local_episode_indices = list(range(process_index, actual_num_episodes, num_processes))
        else:
            process_index = 0
            local_episode_indices = list(range(actual_num_episodes))

        local_rewards = []
        local_episode_times = []

        for i in local_episode_indices:
            try:
                initial_state = self.eval_states[i]
                traj = self.run_episode(use_ref_model=False, log_samples=False, initial_state=initial_state)
                local_rewards.append(traj["total_reward"])
                local_episode_times.append(traj["rollout_time"])
                logger.info(f"  GPU{process_index} Ep{i+1}: R={traj['total_reward']:.2f}, Steps={traj['steps']}, Time={traj['rollout_time']:.2f}s")
                del traj
                if torch.cuda.is_available() and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Evaluation episode {i+1} failed: {e}")
                continue

        if self.accelerator is not None:
            model_device = next(self.model.parameters()).device
            local_len = len(local_rewards)
            max_len_tensor = torch.tensor([local_len], dtype=torch.long, device=model_device)
            all_lens = self.accelerator.gather(max_len_tensor)

            max_len = all_lens.max().item()
            local_rewards_padded = local_rewards + [0.0] * (max_len - len(local_rewards))
            local_times_padded = local_episode_times + [0.0] * (max_len - len(local_episode_times))

            all_rewards = self.accelerator.gather(torch.tensor(local_rewards_padded, dtype=torch.float32, device=model_device))
            all_times = self.accelerator.gather(torch.tensor(local_times_padded, dtype=torch.float32, device=model_device))

            if self.accelerator.is_main_process:
                rewards = []
                episode_times = []
                for proc_idx in range(self.accelerator.num_processes):
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
            self.model.train()
            return 0.0, 0.0

        if self.accelerator is None or self.accelerator.is_main_process:
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            total_eval_time = time.time() - eval_start_time
            avg_episode_time = np.mean(episode_times) if episode_times else 0.0

            logger.info(f"\nReward: {avg_reward:.2f}±{std_reward:.2f} [{min(rewards):.2f}, {max(rewards):.2f}]")
            logger.info(f"Evaluation Time: avg={avg_episode_time:.2f}s/episode, total={total_eval_time:.2f}s")

            if self.config.use_wandb and current_episode is not None:
                wandb.log({
                    "eval/reward_mean": avg_reward, "eval/reward_std": std_reward,
                    "eval/reward_min": min(rewards), "eval/reward_max": max(rewards),
                    "eval/episode_time": avg_episode_time, "eval/total_time": total_eval_time,
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
        logger.info("TRAJECTORY VISUALIZATION (COMPOUND JSON ACTIONS)")
        logger.info("="*80)

        # Run one episode with detailed logging
        env = CleanupEnvMove(self.env_config)
        obs = env.reset()
        initial_dirt_count = sum(row.count('#') for row in env.items)

        # Select model
        if use_ref_model and self.ref_model is None:
            logger.warning("Reference model requested but not available. Using current policy.")
            model = self.model
        else:
            model = self.ref_model if use_ref_model else self.model
        model.eval()

        total_reward = 0
        output_lines = []
        step_times = []

        def log_and_save(line):
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
                # Use internal coordinates to match helpers' coordinate system
                ax, ay = env.agents[agent_id]

                # Unpack batch results
                action, log_prob, response, action_input_ids, action_ids = batch_results[agent_id]
                actions[agent_id] = action

                # Reconstruct prompt for display
                display_prompt = self.create_single_stage_prompt(obs[agent_id], agent_id, step, env)

                # Natural-language observation using helpers (uses same internal coords)
                obs_nl = helpers.get_observation_description(env, agent_id)

                step_info.append({
                    'agent_id': agent_id,
                    'position': (ax, ay),
                    'obs_nl': obs_nl,
                    'prompt': display_prompt,
                    'response': response.strip(),
                    'action': action,
                    'log_prob': log_prob.item(),
                })

            # Display agent decisions
            log_and_save("\nAgent Decisions:")
            for info in step_info:
                log_and_save(f"\n  Agent {info['agent_id']} at {info['position']}:")
                log_and_save(f"    Observation: {info['obs_nl']}")

                log_and_save(f"\n    --- PROMPT ---")
                for line in info['prompt'].split('\n'):
                    log_and_save(f"    {line}")
                log_and_save(f"    --- END PROMPT ---")

                log_and_save(f"\n    --- RAW PROMPT ---")
                log_and_save(f"    {repr(info['prompt'])}")
                log_and_save(f"    --- END RAW PROMPT ---")

                log_and_save(f"\n    Response (thinking + JSON):")
                for line in info['response'].split('\n'):
                    log_and_save(f"    {line}")

                log_and_save(f"\n    --- RAW RESPONSE ---")
                log_and_save(f"    {repr(info['response'])}")
                log_and_save(f"    --- END RAW RESPONSE ---")

                log_and_save(f"    Action (parsed from JSON): {info['action']}")
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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO with Compound High-Level Actions (helpers.py)")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--thinking_tokens", type=int, default=100)
    parser.add_argument("--action_tokens", type=int, default=80)
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "drgrpo"])
    parser.add_argument("--num_episodes", type=int, default=800)
    parser.add_argument("--episodes_per_update", type=int, default=8)
    parser.add_argument("--episodes_per_gpu", type=int, default=4)
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--max_env_steps", type=int, default=20)
    parser.add_argument("--eat_reward", type=float, default=1.0,
                        help="Reward for eating an apple (default: 1.0)")
    parser.add_argument("--clean_reward", type=float, default=0.0,
                        help="Shaped reward for successfully cleaning a dirt tile (default: 0.0 = disabled)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./grpo_compound_checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--use_accelerate", action="store_true", default=False)
    parser.add_argument("--num_inner_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--samples_per_micro_batch", type=int, default=2)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--eval_interval", type=int, default=128)
    parser.add_argument("--num_eval_episodes", type=int, default=20)
    parser.add_argument("--log_trajectory", action="store_true", default=True)
    parser.add_argument("--no_log_trajectory", action="store_false", dest="log_trajectory")
    parser.add_argument("--trajectory_log_file", type=str, default="episode_trajectories.txt")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default="grpo_compound_actions")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--skip_pre_eval", action="store_true", default=False)
    parser.add_argument("--skip_post_eval", action="store_true", default=False)

    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Visualize a rollout/trajectory step-by-step (skip training)")
    parser.add_argument("--viz_save_file", type=str, default=None,
                       help="Save visualization to file (e.g., rollout.txt)")
    parser.add_argument("--viz_use_ref", action="store_true", default=False,
                       help="Use reference model for visualization")

    return parser.parse_args()


def main():
    args = parse_args()

    config = GRPOConfig(
        model_name=args.model_name,
        thinking_tokens=args.thinking_tokens,
        action_tokens=args.action_tokens,
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
        micro_batch_size=args.micro_batch_size,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        log_trajectory=args.log_trajectory,
        trajectory_log_file=args.trajectory_log_file,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    trainer = CleanupGameGRPOCompound(config)

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

    if not args.skip_pre_eval:
        logger.info("\n=== Pre-Training Evaluation ===")
        trainer.evaluate(num_episodes=args.num_eval_episodes)

    model = trainer.train()

    if not args.skip_post_eval:
        logger.info("\n=== Post-Training Evaluation ===")
        trainer.evaluate(num_episodes=args.num_eval_episodes)

    if config.use_wandb:
        wandb.finish()
        logger.info("Wandb run finished.")


if __name__ == "__main__":
    main()
