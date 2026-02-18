"""
grpo_template.py — GRPO training template for LLM agents in game environments.

HOW TO USE THIS TEMPLATE
========================
This file is a self-contained reference template. To adapt it to a new game:

  1. Search for every block marked  # === CUSTOMIZE ===
     and replace the placeholder code with your environment's logic.

  2. Do NOT touch blocks marked  # === GENERIC — DO NOT MODIFY ===
     Those implement the core GRPO algorithm and are environment-agnostic.

  3. Import your config:
       from grpo_config import GRPOConfig

ALGORITHM OVERVIEW
==================
GRPO (Group Relative Policy Optimization) is an RL fine-tuning method for LLMs.

Each training "group" consists of:
  1. [ROLLOUT]     Run K episodes with the current policy (π_old frozen copy).
                   Store (prompt_ids, action_ids, log_prob_old, reward) per step.
  2. [ADVANTAGES]  Normalize rewards across the group:
                   GRPO  → A = (R - mean) / std
                   DrGRPO→ A = (R - mean)
  3. [INNER EPOCHS] For E epochs, sample mini-batches from the buffer and update:
                   Loss = -E[min(ratio · A, clip(ratio, 1±ε) · A)]
                   where ratio = exp(log π_new − log π_old)

Two-stage generation:
  Stage 1 – Model produces free-form "thinking" text (N tokens).
  Stage 2 – Model reads its own thinking and outputs a single action word.
  The log-prob used for training can cover both stages ("action+thinking")
  or just the action stage ("action").

Required external packages:
  pip install torch transformers peft accelerate wandb numpy
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import copy
import random
import time
import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import wandb

# === CUSTOMIZE ===
# Import your environment and its config here.
# Your environment must expose the following interface (see run_episode below):
#   env = YourEnv(env_cfg)          # constructor
#   obs = env.reset()               # returns obs dict {agent_id: obs_str, ...}
#   obs, rewards, done, info = env.step(actions)  # actions: {agent_id: action_str}
#   env.get_state() / env.set_state(state)        # optional, for fixed eval states
#
# Example:
#   from my_game_env import MyGameEnv, MyEnvConfig
# ==================

from grpo_config import GRPOConfig

warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# UTILITY — CUDA MEMORY LOGGING
# ============================================================

def log_cuda_memory(stage: str, device: int = 0):
    """Log CUDA memory usage at a given stage (GB)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        logger.info(
            f"[CUDA Memory — {stage}] "
            f"Alloc: {alloc:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {peak:.2f} GB"
        )


# ============================================================
# LOGITS PROCESSOR — restrict output to action vocabulary
# ============================================================

class AllowOnlyActionWords(LogitsProcessor):
    """
    Logits processor that masks all tokens except the allowed action words.

    Use this during Stage 2 (action generation) to guarantee the model
    outputs only a valid action token.  Not required if you extract the
    action by string matching (get_action_from_response), but it helps
    keep action-token log-probs well-behaved.
    """

    def __init__(self, tokenizer, action_words: List[str]):
        self.tokenizer = tokenizer
        self.action_words = action_words

        self.allowed_token_ids: set = set()
        for word in action_words:
            for variant in [
                word, word.lower(), word.upper(), word.capitalize(),
                f" {word}", f" {word.lower()}", f" {word.upper()}", f" {word.capitalize()}",
            ]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                if len(tokens) == 1:          # only single-token encodings
                    self.allowed_token_ids.add(tokens[0])

        if tokenizer.eos_token_id is not None:
            self.allowed_token_ids.add(tokenizer.eos_token_id)

        self._allowed_tensor: Optional[torch.Tensor] = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._allowed_tensor is None or self._allowed_tensor.device != scores.device:
            self._allowed_tensor = torch.tensor(
                list(self.allowed_token_ids), device=scores.device, dtype=torch.long
            )
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self._allowed_tensor] = scores[:, self._allowed_tensor]
        return mask


# ============================================================
# MAIN TRAINER CLASS
# ============================================================

class YourGameGRPOTrainer:
    """
    GRPO trainer template.

    Replace 'YourGame' with your project name throughout this file.
    Sections marked  # === CUSTOMIZE ===  must be adapted to your environment.
    Sections marked  # === GENERIC ===  implement the core algorithm and
    should generally be left unchanged.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)

        # ------------------------------------------------------------------
        # ACCELERATOR (multi-GPU) setup  — GENERIC
        # ------------------------------------------------------------------
        if config.use_accelerate and not (config.use_8bit or config.use_4bit):
            timeout_kw = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
                log_with=None,
                kwargs_handlers=[timeout_kw],
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = config.device

        is_main = self.accelerator is None or self.accelerator.is_main_process

        # ------------------------------------------------------------------
        # TOKENIZER  — GENERIC
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required for batch generation with decoder-only models
        self.tokenizer.padding_side = "left"

        # ------------------------------------------------------------------
        # === CUSTOMIZE ===
        # Define the discrete action vocabulary for your environment.
        # These are the plain-text words the model is expected to output.
        # ------------------------------------------------------------------
        self.action_words: List[str] = [
            "action_a", "action_b", "action_c",   # <-- replace with your actions
        ]
        # Example for a grid-world:
        #   self.action_words = ["up", "down", "left", "right", "stay"]
        # Example for a card game:
        #   self.action_words = ["hit", "stand", "double", "split"]
        # ==================

        self.action_logits_processor = AllowOnlyActionWords(self.tokenizer, self.action_words)

        if is_main:
            logger.info(f"Actions        : {self.action_words}")
            logger.info(f"Model          : {config.model_name}")
            logger.info(f"Thinking tokens: {config.thinking_tokens}")
            logger.info(f"Log-prob mode  : {config.logprob_mode}")

        # ------------------------------------------------------------------
        # MODEL LOADING  — GENERIC
        # ------------------------------------------------------------------
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

        if config.use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif config.use_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        if not (config.use_8bit or config.use_4bit) and self.accelerator is None:
            base_model = base_model.to(self.device)

        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        if config.use_8bit or config.use_4bit:
            from peft import prepare_model_for_kbit_training
            base_model = prepare_model_for_kbit_training(base_model)

        # ------------------------------------------------------------------
        # LoRA  — GENERIC
        # ------------------------------------------------------------------
        if config.use_lora:
            target_modules = config.lora_target_modules
            if target_modules is None:
                linear_layers = {
                    name.split(".")[-1]
                    for name, m in base_model.named_modules()
                    if isinstance(m, torch.nn.Linear)
                }
                exclude = {"lm_head", "embed_tokens", "wte", "wpe", "ln", "norm"}
                target_modules = [n for n in linear_layers if not any(e in n.lower() for e in exclude)]
                if not target_modules:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(base_model, peft_cfg)
            if is_main:
                self.model.print_trainable_parameters()
        else:
            self.model = base_model

        self.model.train()
        # old_model is created on the first group (update_old_model)
        self.old_model = None

        # ------------------------------------------------------------------
        # OPTIMIZER & SCHEDULER  — GENERIC
        # ------------------------------------------------------------------
        trainable_params = (
            [p for p in self.model.parameters() if p.requires_grad]
            if config.use_lora
            else list(self.model.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, eps=1e-8, weight_decay=0.01
        )

        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return 0.95 ** ((step - config.warmup_steps) / 10)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.accelerator is not None and not (config.use_8bit or config.use_4bit):
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )

        # ------------------------------------------------------------------
        # === CUSTOMIZE ===
        # Instantiate your environment configuration and create the env.
        # The env object itself is typically recreated each episode (run_episode),
        # but keep the config here so episodes are reproducible.
        # ------------------------------------------------------------------
        # Example:
        #   self.env_config = MyEnvConfig(
        #       n_agents=config.num_agents,
        #       max_steps=config.max_env_steps,
        #       seed=config.seed,
        #   )
        self.env_config = None   # <-- replace with your env config
        # ==================

        # ------------------------------------------------------------------
        # TRAINING STATS & WANDB  — GENERIC
        # ------------------------------------------------------------------
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.training_step: int = 0

        if config.use_wandb and is_main:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=vars(config),
                reinit=True,
            )

        # Pre-generate fixed evaluation states
        self.eval_states: List = []
        self._generate_eval_states(num_states=config.num_eval_episodes)

    # ======================================================================
    # === CUSTOMIZE ===
    # OBSERVATION FORMATTING
    # ======================================================================

    def format_observation(self, raw_obs, agent_id: int, env) -> str:
        """
        Convert the raw environment observation into a natural-language string
        that will be passed to the LLM as the user message.

        Args:
            raw_obs : Whatever your env.reset() / env.step() returns for one agent.
                      Could be a dict, numpy array, string, etc.
            agent_id: Integer agent identifier.
            env     : The live environment object (access map/grid/etc. if needed).

        Returns:
            A human-readable string describing the current state for this agent.

        Example (grid-world):
            return (
                f"You are at position ({x}, {y}). "
                f"Enemies are at {enemy_positions}. "
                f"Goal is at {goal_pos}."
            )
        """
        # TODO: implement observation formatting for your environment
        return str(raw_obs)

    # ======================================================================
    # === CUSTOMIZE ===
    # PROMPT BUILDERS
    # ======================================================================

    def create_thinking_prompt(self, obs_text: str, agent_id: int) -> str:
        """
        Build the Stage-1 prompt (thinking/reasoning).

        The model is asked to reason about the situation.  Its output will be
        appended to the Stage-2 prompt so it can make an informed action choice.

        Args:
            obs_text : Formatted observation string (from format_observation).
            agent_id : Integer agent ID.

        Returns:
            A chat-template formatted prompt string (tokenizer handles encoding).
        """
        # === CUSTOMIZE: adjust system message to match your game ===
        system_msg = (
            "You are an agent playing [YOUR GAME NAME]. "
            "Your goal is to [DESCRIBE OBJECTIVE]. "
            "Available actions: "
            + ", ".join(f"{a} = [describe {a}]" for a in self.action_words)
            + ". Think step-by-step about the best action."
        )
        # ===========================================================

        messages = [
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": obs_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def create_action_prompt(self, obs_text: str, thinking_text: str) -> str:
        """
        Build the Stage-2 prompt (action selection).

        Combines the observation and the model's own thinking, then asks for
        a single action word.

        Args:
            obs_text     : Formatted observation string.
            thinking_text: The model's Stage-1 generated reasoning text.

        Returns:
            A chat-template formatted prompt string.
        """
        # === CUSTOMIZE: keep the same system message as thinking prompt ===
        system_msg = (
            "You are an agent playing [YOUR GAME NAME]. "
            "Your goal is to [DESCRIBE OBJECTIVE]. "
            "Available actions: "
            + ", ".join(f"{a} = [describe {a}]" for a in self.action_words)
            + "."
        )
        action_instruction = (
            "Based on your reasoning above, output ONLY ONE action word: "
            + ", ".join(self.action_words)
            + "."
        )
        # ==================================================================

        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": obs_text},
            {"role": "assistant", "content": thinking_text},
            {"role": "user",      "content": action_instruction},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ======================================================================
    # === CUSTOMIZE ===
    # ACTION EXTRACTION
    # ======================================================================

    def get_action_from_response(self, response: str) -> str:
        """
        Extract a valid action word from the model's raw text output.

        Strategy: scan the last few words of the response for a known action.
        Falls back to the first action in self.action_words on failure.

        Args:
            response: Raw decoded text from the model.

        Returns:
            A string that is guaranteed to be in self.action_words.
        """
        response = response.strip().lower()
        words = response.split()
        for i in range(min(5, len(words))):
            candidate = words[-(i + 1)].strip(".,!?;:")
            if candidate in self.action_words:
                return candidate

        # === CUSTOMIZE: change fallback action if needed ===
        default_action = self.action_words[0]
        # ===================================================
        logger.debug(f"No valid action in response '{response}', defaulting to '{default_action}'")
        return default_action

    # ======================================================================
    # === CUSTOMIZE ===
    # EVALUATION STATE GENERATION
    # ======================================================================

    def _generate_eval_states(self, num_states: int = 20):
        """
        Pre-generate fixed initial environment states for reproducible evaluation.

        Each state is captured after env.reset() so the same scenario can be
        replayed at evaluation time via env.set_state(state).

        If your environment does not support get_state/set_state, store
        seeds instead and reset with env.reset(seed=...).
        """
        # === CUSTOMIZE ===
        for i in range(num_states):
            # Example:
            #   env = MyGameEnv(MyEnvConfig(
            #       n_agents=self.config.num_agents,
            #       max_steps=self.config.max_env_steps,
            #       seed=self.config.seed + 1000 + i,
            #   ))
            #   env.reset()
            #   self.eval_states.append(env.get_state())
            self.eval_states.append({"seed": self.config.seed + 1000 + i})
        # ==================

        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            logger.info(f"Generated {num_states} fixed evaluation states.")

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # CORE GRPO UTILITIES
    # ======================================================================

    def update_old_model(self):
        """
        Snapshot the current policy as θ_old.

        Called once at the start of each training group.  The old model is
        frozen for all subsequent inner-epoch updates within that group.
        """
        is_main = self.accelerator is None or self.accelerator.is_main_process

        if self.old_model is None:
            if is_main:
                logger.info("Creating old model snapshot (first group)...")
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad = False
        else:
            if is_main:
                logger.info("Updating old model snapshot...")
            self.old_model.load_state_dict(self.model.state_dict())
            self.old_model.eval()

    def compute_batch_sequence_log_prob(
        self,
        model,
        prompt_input_ids_list: List[torch.Tensor],
        generated_ids_list: List[torch.Tensor],
        device: torch.device,
        pad_token_id: int,
        need_grad: bool = False,
    ) -> torch.Tensor:
        """
        Compute the summed log-probability of the generated tokens for each
        (prompt, generation) pair in a batched forward pass.

        Returns:
            Tensor of shape [batch_size] — one summed log-prob per sequence.
        """
        full_sequences, prompt_lens, gen_lens = [], [], []

        for p_ids, g_ids in zip(prompt_input_ids_list, generated_ids_list):
            p = p_ids.to(device).view(-1)
            p = p[p != pad_token_id]
            g = g_ids.to(device).view(-1)
            g = g[g != pad_token_id]
            prompt_lens.append(len(p))
            gen_lens.append(len(g))
            full_sequences.append(torch.cat([p, g]))

        batch_ids = pad_sequence(full_sequences, batch_first=True, padding_value=pad_token_id)
        attn_mask = (batch_ids != pad_token_id).long()

        def _forward():
            return model(input_ids=batch_ids, attention_mask=attn_mask).logits

        logits = _forward() if need_grad else (lambda: (torch.no_grad().__enter__(), _forward())[1])()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        token_log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # Mask to only the generated portion
        action_mask = torch.zeros_like(token_log_probs)
        for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
            start = max(0, p_len - 1)
            if g_len > 0:
                action_mask[i, start : start + g_len] = 1.0

        return (token_log_probs * action_mask).sum(dim=1)

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # ACTION GENERATION (single agent or batch)
    # ======================================================================

    def _get_gen_model(self, model):
        """Unwrap DDP/Accelerate wrapper for generation."""
        if self.accelerator is not None and hasattr(model, "module"):
            return model.module
        return model

    def generate_action(
        self,
        raw_obs,
        agent_id: int,
        step: int,
        env,
        model,
    ) -> Tuple[str, torch.Tensor, str, str, str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate one action for one agent using two-stage generation (sequential).

        Returns:
            action          : Chosen action string.
            log_prob        : Summed log-prob of the generated tokens (detached scalar tensor).
            thinking_text   : Stage-1 reasoning text.
            full_response   : Combined "thinking → action_text".
            action_text     : Raw Stage-2 output text.
            action_input_ids: Token IDs of the Stage-2 prompt (for training).
            action_ids      : Token IDs of the Stage-2 generated response.
        """
        target_device = self.device
        gen_model = self._get_gen_model(model)

        obs_text = self.format_observation(raw_obs, agent_id, env)

        # ---- Stage 1: thinking --------------------------------------------
        thinking_prompt = self.create_thinking_prompt(obs_text, agent_id)
        t_inputs = self.tokenizer(
            thinking_prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_length, padding=False,
        ).to(target_device)

        try:
            with torch.no_grad():
                t_outputs = gen_model.generate(
                    **t_inputs,
                    max_new_tokens=self.config.thinking_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Stage-1 generation failed: {e}")
            return (self.action_words[0], torch.tensor(0.0, device=target_device),
                    "", "", "", None, None)

        thinking_ids = t_outputs.sequences[0][t_inputs.input_ids.shape[1]:]
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True)

        # ---- Stage 2: action ----------------------------------------------
        action_prompt = self.create_action_prompt(obs_text, thinking_text)
        a_inputs = self.tokenizer(
            action_prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_length, padding=False,
        ).to(target_device)

        try:
            with torch.no_grad():
                a_outputs = gen_model.generate(
                    **a_inputs,
                    max_new_tokens=10,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Stage-2 generation failed: {e}")
            return (self.action_words[0], torch.tensor(0.0, device=target_device),
                    thinking_text, thinking_text, "", None, None)

        action_ids = a_outputs.sequences[0][a_inputs.input_ids.shape[1]:]
        action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
        action = self.get_action_from_response(action_text)

        # ---- Log-prob -----------------------------------------------------
        try:
            lp_model = self.old_model if self.old_model is not None else gen_model

            if self.config.logprob_mode == "action+thinking":
                t_lp = self.compute_batch_sequence_log_prob(
                    lp_model, [t_inputs.input_ids], [thinking_ids],
                    target_device, self.tokenizer.pad_token_id, need_grad=False,
                )[0]
                a_lp = self.compute_batch_sequence_log_prob(
                    lp_model, [a_inputs.input_ids], [action_ids],
                    target_device, self.tokenizer.pad_token_id, need_grad=False,
                )[0]
                log_prob = t_lp + a_lp
            else:
                log_prob = self.compute_batch_sequence_log_prob(
                    lp_model, [a_inputs.input_ids], [action_ids],
                    target_device, self.tokenizer.pad_token_id, need_grad=False,
                )[0]
        except Exception as e:
            logger.warning(f"Log-prob computation failed: {e}")
            log_prob = torch.tensor(-10.0, device=target_device)

        return (action, log_prob, thinking_text,
                f"{thinking_text} -> {action_text}", action_text,
                a_inputs.input_ids, action_ids)

    def generate_actions_batch(
        self,
        obs_dict: Dict[int, object],
        step: int,
        env,
        model,
    ) -> Dict[int, Tuple]:
        """
        Generate actions for all agents in one batched forward pass.

        Args:
            obs_dict: {agent_id → raw_obs} mapping.

        Returns:
            {agent_id → (action, log_prob, thinking_text, full_response,
                         action_text, action_prompt, action_input_ids, action_ids)}
        """
        target_device = self.device
        gen_model = self._get_gen_model(model)
        agent_ids = sorted(obs_dict.keys())
        n = len(agent_ids)

        # ---- Stage 1 batch ------------------------------------------------
        obs_texts, thinking_prompts = [], []
        for aid in agent_ids:
            obs_text = self.format_observation(obs_dict[aid], aid, env)
            obs_texts.append(obs_text)
            thinking_prompts.append(self.create_thinking_prompt(obs_text, aid))

        t_inputs = self.tokenizer(
            thinking_prompts, return_tensors="pt",
            truncation=True, max_length=self.config.max_length, padding=True,
        ).to(target_device)

        try:
            with torch.no_grad():
                t_outputs = gen_model.generate(
                    **t_inputs,
                    max_new_tokens=self.config.thinking_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch Stage-1 failed: {e}, falling back to sequential")
            return self._sequential_fallback(obs_dict, step, env, model)

        thinking_texts = [
            self.tokenizer.decode(
                t_outputs.sequences[i][t_inputs.input_ids[i].shape[0]:],
                skip_special_tokens=True,
            )
            for i in range(n)
        ]

        # ---- Stage 2 batch ------------------------------------------------
        action_prompts = [
            self.create_action_prompt(obs_texts[i], thinking_texts[i])
            for i in range(n)
        ]
        a_inputs = self.tokenizer(
            action_prompts, return_tensors="pt",
            truncation=True, max_length=self.config.max_length, padding=True,
        ).to(target_device)

        try:
            with torch.no_grad():
                a_outputs = gen_model.generate(
                    **a_inputs,
                    max_new_tokens=10,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Batch Stage-2 failed: {e}, falling back to sequential")
            return self._sequential_fallback(obs_dict, step, env, model)

        actions, action_texts, action_ids_list, action_input_ids_list = [], [], [], []
        for i in range(n):
            a_ids = a_outputs.sequences[i][a_inputs.input_ids[i].shape[0]:]
            a_text = self.tokenizer.decode(a_ids, skip_special_tokens=True)
            actions.append(self.get_action_from_response(a_text))
            action_texts.append(a_text)
            action_ids_list.append(a_ids)
            action_input_ids_list.append(a_inputs.input_ids[i])

        # ---- Log-probs batch ----------------------------------------------
        try:
            lp_model = self.old_model if self.old_model is not None else gen_model
            log_probs = self.compute_batch_sequence_log_prob(
                lp_model, action_input_ids_list, action_ids_list,
                target_device, self.tokenizer.pad_token_id, need_grad=False,
            )
        except Exception as e:
            logger.warning(f"Batch log-prob failed: {e}")
            log_probs = [torch.tensor(-10.0, device=target_device)] * n

        results = {}
        for i, aid in enumerate(agent_ids):
            results[aid] = (
                actions[i], log_probs[i], thinking_texts[i],
                f"{thinking_texts[i]} -> {action_texts[i]}",
                action_texts[i], action_prompts[i],
                action_input_ids_list[i], action_ids_list[i],
            )
        return results

    def _sequential_fallback(self, obs_dict, step, env, model):
        """Fallback: generate actions one agent at a time."""
        results = {}
        for aid in sorted(obs_dict.keys()):
            action, lp, thinking, full, a_text, a_in_ids, a_ids = self.generate_action(
                obs_dict[aid], aid, step, env, model
            )
            results[aid] = (action, lp, thinking, full, a_text, "", a_in_ids, a_ids)
        return results

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # EPISODE ROLLOUT
    # ======================================================================

    def run_episode(
        self,
        log_samples: bool = False,
        initial_state=None,
    ) -> Dict:
        """
        Run one full episode and collect the trajectory.

        Returns a dict with keys:
          prompts, actions, responses, action_prompts, action_texts,
          rewards, log_probs, agent_ids, observations,
          action_input_ids, action_ids,
          total_reward, steps, rollout_time, final_scores (if info provides it).
        """
        start_time = time.time()

        # === CUSTOMIZE ===
        # Create and reset your environment here.
        # If initial_state is provided, restore it for reproducible evaluation.
        # Example:
        #   env = MyGameEnv(self.env_config)
        #   if initial_state is not None:
        #       env.set_state(initial_state)
        #       obs = env._observation()
        #   else:
        #       obs = env.reset()
        env = None          # <-- replace
        obs = {}            # <-- replace: {agent_id: raw_obs, ...}
        # ==================

        trajectory = {
            "prompts": [], "actions": [], "responses": [],
            "action_prompts": [], "action_texts": [],
            "rewards": [], "log_probs": [], "agent_ids": [],
            "observations": [], "action_input_ids": [], "action_ids": [],
        }
        total_reward = 0.0

        for step in range(self.config.max_env_steps):
            batch_results = self.generate_actions_batch(obs, step, env, self.model)
            actions_to_env = {}

            for agent_id in range(1, self.config.num_agents + 1):
                (action, log_prob, thinking_text, full_response,
                 action_text, action_prompt, action_input_ids, action_ids) = batch_results[agent_id]

                actions_to_env[agent_id] = action

                # Reconstruct thinking prompt for storage
                obs_text = self.format_observation(obs[agent_id], agent_id, env)
                thinking_prompt = self.create_thinking_prompt(obs_text, agent_id)

                trajectory["prompts"].append(thinking_prompt)
                trajectory["actions"].append(action)
                trajectory["responses"].append(full_response)
                trajectory["action_prompts"].append(action_prompt)
                trajectory["action_texts"].append(action_text)
                trajectory["log_probs"].append(log_prob.detach().item())
                trajectory["agent_ids"].append(agent_id)
                trajectory["observations"].append(obs[agent_id])
                trajectory["action_input_ids"].append(action_input_ids)
                trajectory["action_ids"].append(action_ids)

                if log_samples and step == 0 and agent_id == 1:
                    logger.info(f"  Sample: obs='{obs_text[:80]}' thinking='{thinking_text[:80]}' action='{action}'")

            # === CUSTOMIZE ===
            # Step the environment.
            # obs, rewards, done, info = env.step(actions_to_env)
            # rewards should be {agent_id: float}
            obs, rewards, done, info = {}, {a: 0.0 for a in range(1, self.config.num_agents + 1)}, True, {}
            # ==================

            for agent_id in range(1, self.config.num_agents + 1):
                trajectory["rewards"].append(rewards.get(agent_id, 0.0))
                total_reward += rewards.get(agent_id, 0.0)

            if done:
                break

        trajectory["total_reward"] = total_reward
        trajectory["steps"] = step + 1
        trajectory["rollout_time"] = time.time() - start_time
        # === CUSTOMIZE ===
        # Store any per-episode summary from info if available.
        trajectory["final_scores"] = info.get("scores", {})
        # ==================

        return trajectory

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # TRAJECTORY LOGGING
    # ======================================================================

    def log_episode_to_file(self, trajectory: Dict, group_num: int, episode_idx: int):
        """Append one episode trajectory to a human-readable log file."""
        if not self.config.log_trajectory:
            return
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        log_path = os.path.join(self.config.output_dir, self.config.trajectory_log_file)
        with open(log_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"GROUP {group_num} | EPISODE {episode_idx}\n")
            f.write(f"Total Reward: {trajectory['total_reward']:.2f} | Steps: {trajectory['steps']}\n")
            f.write("=" * 80 + "\n\n")

            for step in range(trajectory["steps"]):
                f.write(f"--- Step {step + 1} ---\n")
                for a_idx in range(self.config.num_agents):
                    idx = step * self.config.num_agents + a_idx
                    if idx >= len(trajectory["observations"]):
                        break
                    agent_id = trajectory["agent_ids"][idx]
                    obs      = trajectory["observations"][idx]
                    action   = trajectory["actions"][idx]
                    reward   = trajectory["rewards"][idx]
                    response = trajectory["responses"][idx] if idx < len(trajectory["responses"]) else "N/A"
                    truncated = response[:300] + "..." if len(response) > 300 else response
                    f.write(f"\n  [Agent {agent_id}]\n")
                    f.write(f"    Obs    : {obs}\n")
                    f.write(f"    Think  : {truncated}\n")
                    f.write(f"    Action : {action}\n")
                    f.write(f"    Reward : {reward:.2f}\n")
                f.write("\n")

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # ADVANTAGE COMPUTATION
    # ======================================================================

    def compute_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Normalize episode returns to produce per-trajectory advantages.

        GRPO  : A = (R - mean_R) / (std_R + ε)
        DrGRPO: A = R - mean_R
        """
        returns = [t["total_reward"] for t in trajectories]
        use_std = self.config.loss_type.lower() == "grpo"

        if self.config.advantage_normalization and len(returns) > 1:
            if self.accelerator is not None:
                # Gather across GPUs for global normalization
                dev = next(self.model.parameters()).device
                local_t = torch.tensor(returns, dtype=torch.float32, device=dev)
                all_r = self.accelerator.gather(local_t)
                global_mean = all_r.mean().item()
                if use_std:
                    global_std = all_r.std().item() + 1e-8
                    norm_returns = [(r - global_mean) / global_std for r in returns]
                else:
                    norm_returns = [r - global_mean for r in returns]
            else:
                mean_r = float(np.mean(returns))
                if use_std:
                    std_r = float(np.std(returns)) + 1e-8
                    norm_returns = [(r - mean_r) / std_r for r in returns]
                else:
                    norm_returns = [r - mean_r for r in returns]
        else:
            norm_returns = list(returns)

        norm_returns = [
            float(np.clip(a, -self.config.clip_advantage, self.config.clip_advantage))
            for a in norm_returns
        ]

        for traj, adv in zip(trajectories, norm_returns):
            traj["advantage"] = adv
            traj["advantages"] = [adv] * len(traj["rewards"])

        return trajectories

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # LOSS COMPUTATION (micro-batching + gradient accumulation)
    # ======================================================================

    def flatten_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Convert a list of trajectory dicts into a flat list of (step-level) samples.
        Each sample contains the token IDs and scalars needed for loss computation.
        """
        samples = []
        for traj in trajectories:
            adv = traj["advantage"]
            for i in range(len(traj["prompts"])):
                a_in_ids = traj["action_input_ids"][i]
                a_ids    = traj["action_ids"][i]
                if a_in_ids is None or a_ids is None or len(a_ids) == 0:
                    continue
                samples.append({
                    "prompt_ids":   a_in_ids,
                    "action_ids":   a_ids,
                    "advantage":    adv,
                    "old_log_prob": traj["log_probs"][i],
                })
        return samples

    def create_minibatch_iterator(self, trajectories: List[Dict], minibatch_size: int):
        """Yield randomly shuffled mini-batches of trajectories."""
        indices = list(range(len(trajectories)))
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            yield [trajectories[i] for i in indices[start : start + minibatch_size]]

    def compute_loss_on_samples(
        self,
        samples: List[Dict],
        device,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Compute the PPO-clipped GRPO loss on a micro-batch of samples.

        Returns:
            loss         : Scalar tensor with gradients attached.
            clip_fraction: Fraction of samples where the ratio was clipped.
            n_samples    : Number of valid samples in this micro-batch.
        """
        if not samples:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

        new_log_probs = self.compute_batch_sequence_log_prob(
            self.model,
            [s["prompt_ids"] for s in samples],
            [s["action_ids"]  for s in samples],
            device, self.tokenizer.pad_token_id, need_grad=True,
        )
        old_log_probs = torch.tensor(
            [s["old_log_prob"] for s in samples], device=device, dtype=torch.float32
        )
        advantages = torch.tensor(
            [s["advantage"] for s in samples], device=device, dtype=torch.float32
        )

        log_ratio = new_log_probs - old_log_probs
        ratio     = torch.exp(log_ratio)
        eps       = self.config.epsilon
        clipped   = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        with torch.no_grad():
            clip_fraction = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean().item()

        return policy_loss, clip_fraction, len(samples)

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # MAIN TRAINING LOOP
    # ======================================================================

    def train(self):
        """Main GRPO training loop."""
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            os.makedirs(self.config.output_dir, exist_ok=True)
            logger.info("=" * 70)
            logger.info("GRPO Training")
            logger.info(f"  Model   : {self.config.model_name}")
            logger.info(f"  Loss    : {self.config.loss_type.upper()}")
            logger.info(f"  Agents  : {self.config.num_agents}")
            logger.info(f"  Actions : {self.action_words}")
            logger.info(f"  Episodes: {self.config.num_episodes}")
            logger.info("=" * 70)

        episode = 0
        best_reward = float("-inf")
        group_num = 0

        while episode < self.config.num_episodes:
            group_start = episode
            if is_main:
                logger.info(f"\n[Group {group_num}] Starting at episode {episode}")

            # 1. Snapshot old model
            self.update_old_model()

            # 2. Collect trajectories
            if self.accelerator is not None:
                trajectories = []
                for i in range(self.config.episodes_per_gpu):
                    try:
                        log_samples = (
                            i == 0
                            and group_num % self.config.log_interval == 0
                            and self.accelerator.is_main_process
                        )
                        traj = self.run_episode(log_samples=log_samples)
                        trajectories.append(traj)
                        logger.info(
                            f"  [GPU{self.accelerator.process_index}] "
                            f"Ep{i}: R={traj['total_reward']:.2f}, Steps={traj['steps']}"
                        )
                    except Exception as e:
                        logger.error(f"Episode failed: {e}", exc_info=True)

                # Count episodes across GPUs
                dev = next(self.model.parameters()).device
                ct  = self.accelerator.gather(
                    torch.tensor([len(trajectories)], dtype=torch.long, device=dev)
                )
                episode += ct.sum().item()
            else:
                trajectories = []
                for i in range(self.config.episodes_per_update):
                    try:
                        log_samples = (i == 0 and group_num % self.config.log_interval == 0)
                        traj = self.run_episode(log_samples=log_samples)
                        trajectories.append(traj)
                        logger.info(
                            f"  Ep{episode + i}: R={traj['total_reward']:.2f}, Steps={traj['steps']}"
                        )
                    except Exception as e:
                        logger.error(f"Episode failed: {e}", exc_info=True)
                episode += len(trajectories)

            if not trajectories:
                logger.warning("No valid trajectories, skipping update.")
                group_num += 1
                continue

            # Log one episode for monitoring
            if is_main and self.config.log_trajectory:
                idx = random.randint(0, len(trajectories) - 1)
                self.log_episode_to_file(trajectories[idx], group_num, idx)

            # 3. Compute advantages
            trajectories = self.compute_advantages(trajectories)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            # Log reward statistics
            rewards_list = [t["total_reward"] for t in trajectories]
            if self.accelerator is not None:
                dev = next(self.model.parameters()).device
                all_r = self.accelerator.gather(
                    torch.tensor(rewards_list, dtype=torch.float32, device=dev)
                ).cpu().tolist()
            else:
                all_r = rewards_list

            if is_main:
                avg_r = float(np.mean(all_r))
                std_r = float(np.std(all_r))
                self.episode_rewards.append(avg_r)
                logger.info(
                    f"  Reward: {avg_r:.2f}±{std_r:.2f} "
                    f"[{min(all_r):.2f}, {max(all_r):.2f}]"
                )

                # Log action distribution
                all_actions = [a for t in trajectories for a in t["actions"]]
                total_a = max(len(all_actions), 1)
                dist = " | ".join(
                    f"{a}:{all_actions.count(a) * 100 // total_a}%"
                    for a in self.action_words
                )
                logger.info(f"  Actions: {dist}")

                if self.config.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "reward/mean": avg_r,
                        "reward/std": std_r,
                        **{f"actions/{a}": all_actions.count(a) / total_a for a in self.action_words},
                    }, step=episode)

            # 4. Inner optimization epochs
            try:
                epoch_losses: List[float] = []
                epoch_clips:  List[float] = []
                final_grad_norm = 0.0

                for inner_epoch in range(self.config.num_inner_epochs):
                    ep_loss_sum, ep_clip_sum, ep_n = 0.0, 0.0, 0

                    for mb in self.create_minibatch_iterator(trajectories, self.config.minibatch_size):
                        self.optimizer.zero_grad()

                        dev = (
                            next(self.model.parameters()).device
                            if self.accelerator is not None
                            else self.device
                        )

                        all_samples = self.flatten_trajectories(mb)
                        total_s = len(all_samples)
                        if total_s == 0:
                            continue

                        mbs = self.config.samples_per_micro_batch
                        n_chunks = (total_s + mbs - 1) // mbs

                        for i in range(0, total_s, mbs):
                            chunk = all_samples[i : i + mbs]
                            loss, cf, ns = self.compute_loss_on_samples(chunk, dev)
                            if ns > 0:
                                (loss / n_chunks).backward()
                                ep_loss_sum += loss.item() * ns
                                ep_clip_sum += cf * ns
                                ep_n += ns

                        if ep_n > 0:
                            if self.accelerator is not None:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(), self.config.max_grad_norm
                                )
                            else:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.config.max_grad_norm
                                )
                            self.optimizer.step()
                            final_grad_norm = grad_norm

                    if ep_n > 0:
                        epoch_losses.append(ep_loss_sum / ep_n)
                        epoch_clips.append(ep_clip_sum / ep_n)

                self.scheduler.step()
                self.training_step += 1

                if is_main and epoch_losses:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"  Loss={epoch_losses[-1]:.4f}, "
                        f"ClipFrac={epoch_clips[-1]:.3f}, "
                        f"GradNorm={final_grad_norm:.4f}, LR={lr:.2e}"
                    )
                    if self.config.use_wandb:
                        wandb.log({
                            "train/loss": epoch_losses[-1],
                            "train/clip_fraction": epoch_clips[-1],
                            "train/grad_norm": final_grad_norm,
                            "train/learning_rate": lr,
                        }, step=episode)

            except Exception as e:
                logger.error(f"Training step failed: {e}", exc_info=True)
                group_num += 1
                continue

            # 5. Checkpoint
            if is_main and (
                episode % self.config.save_steps == 0
                or episode >= self.config.num_episodes
            ):
                avg_r = float(np.mean([t["total_reward"] for t in trajectories]))
                if avg_r >= best_reward:
                    best_reward = avg_r
                    ckpt_dir = os.path.join(self.config.output_dir, "best_model")
                    try:
                        m = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        m.save_pretrained(ckpt_dir)
                        self.tokenizer.save_pretrained(ckpt_dir)
                        logger.info(f"  Saved best model (R={best_reward:.2f}) → {ckpt_dir}")
                    except Exception as e:
                        logger.warning(f"  Checkpoint save failed: {e}")

                with open(os.path.join(self.config.output_dir, "training_stats.json"), "w") as f:
                    json.dump({
                        "episode": episode, "group": group_num,
                        "rewards": self.episode_rewards, "best_reward": best_reward,
                        "training_step": self.training_step,
                    }, f, indent=2)

            # 6. Mid-training evaluation
            if (
                self.config.eval_interval > 0
                and episode % self.config.eval_interval == 0
                and episode < self.config.num_episodes
            ):
                self.evaluate(
                    num_episodes=self.config.num_eval_episodes,
                    current_episode=episode,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            group_num += 1

        if is_main:
            logger.info(f"\n=== Training complete. Best reward: {best_reward:.2f} ===")

        return self.model

    # ======================================================================
    # === GENERIC — DO NOT MODIFY ===
    # EVALUATION
    # ======================================================================

    def evaluate(self, num_episodes: int = 20, current_episode: int = None):
        """
        Evaluate the current policy on fixed, pre-generated initial states.

        In multi-GPU mode, episodes are split across GPUs in round-robin fashion
        and results are gathered before computing statistics.
        """
        is_main = self.accelerator is None or self.accelerator.is_main_process
        actual_n = min(num_episodes, len(self.eval_states))

        if is_main:
            logger.info(f"\n=== Evaluation ({actual_n} episodes) ===")

        self.model.eval()

        if self.accelerator is not None:
            proc_idx = self.accelerator.process_index
            n_proc   = self.accelerator.num_processes
            local_idxs = list(range(proc_idx, actual_n, n_proc))
        else:
            proc_idx   = 0
            local_idxs = list(range(actual_n))

        local_rewards = []
        for i in local_idxs:
            try:
                traj = self.run_episode(
                    log_samples=False,
                    initial_state=self.eval_states[i],
                )
                local_rewards.append(traj["total_reward"])
            except Exception as e:
                logger.error(f"Eval episode {i} failed: {e}", exc_info=True)

        # Gather across GPUs
        if self.accelerator is not None:
            dev = next(self.model.parameters()).device
            r_t = torch.tensor(local_rewards or [0.0], dtype=torch.float32, device=dev)
            all_r = self.accelerator.gather(r_t).cpu().tolist()
        else:
            all_r = local_rewards

        if is_main and all_r:
            avg_r = float(np.mean(all_r))
            std_r = float(np.std(all_r))
            logger.info(f"  Eval Reward: {avg_r:.2f}±{std_r:.2f}")
            if self.config.use_wandb and current_episode is not None:
                wandb.log({"eval/reward_mean": avg_r, "eval/reward_std": std_r},
                          step=current_episode)

        self.model.train()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GRPO Training")
    # Override any GRPOConfig field from the command line:
    parser.add_argument("--model_name",         default=None)
    parser.add_argument("--num_episodes",        type=int,   default=None)
    parser.add_argument("--num_agents",          type=int,   default=None)
    parser.add_argument("--episodes_per_update", type=int,   default=None)
    parser.add_argument("--learning_rate",       type=float, default=None)
    parser.add_argument("--output_dir",          default=None)
    parser.add_argument("--use_wandb",           action="store_true")
    parser.add_argument("--seed",                type=int,   default=None)
    args = parser.parse_args()

    config = GRPOConfig()

    # Apply any command-line overrides
    for field_name, value in vars(args).items():
        if value is not None and hasattr(config, field_name):
            setattr(config, field_name, value)

    trainer = YourGameGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
