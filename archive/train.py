"""
train.py — GRPO training for CleanupEnvMove using LLM agents.

This script implements the GRPO algorithm to train agents in the Cleanup environment.
It uses the provided helper functions and prompt templates to structure the interaction.
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
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any

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

# === CUSTOMIZE: Import Environment and Helpers ===
from env_move import CleanupEnvMove, Config as EnvConfig
import helpers
import prompt_template
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
# MAIN TRAINER CLASS
# ============================================================

class CleanupGRPOTrainer:
    """
    GRPO trainer for the CleanupEnvMove environment.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)

        # ------------------------------------------------------------------
        # ACCELERATOR (multi-GPU) setup
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
        # TOKENIZER
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ------------------------------------------------------------------
        # ACTION VOCABULARY
        # ------------------------------------------------------------------
        # Since the agents output JSON, we do not restrict the vocabulary 
        # to specific words using LogitsProcessor in this implementation.
        # We define the high-level function names for logging purposes.
        self.action_words: List[str] = [
            "move_to", "clean_at", "eat_at", "random_explore"
        ]

        if is_main:
            logger.info(f"High-level Actions: {self.action_words}")
            logger.info(f"Model          : {config.model_name}")
            logger.info(f"Thinking tokens: {config.thinking_tokens}")
            logger.info(f"Log-prob mode  : {config.logprob_mode}")

        # ------------------------------------------------------------------
        # MODEL LOADING
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
        # LoRA
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
        self.old_model = None

        # ------------------------------------------------------------------
        # OPTIMIZER & SCHEDULER
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
        # ENVIRONMENT CONFIG
        # ------------------------------------------------------------------
        self.env_config = EnvConfig(
            n_agents=config.num_agents,
            max_steps=config.max_env_steps,
            seed=config.seed,
            # Adjust spawn rates if necessary for training difficulty
            dirt_spawn_prob=0.05,
            apple_spawn_base=0.05
        )

        # ------------------------------------------------------------------
        # TRAINING STATS & WANDB
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
    # OBSERVATION FORMATTING
    # ======================================================================

    def format_observation(self, raw_obs, agent_id: int, env) -> str:
        """
        Uses the helper function to generate the natural language observation.
        Note: 'raw_obs' from env.step() is the local grid string, but the helper
        needs the full env object to generate the description.
        """
        # The helper function accesses env internals directly
        return helpers.get_observation_description(env, agent_id)

    # ======================================================================
    # PROMPT BUILDERS
    # ======================================================================

    def create_thinking_prompt(self, obs_text: str, agent_id: int, env: Any) -> str:
        """
        Build the Stage-1 prompt (thinking/reasoning) using prompt_template.
        """
        # prompt_template.create_thinking_prompt returns a list of dicts (messages)
        # We need to pass the env and agent_id to it.
        messages = prompt_template.create_thinking_prompt(env, agent_id)
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def create_action_prompt(self, obs_text: str, thinking_text: str, agent_id: int, env: Any) -> str:
        """
        Build the Stage-2 prompt (action selection).
        Reconstructs the prompt based on the prompt_template logic.
        """
        # Reconstruct the system message and user context
        system_msg = prompt_template.get_system_context() + "\n" + prompt_template.get_action_api()
        
        # We need the observation text again
        current_obs = helpers.get_observation_description(env, agent_id)
        
        user_content = f"""
You are Agent {agent_id}.
Observation: {current_obs}

Your Analysis:
{thinking_text}

Based on your analysis, output the specific JSON action to execute now.
"""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ======================================================================
    # ACTION EXTRACTION
    # ======================================================================

    def get_action_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract the JSON object from the model's response.
        Returns a dict with keys: action, agent_id, args.
        """
        response = response.strip()
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                if "action" in data:
                    return data
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse raw string if it looks like a function call (legacy support)
        # or just return a random explore action if parsing fails
        return {"action": "random_explore", "agent_id": 0, "args": {}}

    # ======================================================================
    # EVALUATION STATE GENERATION
    # ======================================================================

    def _generate_eval_states(self, num_states: int = 20):
        """
        Pre-generate fixed initial environment states for reproducible evaluation.
        """
        for i in range(num_states):
            env = CleanupEnvMove(self.env_config)
            # Override seed for diversity
            env.cfg.seed = self.config.seed + 1000 + i
            env.rng = random.Random(env.cfg.seed)
            env.reset()
            self.eval_states.append(env.get_state())

        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            logger.info(f"Generated {num_states} fixed evaluation states.")

    # ======================================================================
    # CORE GRPO UTILITIES
    # ======================================================================

    def update_old_model(self):
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

        if need_grad:
            logits = _forward()
        else:
            with torch.no_grad():
                logits = _forward()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        token_log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        action_mask = torch.zeros_like(token_log_probs)
        for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
            start = max(0, p_len - 1)
            if g_len > 0:
                action_mask[i, start : start + g_len] = 1.0

        return (token_log_probs * action_mask).sum(dim=1)

    def _get_gen_model(self, model):
        if self.accelerator is not None and hasattr(model, "module"):
            return model.module
        return model

    # ======================================================================
    # ACTION GENERATION
    # ======================================================================

    def generate_action(
        self,
        raw_obs,
        agent_id: int,
        step: int,
        env,
        model,
    ) -> Tuple[Dict, torch.Tensor, str, str, str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate one action for one agent using two-stage generation.
        Returns parsed JSON dict as 'action'.
        """
        target_device = self.device
        gen_model = self._get_gen_model(model)

        obs_text = self.format_observation(raw_obs, agent_id, env)

        # ---- Stage 1: thinking --------------------------------------------
        thinking_prompt = self.create_thinking_prompt(obs_text, agent_id, env)
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
            return ({"action": "stay"}, torch.tensor(0.0, device=target_device),
                    "", "", "", None, None)

        thinking_ids = t_outputs.sequences[0][t_inputs.input_ids.shape[1]:]
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True)

        # ---- Stage 2: action (JSON) ---------------------------------------
        action_prompt = self.create_action_prompt(obs_text, thinking_text, agent_id, env)
        a_inputs = self.tokenizer(
            action_prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_length, padding=False,
        ).to(target_device)

        try:
            with torch.no_grad():
                a_outputs = gen_model.generate(
                    **a_inputs,
                    max_new_tokens=128, # Enough for JSON
                    temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        except RuntimeError as e:
            logger.warning(f"Stage-2 generation failed: {e}")
            return ({"action": "stay"}, torch.tensor(0.0, device=target_device),
                    thinking_text, thinking_text, "", None, None)

        action_ids = a_outputs.sequences[0][a_inputs.input_ids.shape[1]:]
        action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
        
        # Parse JSON
        action_dict = self.get_action_from_response(action_text)

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

        return (action_dict, log_prob, thinking_text,
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
            thinking_prompts.append(self.create_thinking_prompt(obs_text, aid, env))

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
            self.create_action_prompt(obs_texts[i], thinking_texts[i], agent_ids[i], env)
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
                    max_new_tokens=128, # JSON length
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
        results = {}
        for aid in sorted(obs_dict.keys()):
            action, lp, thinking, full, a_text, a_in_ids, a_ids = self.generate_action(
                obs_dict[aid], aid, step, env, model
            )
            results[aid] = (action, lp, thinking, full, a_text, "", a_in_ids, a_ids)
        return results

    # ======================================================================
    # EPISODE ROLLOUT
    # ======================================================================

    def run_episode(
        self,
        log_samples: bool = False,
        initial_state=None,
    ) -> Dict:
        """
        Run one full episode and collect the trajectory.
        """
        start_time = time.time()

        # Initialize environment
        env = CleanupEnvMove(self.env_config)
        if initial_state is not None:
            env.set_state(initial_state)
            obs = env._observation()
        else:
            obs = env.reset()

        trajectory = {
            "prompts": [], "actions": [], "responses": [],
            "action_prompts": [], "action_texts": [],
            "rewards": [], "log_probs": [], "agent_ids": [],
            "observations": [], "action_input_ids": [], "action_ids": [],
        }
        total_reward = 0.0

        for step in range(self.config.max_env_steps):
            batch_results = self.generate_actions_batch(obs, step, env, self.model)
            
            # Prepare low-level actions for the environment
            env_actions = {}

            for agent_id in range(1, self.config.num_agents + 1):
                (action_dict, log_prob, thinking_text, full_response,
                 action_text, action_prompt, action_input_ids, action_ids) = batch_results[agent_id]

                # Convert High-Level JSON Action -> Low-Level Env Action
                # action_dict looks like: {"action": "move_to", "args": {"coord_x": 1, "coord_y": 2}}
                func_name = action_dict.get("action", "stay")
                args = action_dict.get("args", {})
                
                low_level_action = "stay"
                
                try:
                    if func_name == "move_to":
                        low_level_action, _ = helpers.move_to(env, agent_id, args.get("coord_x", 0), args.get("coord_y", 0))
                    elif func_name == "clean_at":
                        low_level_action, _ = helpers.clean_at(env, agent_id, args.get("coord_x", 0), args.get("coord_y", 0))
                    elif func_name == "eat_at":
                        low_level_action, _ = helpers.eat_at(env, agent_id, args.get("coord_x", 0), args.get("coord_y", 0))
                    elif func_name == "random_explore":
                        low_level_action, _ = helpers.random_explore(env, agent_id)
                    else:
                        low_level_action = "stay"
                except Exception as e:
                    # Fallback if args are missing or invalid
                    low_level_action = "stay"

                env_actions[agent_id] = low_level_action

                # Reconstruct thinking prompt for storage (needed for loss calculation later)
                obs_text = self.format_observation(obs[agent_id], agent_id, env)
                thinking_prompt = self.create_thinking_prompt(obs_text, agent_id, env)

                trajectory["prompts"].append(thinking_prompt)
                trajectory["actions"].append(func_name) # Store high-level name
                trajectory["responses"].append(full_response)
                trajectory["action_prompts"].append(action_prompt)
                trajectory["action_texts"].append(action_text)
                trajectory["log_probs"].append(log_prob.detach().item())
                trajectory["agent_ids"].append(agent_id)
                trajectory["observations"].append(obs[agent_id])
                trajectory["action_input_ids"].append(action_input_ids)
                trajectory["action_ids"].append(action_ids)

                if log_samples and step == 0 and agent_id == 1:
                    logger.info(f"  Sample: obs='{obs_text[:80]}...'")
                    logger.info(f"  Think : '{thinking_text[:80]}...'")
                    logger.info(f"  JSON  : '{action_text}'")
                    logger.info(f"  Exec  : {func_name} -> {low_level_action}")

            # Step the environment
            obs, rewards, done, info = env.step(env_actions)

            for agent_id in range(1, self.config.num_agents + 1):
                trajectory["rewards"].append(rewards.get(agent_id, 0.0))
                total_reward += rewards.get(agent_id, 0.0)

            if done:
                break

        trajectory["total_reward"] = total_reward
        trajectory["steps"] = step + 1
        trajectory["rollout_time"] = time.time() - start_time
        trajectory["final_scores"] = info.get("scores", {})

        return trajectory

    # ======================================================================
    # TRAJECTORY LOGGING
    # ======================================================================

    def log_episode_to_file(self, trajectory: Dict, group_num: int, episode_idx: int):
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
                    action   = trajectory["actions"][idx]
                    reward   = trajectory["rewards"][idx]
                    response = trajectory["responses"][idx] if idx < len(trajectory["responses"]) else "N/A"
                    truncated = response[:300] + "..." if len(response) > 300 else response
                    f.write(f"\n  [Agent {agent_id}]\n")
                    f.write(f"    Think+JSON : {truncated}\n")
                    f.write(f"    Action Name: {action}\n")
                    f.write(f"    Reward     : {reward:.2f}\n")
                f.write("\n")

    # ======================================================================
    # ADVANTAGE COMPUTATION
    # ======================================================================

    def compute_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        returns = [t["total_reward"] for t in trajectories]
        use_std = self.config.loss_type.lower() == "grpo"

        if self.config.advantage_normalization and len(returns) > 1:
            if self.accelerator is not None:
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
    # LOSS COMPUTATION
    # ======================================================================

    def flatten_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
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
        indices = list(range(len(trajectories)))
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            yield [trajectories[i] for i in indices[start : start + minibatch_size]]

    def compute_loss_on_samples(
        self,
        samples: List[Dict],
        device,
    ) -> Tuple[torch.Tensor, float, int]:
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
    # MAIN TRAINING LOOP
    # ======================================================================

    def train(self):
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            os.makedirs(self.config.output_dir, exist_ok=True)
            logger.info("=" * 70)
            logger.info("GRPO Training - Cleanup Environment")
            logger.info(f"  Model   : {self.config.model_name}")
            logger.info(f"  Agents  : {self.config.num_agents}")
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
    # EVALUATION
    # ======================================================================

    def evaluate(self, num_episodes: int = 20, current_episode: int = None):
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

    parser = argparse.ArgumentParser(description="GRPO Training for CleanupEnv")
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

    for field_name, value in vars(args).items():
        if value is not None and hasattr(config, field_name):
            setattr(config, field_name, value)

    trainer = CleanupGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()