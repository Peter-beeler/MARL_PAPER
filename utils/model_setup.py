"""
Model loading, LoRA setup, optimizer/scheduler, and AllowOnlyActionWords.
"""

import torch
import logging
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from .config import GRPOConfig

logger = logging.getLogger(__name__)


def log_cuda_memory(stage: str, device: int = 0):
    """Log CUDA memory usage at a specific stage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info(
            f"[CUDA Memory - {stage}] Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB"
        )


class AllowOnlyActionWords(LogitsProcessor):
    """Logits processor that restricts output to only action word tokens (text mode)."""

    def __init__(self, tokenizer, action_words: List[str]):
        self.tokenizer = tokenizer
        self.action_words = action_words
        self.allowed_token_ids = set()

        for word in action_words:
            for variant in [
                word, word.lower(), word.upper(), word.capitalize(),
                f" {word}", f" {word.lower()}", f" {word.upper()}", f" {word.capitalize()}"
            ]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                if len(tokens) == 1:
                    self.allowed_token_ids.add(tokens[0])

        if tokenizer.eos_token_id is not None:
            self.allowed_token_ids.add(tokenizer.eos_token_id)

        self.allowed_tensor = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.allowed_tensor is None or self.allowed_tensor.device != scores.device:
            self.allowed_tensor = torch.tensor(
                list(self.allowed_token_ids), device=scores.device, dtype=torch.long
            )
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed_tensor] = scores[:, self.allowed_tensor]
        return mask


def setup_accelerator(config: GRPOConfig):
    """Initialize Accelerator for multi-GPU training."""
    if config.use_accelerate and not (config.use_8bit or config.use_4bit):
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

        if getattr(config, 'use_deepspeed', False):
            from accelerate.utils import DeepSpeedPlugin
            # ZeRO-3: shard params + gradients across all GPUs.
            # Optimizer states are CPU-offloaded (~8 GB/GPU saved on 4B model).
            # gradient_clipping is configured here so we must NOT call
            # accelerator.clip_grad_norm_() in the training loop.
            ds_plugin = DeepSpeedPlugin(
                zero_stage=3,
                gradient_accumulation_steps=1,
                gradient_clipping=config.max_grad_norm,
                offload_optimizer_device="cpu",
                zero3_init_flag=True,        # partition params during from_pretrained
                zero3_save_16bit_model=True, # gather all shards when saving
            )
            accelerator = Accelerator(
                mixed_precision="bf16",
                deepspeed_plugin=ds_plugin,
                kwargs_handlers=[timeout_kwargs],
            )
            logger.info(
                f"Using DeepSpeed ZeRO-3 (optimizer CPU-offload) "
                f"with {accelerator.num_processes} GPUs"
            )
        else:
            accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=1,
                log_with=None,
                kwargs_handlers=[timeout_kwargs]
            )
            logger.info(f"Using Accelerator with {accelerator.num_processes} processes")

        device = accelerator.device
    else:
        accelerator = None
        device = config.device
        if config.use_8bit or config.use_4bit:
            logger.info(f"Using quantization (8bit={config.use_8bit}, 4bit={config.use_4bit})")
    return accelerator, device


def load_tokenizer(config: GRPOConfig):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


from accelerate.state import AcceleratorState
def load_base_model(config: GRPOConfig):
    accelerator = Accelerator()
    
    # If using ZeRO-3, we must use the deepspeed_config via accelerator
    # to ensure the model is partitioned correctly during loading.
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    # Only use low_cpu_mem_usage if NOT using ZeRO-3, 
    # as DS-ZeRO-3 has its own memory management logic.
    is_ds_zero3 = (
        getattr(accelerator.state, "deepspeed_plugin", None) is not None and 
        accelerator.state.deepspeed_plugin.zero_stage == 3
    )
    
    if is_ds_zero3:
        # ZeRO-3 specific: No device_map, no low_cpu_mem_usage
        model_kwargs["device_map"] = None
        model_kwargs["low_cpu_mem_usage"] = False 
    else:
        model_kwargs["device_map"] = {"": accelerator.process_index}
        model_kwargs["low_cpu_mem_usage"] = True

    # Critical: Load the model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    return model

def apply_lora(base_model, config: GRPOConfig, accelerator=None, device=None):
    """Apply LoRA to base model. Returns the PEFT model."""
    if not config.use_lora:
        return base_model

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

    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Applying LoRA to modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)

    if accelerator is None or accelerator.is_main_process:
        model.print_trainable_parameters()
        log_cuda_memory("After LoRA model loaded")

    return model


def setup_model_for_training(base_model, config: GRPOConfig, accelerator=None, device=None):
    """
    Full model setup pipeline:
    1. Move to device (if not quantized/accelerate)
    2. Apply gradient checkpointing
    3. Prepare for kbit training
    4. Apply LoRA
    5. Set to train mode
    Returns the configured model.
    """
    # Move to device if not using quantization or accelerate
    if not (config.use_8bit or config.use_4bit) and (accelerator is None):
        base_model = base_model.to(device)

    # Gradient checkpointing
    if config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    # Prepare for kbit training
    if config.use_8bit or config.use_4bit:
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model)

    # Apply LoRA (or keep as base model)
    model = apply_lora(base_model, config, accelerator, device)

    if not config.use_lora and (accelerator is None or accelerator.is_main_process):
        log_cuda_memory("After base model loaded")

    model.train()
    return model


def setup_optimizer_and_scheduler(model, config: GRPOConfig):
    """Set up AdamW optimizer and exponential LR scheduler."""
    if config.use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    else:
        trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        eps=1e-8,
        weight_decay=0.01
    )

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            return 0.95 ** ((step - config.warmup_steps) / 10)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler
