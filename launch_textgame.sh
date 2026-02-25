#!/bin/bash
#SBATCH --job-name=grpo_textgame
#SBATCH --account=PAS2138
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=quad
#SBATCH --output=grpo_textgame_%j.out
#SBATCH --error=grpo_textgame_%j.err

#
# SLURM batch script for grpo_textgame.py — supports both text and compound modes.
#
# Usage:
#   sbatch launch_textgame.sh
#
# Key configuration knobs:
#   ACTION_MODE    — "text" (word actions) or "compound" (JSON helper actions)
#   USE_TWO_STAGE  — true/false (text mode only: two-stage thinking→action)
#   LOGPROB_MODE   — "action" or "action+thinking"
#   WANDB_*        — wandb logging settings
#

source ~/.bashrc
conda activate py311LLM
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# module load cuda/12.4.1

# Number of GPUs (reads from SLURM allocation, default 2)
NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}

echo "=========================================="
echo "GRPO TextGame Training (SLURM Job)"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURM_NODELIST:-localhost}"
echo "Num GPUs:   $NUM_GPUS"
echo "Working dir: $(pwd)"
echo ""

# ── Environment variables ──────────────────────────────────────────────────
# Force PyTorch to avoid any fancy collective networking
export COLL_NET_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
# Force NCCL to stay away from P2P and Shared Memory
# ── Hardware/Kernel Workarounds ───────────────────────────────────────────
# Force NCCL to behave like it's on a slow, non-GPU-direct network
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_GDR_LEVEL=0

# Explicitly tell NCCL to use the Loopback interface for all communication
# This avoids it trying to "probe" physical NICs that might trigger the crash
export NCCL_SOCKET_IFNAME=lo 

# Additional PyTorch/CUDA safety for older kernels
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DIST_DEBUG=INFO          # Disable collective networking

# Ensure PyTorch doesn't try to use the UCX backend which can also SIGSEGV
export COLL_NET_ENABLE=0
export PYTHONFAULTHANDLER=1        # Will print a traceback for the SIGSEGV
# A4000 GPUs have no NVLink; both PCIe P2P and POSIX SHM transports
# segfault on Linux kernel < 5.5 with NCCL 2.21+.
# Disable both so NCCL falls back to TCP socket transport (slower but stable).
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Suppress TensorFlow CUDA-plugin conflict noise (TF is in env but not used).
export TF_CPP_MIN_LOG_LEVEL=3
# Point Triton autotune cache to local /tmp to avoid NFS-related hangs.
export TRITON_CACHE_DIR=/tmp/triton_cache_$$

NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "Setting CUDA_HOME to: $CUDA_HOME"
fi

# ── Action mode ────────────────────────────────────────────────────────────
ACTION_MODE="compound"          # "text" or "compound"
USE_TWO_STAGE=true          # text mode: true = two-stage (thinking→word), false = single-stage
LOGPROB_MODE="action"       # "action" or "action+thinking" (text mode only)

# ── DeepSpeed ZeRO-3 ───────────────────────────────────────────────────────
# Shards model params, gradients, and optimizer states across all GPUs so a
# 4B model fits in 4 × 16 GB A4000s.  Optimizer states are CPU-offloaded.
# Set to false to use plain DDP (requires larger per-GPU VRAM).
USE_DEEPSPEED=true

# ── Model ──────────────────────────────────────────────────────────────────
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME="Qwen/Qwen2.5-0.5B"  # smaller for testing;
# ── Token budgets ──────────────────────────────────────────────────────────
THINKING_TOKENS=256         # tokens for stage 1 reasoning
ACTION_TOKENS=128           # tokens for stage 2 action/JSON generation

# ── Training ───────────────────────────────────────────────────────────────
TOTAL_EPISODES=2048
EPISODES_PER_GPU=2         # episodes each GPU collects per group
MAX_ENV_STEPS=3
NUM_AGENTS=1
LEARNING_RATE=1e-6
LOSS_TYPE="drgrpo"          # "grpo" or "drgrpo"

# ── Inner optimization (PPO-style) ─────────────────────────────────────────
NUM_INNER_EPOCHS=4
MINIBATCH_SIZE=8
SAMPLES_PER_MICRO_BATCH=3  # lower if OOM; 1-2 for A100 40 GB

# ── Rewards ────────────────────────────────────────────────────────────────
EAT_REWARD=1.0
CLEAN_REWARD=0.2            # 0.0 = disabled

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR="./grpo_textgame_checkpoints"
NUM_EVAL_EPISODES=10

# ── Wandb ─────────────────────────────────────────────────────────────────
USE_WANDB=true
WANDB_PROJECT="grpo_textgame"
WANDB_ENTITY=""             # leave empty for default account
WANDB_RUN_NAME="${ACTION_MODE}_run"

# ── Print config ──────────────────────────────────────────────────────────
echo "Configuration:"
echo "  Action mode:             $ACTION_MODE"
if [ "$ACTION_MODE" = "text" ]; then
    echo "  Two-stage generation:    $USE_TWO_STAGE"
    echo "  Log-prob mode:           $LOGPROB_MODE"
fi
echo "  Model:                   $MODEL_NAME"
echo "  Thinking tokens:         $THINKING_TOKENS"
echo "  Action tokens:           $ACTION_TOKENS"
echo "  Episodes per GPU:        $EPISODES_PER_GPU"
echo "  Total episodes/group:    $((NUM_GPUS * EPISODES_PER_GPU))"
echo "  Inner epochs:            $NUM_INNER_EPOCHS"
echo "  Minibatch size:          $MINIBATCH_SIZE"
echo "  Micro-batch size:        $SAMPLES_PER_MICRO_BATCH"
echo "  Total episodes:          $TOTAL_EPISODES"
echo "  Max env steps:           $MAX_ENV_STEPS"
echo "  Num agents:              $NUM_AGENTS"
echo "  Eat reward:              $EAT_REWARD"
echo "  Clean reward:            $CLEAN_REWARD"
echo "  Learning rate:           $LEARNING_RATE"
echo "  Loss type:               $LOSS_TYPE"
echo "  Output dir:              $OUTPUT_DIR"
echo "  Wandb enabled:           $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb project:           $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ]   && echo "  Wandb entity:            $WANDB_ENTITY"
    [ -n "$WANDB_RUN_NAME" ] && echo "  Wandb run name:          $WANDB_RUN_NAME"
fi
echo "  DeepSpeed ZeRO-3:        $USE_DEEPSPEED"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────────
echo "GPU Information:"
nvidia-smi
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Build optional flag strings ────────────────────────────────────────────
TWO_STAGE_FLAG=""
if [ "$ACTION_MODE" = "text" ]; then
    if [ "$USE_TWO_STAGE" = true ]; then
        TWO_STAGE_FLAG="--use_two_stage"
    else
        TWO_STAGE_FLAG="--no_two_stage"
    fi
fi

DEEPSPEED_FLAG=""
if [ "$USE_DEEPSPEED" = true ]; then
    DEEPSPEED_FLAG="--use_deepspeed"
fi

WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ]   && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
    [ -n "$WANDB_RUN_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
else
    WANDB_ARGS="--no_wandb"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
echo "Launching accelerate with $NUM_GPUS GPUs..."
echo ""

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    $DEEPSPEED_FLAG \
    --zero_stage 3 \
    --offload_optimizer_device cpu \
    --offload_param_device cpu \
    --no_python \
    python $SCRIPT_DIR/grpo_textgame.py \
    --action_mode "$ACTION_MODE" \
    $TWO_STAGE_FLAG \
    --logprob_mode "$LOGPROB_MODE" \
    --model_name "$MODEL_NAME" \
    --thinking_tokens $THINKING_TOKENS \
    --action_tokens $ACTION_TOKENS \
    --num_episodes $TOTAL_EPISODES \
    --episodes_per_gpu $EPISODES_PER_GPU \
    --max_env_steps $MAX_ENV_STEPS \
    --num_agents $NUM_AGENTS \
    --learning_rate $LEARNING_RATE \
    --loss_type "$LOSS_TYPE" \
    --num_inner_epochs $NUM_INNER_EPOCHS \
    --minibatch_size $MINIBATCH_SIZE \
    --micro_batch_size $SAMPLES_PER_MICRO_BATCH \
    --eat_reward $EAT_REWARD \
    --clean_reward $CLEAN_REWARD \
    --output_dir "$OUTPUT_DIR" \
    --use_accelerate \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --skip_pre_eval \
    --seed 42 \
    $DEEPSPEED_FLAG \
    $WANDB_ARGS \
    2>&1 | tee "$OUTPUT_DIR/training_log_${SLURM_JOB_ID:-local}.txt"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Log saved to: $OUTPUT_DIR/training_log_${SLURM_JOB_ID:-local}.txt"
echo "SLURM output: grpo_textgame_${SLURM_JOB_ID:-local}.out"
echo "=========================================="
