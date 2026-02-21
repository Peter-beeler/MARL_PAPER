#!/bin/bash
#SBATCH --job-name=grpo_multi_gpu
#SBATCH --account=PAS2138
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=quad
#SBATCH --output=grpo_multi_gpu_%j_drgrpo_noobscast_actionlog.out
#SBATCH --error=grpo_multi_gpu_%j_drgrpo_noobscast_actionlog.err

#
# SLURM batch script for multi-GPU GRPO training with text actions
#
# Usage:
#   sbatch launch_multi_gpu_slurm.sh
#
# Configuration:
#   - To customize GPUs, edit the #SBATCH --gpus-per-node directive above
#   - To enable/disable wandb logging, set USE_WANDB=true/false below
#   - To configure wandb project/entity, edit WANDB_PROJECT and WANDB_ENTITY below
#   - To adjust memory usage, modify SAMPLES_PER_MICRO_BATCH (1-2 for A100, lower if OOM)
#   - To change log probability mode, set LOGPROB_MODE="action" or "action+thinking"
#
source ~/.bashrc
conda activate py311LLM
module load cuda/12.4.1
# Number of GPUs to use (reads from SLURM allocation)
NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}

echo "=========================================="
echo "GRPO Multi-GPU Training (SLURM Job)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $NUM_GPUS"
echo "Working directory: $(pwd)"
echo ""

# Set environment variables for better distributed training
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings in multi-process
export NCCL_DEBUG=WARN  # Set to INFO for more verbose NCCL debugging
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export OMP_NUM_THREADS=4  # Control OpenMP threads
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Fix CUDA_HOME to match actual CUDA installation
NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "Setting CUDA_HOME to: $CUDA_HOME"
fi

# Model and training configuration
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME="Qwen/Qwen3-4B-Base"
EPISODES_PER_GPU=2 # Each GPU will collect this many episodes per group
TOTAL_EPISODES=1024
MAX_ENV_STEPS=10
NUM_AGENTS=1
LEARNING_RATE=1e-6
THINKING_TOKENS=256
ACTION_TOKENS=128
LOGPROB_MODE="action"  # "action" (only action tokens) or "action+thinking" (both thinking and action tokens)

# Inner epoch optimization (PPO-style)
NUM_INNER_EPOCHS=4         # Number of optimization epochs per group
MINIBATCH_SIZE=8           # Number of trajectories per mini-batch
SAMPLES_PER_MICRO_BATCH=10  # Number of (prompt, action) samples per micro-batch for gradient accumulation
                           # Adjust based on GPU memory: 1-2 for A100, increase if memory allows

EAT_REWARD=1.0           # Reward for eating an apple (default: 1.0)
CLEAN_REWARD=0.2         # Reward for cleaning a dirt tile (0.0 = disabled, e.g. 0.1 to ease cold start)

OUTPUT_DIR="./grpo_multi_gpu_checkpoints"
NUM_EVAL_EPISODES=8

# Wandb configuration
# Note: Make sure you have logged in with 'wandb login' before running this script
USE_WANDB=true  # Set to false to disable wandb logging
WANDB_PROJECT="grpo"  # Wandb project name
WANDB_ENTITY=""  # Your wandb username/team (leave empty for default)
WANDB_RUN_NAME=""  # Run name (leave empty for auto-generated)

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Episodes per GPU: $EPISODES_PER_GPU"
echo "  Total episodes per group: $((NUM_GPUS * EPISODES_PER_GPU))"
echo "  Inner epochs: $NUM_INNER_EPOCHS"
echo "  Minibatch size: $MINIBATCH_SIZE"
echo "  Samples per micro-batch: $SAMPLES_PER_MICRO_BATCH"
echo "  Total updates per group: $(( (NUM_GPUS * EPISODES_PER_GPU + MINIBATCH_SIZE - 1) / MINIBATCH_SIZE * NUM_INNER_EPOCHS ))"
echo "  Total episodes: $TOTAL_EPISODES"
echo "  Max env steps: $MAX_ENV_STEPS"
echo "  Eat reward: $EAT_REWARD"
echo "  Clean reward: $CLEAN_REWARD"
echo "  Num agents: $NUM_AGENTS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Thinking tokens: $THINKING_TOKENS"
echo "  Log prob mode: $LOGPROB_MODE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Wandb enabled: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb project: $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && echo "  Wandb entity: $WANDB_ENTITY"
    [ -n "$WANDB_RUN_NAME" ] && echo "  Wandb run name: $WANDB_RUN_NAME"
fi
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Launch with accelerate
echo "Launching accelerate with $NUM_GPUS GPUs..."
echo ""

# Build wandb arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
    [ -n "$WANDB_RUN_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
else
    WANDB_ARGS="--no_wandb"
fi

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    grpo_text_compound.py \
    --model_name "$MODEL_NAME" \
    --num_episodes $TOTAL_EPISODES \
    --episodes_per_gpu $EPISODES_PER_GPU \
    --max_env_steps $MAX_ENV_STEPS \
    --num_agents $NUM_AGENTS \
    --learning_rate $LEARNING_RATE \
    --thinking_tokens $THINKING_TOKENS \
    --action_tokens $ACTION_TOKENS \
    --num_inner_epochs $NUM_INNER_EPOCHS \
    --minibatch_size $MINIBATCH_SIZE \
    --micro_batch_size $SAMPLES_PER_MICRO_BATCH \
    --output_dir "$OUTPUT_DIR" \
    --use_accelerate \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --seed 42 \
    --loss_type "drgrpo" \
    --eat_reward $EAT_REWARD \
    --clean_reward $CLEAN_REWARD \
    --skip_pre_eval \
    $WANDB_ARGS \
    2>&1 | tee "$OUTPUT_DIR/training_log_${SLURM_JOB_ID}.txt"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Log saved to: $OUTPUT_DIR/training_log_${SLURM_JOB_ID}.txt"
echo "SLURM output: grpo_multi_gpu_${SLURM_JOB_ID}.out"
echo "=========================================="
