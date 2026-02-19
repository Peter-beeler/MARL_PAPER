#!/bin/bash
#SBATCH --job-name=grpo_cleanup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # <--- CHANGE THIS to your cluster's partition name

# ============================================================
# SETUP
# ============================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (Adjust these lines based on your cluster's specific modules)
# module load cuda/12.1
# module load python/3.10

# Activate your python environment
# source venv/bin/activate
# OR
# conda activate grpo_env

# Set environment variables
export OMP_NUM_THREADS=4
export WANDB_PROJECT="cleanup_grpo"
# export WANDB_API_KEY="your_key_here" # Uncomment and set if using WandB
# export HF_TOKEN="your_token_here"    # Uncomment if accessing gated models

# Print debug info
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ============================================================
# TRAINING COMMAND
# ============================================================

# We use 'accelerate launch' to handle device placement and mixed precision.
# Even for 1 GPU, this ensures the environment is set up correctly for the script.

accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    train.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --output_dir "./checkpoints/cleanup_run_v1" \
    --num_episodes 500 \
    --num_agents 1 \
    --learning_rate 1e-5 \
    --episodes_per_update 4 \
    --use_wandb

echo "Training finished at $(date)"