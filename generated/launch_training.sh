#!/bin/bash
#SBATCH --job-name=grpo_cleanup
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the generated files directory
cd "$(dirname "$0")"

# Activate conda environment if needed
# source activate your_env

# Run with accelerate for single GPU
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    train.py \
    --num_episodes 10 \
    --num_agents 1
