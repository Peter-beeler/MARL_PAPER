#!/usr/bin/env bash
# =============================================================================
# launch_training.sh — Launch GRPO training for CleanupEnvMove
#
# Environment: VESSL AI Cloud — 4x NVIDIA A100-SXM4-80GB (80GB VRAM each)
#              PyTorch 2.9.1+cu128 | CUDA 12.8 | Accelerate 1.12.0
#              No SLURM (cloud workspace, direct execution)
#
# Usage:
#   bash launch_training.sh           # full training run
#   bash launch_training.sh --debug   # debug run (10 episodes)
#
# SLURM-style job script included as comments for HPC environments.
# =============================================================================

# --- SLURM directives (for HPC clusters with SLURM) -------------------------
#SBATCH --job-name=grpo_cleanup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
# ----------------------------------------------------------------------------

set -euo pipefail

# ============================================================
# GPU ENVIRONMENT CHECK
# ============================================================
echo "=================================================="
echo "GPU Environment Check"
echo "=================================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

# Check NVIDIA GPU status
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s: %s | VRAM: %s MB total, %s MB free | Temp: %s°C | Util: %s%%\n", \
                        $1, $2, $3, $4, $5, $6}' || echo "  nvidia-smi not available"

echo ""
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "Detected GPUs: $NUM_GPUS"
echo ""

# Check PyTorch / CUDA
echo "--- Python Environment ---"
PYTHON_BIN=$(which python3 || which python)
echo "Python: $PYTHON_BIN"
$PYTHON_BIN -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)')
" 2>/dev/null || echo "PyTorch check failed"

echo ""
echo "--- Accelerate ---"
ACCELERATE_BIN=/root/.local/workspace/python-packages/bin/accelerate
if [ -f "$ACCELERATE_BIN" ]; then
    $PYTHON_BIN -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null
else
    ACCELERATE_BIN=$(which accelerate 2>/dev/null || echo "accelerate")
fi
echo "Accelerate binary: $ACCELERATE_BIN"
echo "=================================================="
echo ""

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

# Output directories (relative to project root)
OUTPUT_DIR="$SCRIPT_DIR/grpo_checkpoints"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Project dir : $PROJECT_DIR"
echo "Train script: $TRAIN_SCRIPT"
echo "Output dir  : $OUTPUT_DIR"
echo ""

# ============================================================
# TRAINING PARAMETERS
# ============================================================
# Model: Qwen3-4B-Instruct (configured in train.py)
# Hardware: 4x A100 80GB → can fit large batch comfortably

# Check for debug mode
DEBUG_MODE=false
NUM_EPISODES=1000
MAX_ENV_STEPS=30
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_MODE=true
    NUM_EPISODES=10
    MAX_ENV_STEPS=10
    echo "=== DEBUG MODE: num_episodes=10, max_env_steps=10 ==="
    echo ""
fi

# Accelerate config for 4-GPU data-parallel training
# episodes_per_gpu=4 → 4 GPUs × 4 eps = 16 total per group
EPISODES_PER_GPU=4
THINKING_TOKENS=384

# ============================================================
# ACCELERATE LAUNCH
# ============================================================
echo "Starting training..."
echo "  Mode         : $([ "$DEBUG_MODE" = true ] && echo DEBUG || echo FULL)"
echo "  Episodes     : $NUM_EPISODES"
echo "  Max env steps: $MAX_ENV_STEPS"
echo "  GPUs         : $NUM_GPUS"
echo "  Episodes/GPU : $EPISODES_PER_GPU"
echo ""

cd "$PROJECT_DIR"

$PYTHON_BIN -m accelerate.commands.launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    --multi_gpu \
    "$TRAIN_SCRIPT" \
    --num_episodes "$NUM_EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    --episodes_per_gpu "$EPISODES_PER_GPU" \
    --thinking_tokens "$THINKING_TOKENS" \
    --max_env_steps "$MAX_ENV_STEPS" \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=================================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "Logs saved to: $LOG_DIR/"
echo "Checkpoints: $OUTPUT_DIR/"
echo "=================================================="
exit $EXIT_CODE
