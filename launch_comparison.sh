#!/bin/bash
#SBATCH --job-name=model_compare
#SBATCH --account=PAS2138
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition=quad
#SBATCH --output=model_compare_%j.out
#SBATCH --error=model_compare_%j.err

#
# SLURM batch script for model_comparison.py
#
# Usage:
#   sbatch launch_comparison.sh
#
# Loads each checkpoint one at a time on a single GPU, runs 20 eval episodes
# on the same fixed initial states, and prints a summary table.
#
# ── COMPARISON MODE ──────────────────────────────────────────────────────
# Set COMPARE_MODE to control which checkpoints to compare:
#
#   "best"       — best_model from each run  (quick cross-run comparison)
#   "milestones" — specific episode checkpoints you pick from each run
#   "all"        — every checkpoint_ep* in every run dir (thorough but slow)
#

source ~/.bashrc
conda activate py311LLM
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
module load cuda/12.4.1

# ── Environment variables (matching launch_textgame.sh) ──────────────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TF_CPP_MIN_LOG_LEVEL=3
export TRITON_CACHE_DIR=/tmp/triton_cache_$$

NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
fi

# ── Action mode (must match the mode used during training) ────────────────
ACTION_MODE="compound"          # "text" or "compound"
USE_TWO_STAGE=true              # text mode only

# ── Model ─────────────────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
THINKING_TOKENS=256
ACTION_TOKENS=128

# ── Environment (must match training settings) ───────────────────────────
NUM_AGENTS=3
MAX_ENV_STEPS=30
EAT_REWARD=1.0
CLEAN_REWARD=0.2
SEED=42

# ── Evaluation ────────────────────────────────────────────────────────────
NUM_EVAL_EPISODES=20
PARALLEL_ENVS=4                 # envs batched per generate() call
INCLUDE_BASE=true               # also evaluate the untrained base model

# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINT SELECTION
# ══════════════════════════════════════════════════════════════════════════

# All training-run directories to consider
RUN_DIRS=(
    "./grpo_textgame_checkpoints_compound_dirt0"
    "./grpo_textgame_checkpoints_compound_dirt0.2"
)

# Which checkpoints to pick from each run dir:
#   "best"       — only best_model from each run
#   "milestones" — best_model + checkpoints at MILESTONE_EPISODES
#   "all"        — best_model + every checkpoint_ep*
COMPARE_MODE="best"

# Episodes to include when COMPARE_MODE="milestones"
MILESTONE_EPISODES=(256 512 768)

# ══════════════════════════════════════════════════════════════════════════
# AUTO-DISCOVERY  (no need to edit below this line)
# ══════════════════════════════════════════════════════════════════════════

CHECKPOINTS=()
LABELS=()

for run_dir in "${RUN_DIRS[@]}"; do
    [ ! -d "$run_dir" ] && echo "WARNING: Run dir not found, skipping: $run_dir" && continue

    # Short name for this run (strip leading ./)
    run_name=$(basename "$run_dir")

    case "$COMPARE_MODE" in
        best)
            if [ -d "${run_dir}/best_model" ]; then
                CHECKPOINTS+=("${run_dir}/best_model")
                LABELS+=("${run_name}/best")
            fi
            ;;

        milestones)
            # best_model first
            if [ -d "${run_dir}/best_model" ]; then
                CHECKPOINTS+=("${run_dir}/best_model")
                LABELS+=("${run_name}/best")
            fi
            # then specific milestones
            for ep in "${MILESTONE_EPISODES[@]}"; do
                ckpt="${run_dir}/checkpoint_ep${ep}"
                if [ -d "$ckpt" ]; then
                    CHECKPOINTS+=("$ckpt")
                    LABELS+=("${run_name}/ep${ep}")
                fi
            done
            ;;

        all)
            # best_model first
            if [ -d "${run_dir}/best_model" ]; then
                CHECKPOINTS+=("${run_dir}/best_model")
                LABELS+=("${run_name}/best")
            fi
            # then all checkpoint_ep* sorted numerically
            for ckpt in $(ls -d "${run_dir}"/checkpoint_ep* 2>/dev/null | sort -t p -k 2 -n); do
                ep_num=$(basename "$ckpt" | sed 's/checkpoint_ep//')
                CHECKPOINTS+=("$ckpt")
                LABELS+=("${run_name}/ep${ep_num}")
            done
            ;;

        *)
            echo "ERROR: Unknown COMPARE_MODE='$COMPARE_MODE'. Use best, milestones, or all."
            exit 1
            ;;
    esac
done

# ── Print config ─────────────────────────────────────────────────────────
echo "=========================================="
echo "Model Comparison (SLURM Job)"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-localhost}"
echo "Action mode:   $ACTION_MODE"
echo "Base model:    $BASE_MODEL"
echo "Num agents:    $NUM_AGENTS"
echo "Max env steps: $MAX_ENV_STEPS"
echo "Eval episodes: $NUM_EVAL_EPISODES"
echo "Compare mode:  $COMPARE_MODE"
echo "Include base:  $INCLUDE_BASE"
echo ""
echo "Checkpoints (${#CHECKPOINTS[@]} total):"
for i in "${!CHECKPOINTS[@]}"; do
    echo "  [${LABELS[$i]}] ${CHECKPOINTS[$i]}"
done
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

if [ ${#CHECKPOINTS[@]} -eq 0 ] && [ "$INCLUDE_BASE" != true ]; then
    echo "ERROR: No checkpoints found and --include_base is not set."
    exit 1
fi

# ── Build flags ──────────────────────────────────────────────────────────
TWO_STAGE_FLAG=""
if [ "$ACTION_MODE" = "text" ]; then
    if [ "$USE_TWO_STAGE" = true ]; then
        TWO_STAGE_FLAG="--use_two_stage"
    else
        TWO_STAGE_FLAG="--no_two_stage"
    fi
fi

INCLUDE_BASE_FLAG=""
if [ "$INCLUDE_BASE" = true ]; then
    INCLUDE_BASE_FLAG="--include_base"
fi

# ── Launch ───────────────────────────────────────────────────────────────
echo "Launching model comparison..."
echo ""

python $SCRIPT_DIR/model_comparison.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --labels "${LABELS[@]}" \
    $INCLUDE_BASE_FLAG \
    --action_mode "$ACTION_MODE" \
    $TWO_STAGE_FLAG \
    --base_model "$BASE_MODEL" \
    --thinking_tokens $THINKING_TOKENS \
    --action_tokens $ACTION_TOKENS \
    --num_agents $NUM_AGENTS \
    --max_env_steps $MAX_ENV_STEPS \
    --eat_reward $EAT_REWARD \
    --clean_reward $CLEAN_REWARD \
    --seed $SEED \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --parallel_envs $PARALLEL_ENVS \
    2>&1 | tee "model_compare_${SLURM_JOB_ID:-local}.log"

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Log: model_compare_${SLURM_JOB_ID:-local}.log"
echo "=========================================="
