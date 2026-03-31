#!/bin/bash
#SBATCH --job-name=model_compare
#SBATCH --account=PAS2138
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition=nextgen
#SBATCH --output=model_compare_%j.out
#SBATCH --error=model_compare_%j.err

#
# Full ablation comparison: 4 models × 2 action_modes × 2 role_settings = 16 runs
#
# Models:
#   1. base (untuned Qwen3-4B-Instruct)
#   2. V1  — trained with role + role token prob in loss
#   3. V3  — trained with role but NOT in loss
#   4. try2 — trained without role
#
# Action modes:  text, compound
# Role settings: norole (interval=0), role (interval=10)
#
# Usage:
#   sbatch launch_comparison.sh
#

source ~/.bashrc
conda activate py311LLM
SCRIPT_DIR="/users/PAS2056/mypeter8219/Research/LLM_MARL/cleanup_code"
cd "$SCRIPT_DIR"
module load cuda/12.4.1

# ── Environment variables ────────────────────────────────────────────────
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

# ── Shared settings ──────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
THINKING_TOKENS=256
ACTION_TOKENS=128
NUM_AGENTS=3
MAX_ENV_STEPS=50
EAT_REWARD=1.0
SEED=42
NUM_EVAL_EPISODES=20
PARALLEL_ENVS=4

# ── Checkpoint paths (latest ep960 from each) ───────────────────────────
V1_CKPT="./grpo_textgame_checkpoints_compound_role_ascend_sbatch_1e-6_roletokenprob_V1/checkpoint_ep960"
V3_CKPT="./grpo_textgame_checkpoints_compound_role_ascend_sbatch_1e-6_V3/checkpoint_ep960"
TRY2_CKPT="./grpo_textgame_checkpoints_compound_norole_ascend_sbatch_1e-6_try2/checkpoint_ep960"

# ── Print header ─────────────────────────────────────────────────────────
echo "=========================================================================="
echo "FULL ABLATION COMPARISON (SLURM Job ${SLURM_JOB_ID:-local})"
echo "=========================================================================="
echo "Node:          ${SLURM_NODELIST:-localhost}"
echo "Base model:    $BASE_MODEL"
echo "Num agents:    $NUM_AGENTS"
echo "Max env steps: $MAX_ENV_STEPS"
echo "Eval episodes: $NUM_EVAL_EPISODES"
echo "Seed:          $SEED"
echo ""
echo "Checkpoints:"
echo "  V1 (role+tokenprob): $V1_CKPT"
echo "  V3 (role, no token): $V3_CKPT"
echo "  try2 (norole):       $TRY2_CKPT"
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# ── Helper function ──────────────────────────────────────────────────────
run_comparison() {
    local ACTION_MODE="$1"
    local ROLE_INTERVAL="$2"
    local ROLE_DESC="$3"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  RUN: action_mode=${ACTION_MODE}, role=${ROLE_DESC}"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    TWO_STAGE_FLAG=""
    if [ "$ACTION_MODE" = "text" ]; then
        TWO_STAGE_FLAG="--use_two_stage"
    fi

    python "$SCRIPT_DIR/model_comparison.py" \
        --checkpoints "$V1_CKPT" "$V3_CKPT" "$TRY2_CKPT" \
        --labels "V1_roletoken" "V3_role" "try2_norole" \
        --include_base \
        --action_mode "$ACTION_MODE" \
        $TWO_STAGE_FLAG \
        --base_model "$BASE_MODEL" \
        --thinking_tokens $THINKING_TOKENS \
        --action_tokens $ACTION_TOKENS \
        --num_agents $NUM_AGENTS \
        --max_env_steps $MAX_ENV_STEPS \
        --eat_reward $EAT_REWARD \
        --seed $SEED \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --parallel_envs $PARALLEL_ENVS \
        --role_assignment_interval "$ROLE_INTERVAL"

    echo ""
    echo "── Done: action_mode=${ACTION_MODE}, role=${ROLE_DESC} ──"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════
# Run all 4 combinations
# ══════════════════════════════════════════════════════════════════════════

# 1. text mode, no role
run_comparison "text" 0 "norole"

# 2. text mode, with role
run_comparison "text" 10 "role"

# 3. compound mode, no role
run_comparison "compound" 0 "norole"

# 4. compound mode, with role
run_comparison "compound" 10 "role"

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================================================="
echo "ALL COMPARISONS COMPLETE"
echo "=========================================================================="
echo ""
echo "Summary of runs:"
echo "  1. text    + norole  → 4 models (base, V1, V3, try2)"
echo "  2. text    + role    → 4 models (base, V1, V3, try2)"
echo "  3. compound + norole → 4 models (base, V1, V3, try2)"
echo "  4. compound + role   → 4 models (base, V1, V3, try2)"
echo "  Total: 16 configurations evaluated"
echo ""
echo "Log: model_compare_${SLURM_JOB_ID:-local}.out"
echo "=========================================================================="
