#!/bin/bash
#SBATCH --job-name=cmp_base
#SBATCH --account=PAS2138
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition=nextgen
#SBATCH --output=cmp_base_%j.out
#SBATCH --error=cmp_base_%j.err

#
# Job 1: Base (untuned) model only — 3 configs: text, compound, compound+role
#

source ~/.bashrc
conda activate py311LLM
SCRIPT_DIR="/users/PAS2056/mypeter8219/Research/LLM_MARL/cleanup_code"
cd "$SCRIPT_DIR"
module load cuda/12.4.1

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

BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
NUM_AGENTS=3
MAX_ENV_STEPS=50
SEED=42
NUM_EVAL_EPISODES=20
PARALLEL_ENVS=4

echo "=========================================================================="
echo "BASE MODEL COMPARISON (Job ${SLURM_JOB_ID:-local})"
echo "=========================================================================="
echo "3 configs: text, compound, compound+role"
echo ""
nvidia-smi
echo ""

# ── 1. Text mode, no role ────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  RUN 1/3: base untuned — text, norole"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

python "$SCRIPT_DIR/model_comparison.py" \
    --checkpoints "$BASE_MODEL" \
    --labels "base (text, norole)" \
    --action_mode text \
    --use_two_stage \
    --base_model "$BASE_MODEL" \
    --num_agents $NUM_AGENTS \
    --max_env_steps $MAX_ENV_STEPS \
    --seed $SEED \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --parallel_envs $PARALLEL_ENVS \
    --role_assignment_interval 0

# ── 2. Compound mode, no role ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  RUN 2/3: base untuned — compound, norole"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

python "$SCRIPT_DIR/model_comparison.py" \
    --checkpoints "$BASE_MODEL" \
    --labels "base (compound, norole)" \
    --action_mode compound \
    --base_model "$BASE_MODEL" \
    --num_agents $NUM_AGENTS \
    --max_env_steps $MAX_ENV_STEPS \
    --seed $SEED \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --parallel_envs $PARALLEL_ENVS \
    --role_assignment_interval 0

# ── 3. Compound mode, with role ──────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  RUN 3/3: base untuned — compound, role"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

python "$SCRIPT_DIR/model_comparison.py" \
    --checkpoints "$BASE_MODEL" \
    --labels "base (compound, role)" \
    --action_mode compound \
    --base_model "$BASE_MODEL" \
    --num_agents $NUM_AGENTS \
    --max_env_steps $MAX_ENV_STEPS \
    --seed $SEED \
    --num_eval_episodes $NUM_EVAL_EPISODES \
    --parallel_envs $PARALLEL_ENVS \
    --role_assignment_interval 10

echo ""
echo "=========================================================================="
echo "BASE MODEL COMPARISON COMPLETE"
echo "=========================================================================="
