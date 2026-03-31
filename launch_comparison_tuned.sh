#!/bin/bash
#SBATCH --job-name=cmp_tuned
#SBATCH --account=PAS2138
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition=nextgen
#SBATCH --output=cmp_tuned_%j.out
#SBATCH --error=cmp_tuned_%j.err

#
# Job 2: 3 tuned checkpoints — compound + role only
#
#   V1  — trained with role + role token prob in loss
#   V3  — trained with role but NOT in loss
#   try2 — trained without role
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

V1_CKPT="./grpo_textgame_checkpoints_compound_role_ascend_sbatch_1e-6_roletokenprob_V1/checkpoint_ep960"
V3_CKPT="./grpo_textgame_checkpoints_compound_role_ascend_sbatch_1e-6_V3/checkpoint_ep960"
TRY2_CKPT="./grpo_textgame_checkpoints_compound_norole_ascend_sbatch_1e-6_try2/checkpoint_ep960"

echo "=========================================================================="
echo "TUNED MODELS COMPARISON — compound + role (Job ${SLURM_JOB_ID:-local})"
echo "=========================================================================="
echo "3 models: V1 (role+tokenprob), V3 (role), try2 (norole)"
echo "All evaluated with: compound mode + role_interval=10"
echo ""
echo "Checkpoints:"
echo "  V1:   $V1_CKPT"
echo "  V3:   $V3_CKPT"
echo "  try2: $TRY2_CKPT"
echo ""
nvidia-smi
echo ""

python "$SCRIPT_DIR/model_comparison.py" \
    --checkpoints "$V1_CKPT" "$V3_CKPT" "$TRY2_CKPT" \
    --labels "V1_roletoken" "V3_role" "try2_norole" \
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
echo "TUNED MODELS COMPARISON COMPLETE"
echo "=========================================================================="
