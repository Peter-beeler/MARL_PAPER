#!/bin/bash
#SBATCH --job-name=diag_V1
#SBATCH --account=PAS2138
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition=nextgen
#SBATCH --output=diag_V1_%j.out
#SBATCH --error=diag_V1_%j.err

#
# Diagnostic: V1 (roletoken) checkpoint, 20 episodes, full logging
# Logs every step: map, observations, prompts, LLM outputs, actions, rewards
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

V1_CKPT="./grpo_textgame_checkpoints_compound_role_ascend_sbatch_1e-6_roletokenprob_V1/checkpoint_ep960"

echo "=========================================================================="
echo "DIAGNOSTIC EVALUATION — V1 roletoken (Job ${SLURM_JOB_ID:-local})"
echo "=========================================================================="
echo "Checkpoint: $V1_CKPT"
echo ""
nvidia-smi
echo ""

python "$SCRIPT_DIR/diagnostic_eval.py" \
    --checkpoint "$V1_CKPT" \
    --base_model "Qwen/Qwen3-4B-Instruct-2507" \
    --num_agents 3 \
    --max_env_steps 50 \
    --seed 42 \
    --num_eval_episodes 20 \
    --output "diagnostic_V1_${SLURM_JOB_ID:-local}.log"

echo ""
echo "=========================================================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================================================="
