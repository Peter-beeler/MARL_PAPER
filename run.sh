#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1 --ntasks-per-node=8  --gpus-per-node=1
#SBATCH --job-name=zeroshot_exp
#SBATCH --partition=nextgen
#SBATCH --account=PAS2138

source ~/.bashrc
conda activate py311LLM
module load cuda/12.4.1
srun python3 ./archive/zeroshot.py

