#!/bin/bash

#SBATCH --partition=testdlc2_gpu-h200
#SBATCH --array=0-21%11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --job-name=openclip_scan
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%A_%a.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(bo_space_expansion_modified_jes bo_left_budget_modified_jes)
BENCHMARK=openclip

NUM_OPTIMIZERS=${#OPTIMIZERS[@]}  # 2

seed_idx=$(( SLURM_ARRAY_TASK_ID / NUM_OPTIMIZERS ))
opt_idx=$(( SLURM_ARRAY_TASK_ID % NUM_OPTIMIZERS ))

seed=${SEEDS[$seed_idx]}
optimizer=${OPTIMIZERS[$opt_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer seed=$seed"

python -m src.run --optimizer $optimizer --benchmark $BENCHMARK --seed $seed
