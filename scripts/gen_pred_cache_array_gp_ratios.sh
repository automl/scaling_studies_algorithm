#!/bin/bash

#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --job-name=gen_pred_cache
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=0-21

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

benchmark="llm"
optimizers=(bo_left_budget_modified_jes bo_left_budget_jes)

# Map SLURM_ARRAY_TASK_ID (0-21) to optimizer x seed (40-50).
# Fitting is target_flop-independent, so there's no ratio axis to sweep anymore
# (see src/plotting/pred_cache_gen.py) -- each (optimizer, seed) is warmed once.
n_optimizers=${#optimizers[@]}

optimizer_idx=$((SLURM_ARRAY_TASK_ID % n_optimizers))
seed=$((40 + SLURM_ARRAY_TASK_ID / n_optimizers))

optimizer="${optimizers[$optimizer_idx]}"

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer  benchmark=$benchmark  seed=$seed"

python -m src.plotting.pred_cache_gen \
    --optimizer "$optimizer" \
    --benchmark "$benchmark" \
    --seed "$seed"
