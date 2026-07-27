#!/bin/bash

#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --job-name=tab_acq
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=19-43
#SBATCH --time=03:00:00

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(chinchilla_approach1 chinchilla_approach2 carbs random_search)

NUM_OPTIMIZERS=${#OPTIMIZERS[@]}

seed_idx=$(( SLURM_ARRAY_TASK_ID / NUM_OPTIMIZERS ))
opt_idx=$(( SLURM_ARRAY_TASK_ID % NUM_OPTIMIZERS ))

seed=${SEEDS[$seed_idx]}
optimizer=${OPTIMIZERS[$opt_idx]}

benchmark="${1:-tabpfn}"

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer seed=$seed benchmark=$benchmark"

python -m src.run --optimizer "$optimizer" --benchmark "$benchmark" --seed "$seed" --pred tabpfn