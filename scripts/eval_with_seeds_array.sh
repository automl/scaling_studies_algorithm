#!/bin/bash

#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --job-name=bo
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=0-10

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

# Map SLURM_ARRAY_TASK_ID (0-10) to seeds (40-50)
seed=$((40 + SLURM_ARRAY_TASK_ID))

benchmark="${1:-llm}"

echo "Running seed: $seed (array task: $SLURM_ARRAY_TASK_ID)"

# python -m src.run --optimizer bo_left_budget --benchmark $benchmark --seed $seed
python -m src.run --optimizer bo_left_budget_ei --benchmark $benchmark --seed $seed --use-priors

#python -m src.run --optimizer bo_space_expansion --benchmark llm --seed 40