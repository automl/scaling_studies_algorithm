#!/bin/bash

#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --job-name=gen_cache
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/gen_cache/%x-%a-%j.log
#SBATCH --array=0-17

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

optimizers=(
    "random_search"
    "random_search_with_priors"
    "chinchilla_approach1"
    "chinchilla_approach1_with_priors"
    "chinchilla_approach2"
    "chinchilla_approach2_with_priors"
    "carbs"
    "carbs_with_priors"
    "bo_left_budget"
)

benchmarks=("openclip" "llm")

n_benchmarks=${#benchmarks[@]}
optimizer_idx=$(( SLURM_ARRAY_TASK_ID / n_benchmarks ))
benchmark_idx=$(( SLURM_ARRAY_TASK_ID % n_benchmarks ))

optimizer=${optimizers[$optimizer_idx]}
benchmark=${benchmarks[$benchmark_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer  benchmark=$benchmark"

python -m src.plotting.pred_cache_gen --optimizer "$optimizer" --benchmark "$benchmark"
