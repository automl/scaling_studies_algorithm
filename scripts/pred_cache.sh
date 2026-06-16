#!/bin/bash

#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --job-name=gen_cache
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/gen_cache/%x-%a-%j.log
#SBATCH --array=0-13

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

optimizers=(
    "bo_space_expansion_mes"
    "bo_space_expansion_ei"
    "bo_space_expansion_jes"
    "bo_left_budget_ei"
    "bo_left_budget_mes"
    "bo_left_budget_jes"
    "bo_left_budget_zero_mean_ei"
)

benchmarks=("openclip" "llm")

n_benchmarks=${#benchmarks[@]}
optimizer_idx=$(( SLURM_ARRAY_TASK_ID / n_benchmarks ))
benchmark_idx=$(( SLURM_ARRAY_TASK_ID % n_benchmarks ))

optimizer=${optimizers[$optimizer_idx]}
benchmark=${benchmarks[$benchmark_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer  benchmark=$benchmark"

python -m src.plotting.pred_cache_gen --optimizer "$optimizer" --benchmark "$benchmark"
