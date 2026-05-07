#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=3llm_c
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%A_%a.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

benchmark="${1:-llm}"

# Define the parameter space
seeds=({40..50})

commands=(
    # "--optimizer carbs"
    "--optimizer carbs --use-priors"
)
    # "--optimizer chinchilla_approach2"
    # "--optimizer chinchilla_approach1"
    # "--optimizer chinchilla_approach1 --use-priors"
    # "--optimizer random_search"
    # "--optimizer random_search --use-priors"
    # "--optimizer chinchilla_approach2 --use-priors"

# Calculate indices based on the SLURM_ARRAY_TASK_ID
# 8 commands per seed, so we divide by 8 for the seed, and modulo 8 for the command
seed_idx=$((SLURM_ARRAY_TASK_ID / 1))
cmd_idx=$((SLURM_ARRAY_TASK_ID % 1))

current_seed=${seeds[$seed_idx]}
current_cmd=${commands[$cmd_idx]}

echo "Running task ID $SLURM_ARRAY_TASK_ID"
echo "Benchmark: $benchmark"
echo "Seed: $current_seed"
echo "Command args: $current_cmd"

# Run the single specific command for this job (no '&' or 'wait' needed)
python -m src.run --benchmark $benchmark --seed $current_seed $current_cmd