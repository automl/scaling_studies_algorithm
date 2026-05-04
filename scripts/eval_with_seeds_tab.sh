#!/bin/bash

#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=llm1
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%A_%a.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

echo "Running task ID $SLURM_ARRAY_TASK_ID"
# python -m src.plotting.plot

# for seed in 40 41 42 43 44 45 46 47 48 49 50; do
# for seed in 40 41 42 43 44 45; do
python -m src.run --optimizer grid_search --benchmark tabpfn --seed 40 
# python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed &
# python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  &
# python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  &
# python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed &
# python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed  --use-priors &
# done

# wait
# echo "All runs completed"
