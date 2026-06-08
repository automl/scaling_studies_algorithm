#!/bin/bash

#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=llm
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%j.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

benchmark="${1:-llm}"
# seed="${2:-40}"

for seed in 40..50; do
    python -m src.run --optimizer bo_space_expansion --benchmark $benchmark --seed $seed &
# python -m src.run --optimizer grid_search --benchmark $benchmark --seed $seed 
# python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed &
# python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  &
# python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  &
# python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  --use-priors &
# python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed &
# python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed  --use-priors &
# done

wait
echo "All runs completed"
