#!/bin/bash

#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --mem=140G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=eval_with_seeds
#SBATCH --output=%x-%j.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

benchmark="${1:-openclip}"


for seed in 40; do
    # python -m src.run --optimizer grid_search --benchmark $benchmark --seed $seed 
    python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed &
    python -m src.run --optimizer carbs --benchmark $benchmark --seed $seed  --use-priors &
    # python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  &
    # python -m src.run --optimizer chinchilla_approach1 --benchmark $benchmark --seed $seed  --use-priors &
    # python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  &
    # python -m src.run --optimizer chinchilla_approach2 --benchmark $benchmark --seed $seed  --use-priors &
    # python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed &
    # python -m src.run --optimizer random_search --benchmark $benchmark --seed $seed  --use-priors &
done
wait
echo "All runs completed"
