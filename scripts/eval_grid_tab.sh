#!/bin/bash

#SBATCH --partition=ml_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --job-name=grid3_tabpfn
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --error=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.err

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"


python -m src.run --optimizer grid_search --benchmark tabpfn --seed 40 &
wait
