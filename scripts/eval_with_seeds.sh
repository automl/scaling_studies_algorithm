#!/bin/bash

#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --array=0-43%16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --job-name=openclip_scan
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%A_%a.log

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

python -m src.run --optimizer grid_search --benchmark openclip --seed 40
