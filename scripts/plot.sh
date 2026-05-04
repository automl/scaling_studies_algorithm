#!/bin/bash

#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=plot_n
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%j.log
#SBATCH --time=06:00:00

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

python -m src.plotting.plot
