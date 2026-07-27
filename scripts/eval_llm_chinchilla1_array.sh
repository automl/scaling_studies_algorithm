#!/bin/bash

#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --job-name=all-llm-gpu
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=0-1

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

SEEDS=(44 45)
OPTIMIZERS=(random_search)
PREDS=(ensemble_xgb)
# PREDS=(autogluon)
BENCHMARK=openclip

NUM_OPTIMIZERS=${#OPTIMIZERS[@]}
NUM_SEEDS=${#SEEDS[@]}

idx=$SLURM_ARRAY_TASK_ID

opt_idx=$(( idx % NUM_OPTIMIZERS ))
idx=$(( idx / NUM_OPTIMIZERS ))

seed_idx=$(( idx % NUM_SEEDS ))
idx=$(( idx / NUM_SEEDS ))

pred_idx=$idx

optimizer=${OPTIMIZERS[$opt_idx]}
seed=${SEEDS[$seed_idx]}
pred=${PREDS[$pred_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer seed=$seed pred=$pred benchmark=$BENCHMARK"

SCAN_BENCH_MODELS_DIR_LLM="/work/dlc2workfs2/alipourn-scan/llm_models" SCAN_BENCH_MODELS_DIR_VLM="/work/dlc2workfs2/alipourn-scan/vlm_models" python -m src.run --optimizer "$optimizer" --benchmark "$BENCHMARK" --seed "$seed" --pred "$pred"
