#!/bin/bash

#SBATCH --partition=ml_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --gpus=20
#SBATCH --mem=20G
#SBATCH --job-name=tabpfn
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=0-0

# NOTE: this array now covers OpenCLIP only (all optimizers). It is meant to
# be submitted with a dependency on eval_llm_chinchilla1_array.sh finishing
# first -- see scripts/submit_llm_then_openclip.sh.

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

SEEDS=(40)
OPTIMIZERS=(chinchilla_approach1)
PREDS=(autogluon)
BENCHMARK=tabpfn

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

SCAN_BENCH_MODELS_DIR_VLM="/work/dlc2workfs2/alipourn-scan/vlm_models" python -m src.run --optimizer "$optimizer" --benchmark "$BENCHMARK" --seed "$seed" --pred "$pred"

# python -m src.run --optimizer chinchilla_approach1 --benchmark openclip --seed 41 --pred autogluon


# for i in $(seq 40 50); do
#     SCAN_BENCH_MODELS_DIR_VLM="/work/dlc2workfs2/alipourn-scan/vlm_models" python -m src.run --optimizer chinchilla_approach1 --benchmark openclip --seed "$i" --pred autogluon
# done