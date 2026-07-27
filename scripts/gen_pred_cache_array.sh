#!/bin/bash

#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --job-name=llm_pred
#SBATCH --output=/home/alipourn/scaling_studies_algorithm/logs/%x-%a-%j.log
#SBATCH --array=0-175

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

benchmark="openclip"
PREDS=(ensemble_xgb ensemble_lightgbm ensemble_mix tabpfn)
optimizers=(chinchilla_approach1 chinchilla_approach2 random_search carbs)

# Map SLURM_ARRAY_TASK_ID to optimizer x seed (40-50) x pred
n_optimizers=${#optimizers[@]}
n_seeds=11

idx=$SLURM_ARRAY_TASK_ID

optimizer_idx=$(( idx % n_optimizers ))
idx=$(( idx / n_optimizers ))

seed=$(( 40 + idx % n_seeds ))
idx=$(( idx / n_seeds ))

pred_idx=$idx

optimizer="${optimizers[$optimizer_idx]}"
pred="${PREDS[$pred_idx]}"

# Set OVERRIDE=1 (e.g. `sbatch --export=ALL,OVERRIDE=1 gen_pred_cache_array.sh`)
# to force every fit to be recomputed and the cache overwritten instead of
# reusing whatever is already cached.
# OVERRIDE_FLAG=()
# if [ "${OVERRIDE:-0}" = "1" ]; then
#     OVERRIDE_FLAG=(--override)
# fi

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer  benchmark=$benchmark  seed=$seed  pred=$pred  override=${OVERRIDE:-0}"

SCAN_BENCH_MODELS_DIR_LLM="/work/dlc2workfs2/alipourn-scan/llm_models" SCAN_BENCH_MODELS_DIR_VLM="/work/dlc2workfs2/alipourn-scan/vlm_models" python -m src.plotting.pred_cache_gen \
    --optimizer "$optimizer" \
    --benchmark "$benchmark" \
    --seed "$seed" \
    --pred "$pred" \
    --override