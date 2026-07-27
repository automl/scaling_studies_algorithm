#!/bin/bash

source /home/alipourn/miniconda3/etc/profile.d/conda.sh
conda activate scaling

export PYTHONPATH="/home/alipourn/scaling_studies_algorithm:$PYTHONPATH"

PREDS=(tabpfn ensemble_xgb ensemble_lightgbm ensemble_mix)

for pred in "${PREDS[@]}"; do
    echo "Running predictions with SCAN_PREDICTOR_TYPE=$pred"
    SCAN_BENCH_MODELS_DIR_LLM="/work/dlc2workfs2/alipourn-scan/llm_models" \
    SCAN_BENCH_MODELS_DIR_VLM="/work/dlc2workfs2/alipourn-scan/vlm_models" \
    SCAN_PREDICTOR_TYPE="$pred" \
        python -m src.plotting.table
done
