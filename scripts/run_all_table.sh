#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PREDS=(tabpfn ensemble_xgb ensemble_lightgbm ensemble_mix)

for pred in "${PREDS[@]}"; do
    echo "Running predictions with SCAN_PREDICTOR_TYPE=$pred"
    SCAN_PREDICTOR_TYPE="$pred" \
        python -m src.plotting.table
done
