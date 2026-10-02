#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PREDS=(tabpfn)

for pred in "${PREDS[@]}"; do
    echo "Running standalone with SCAN_PREDICTOR_TYPE=$pred"
    SCAN_PREDICTOR_TYPE="$pred" \
        python -m src.plotting.standalone_plots
done
