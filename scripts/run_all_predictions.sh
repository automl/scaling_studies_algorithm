#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PREDS=(tabpfn)

# Optional: plot a post-hoc downstream target instead of each study's own objective,
# e.g. `SCAN_TARGET=imagenet1k_acc1 scripts/run_all_predictions.sh`. The target must
# have been written into the reports first by scripts/down_eval_with_seeds_array.sh.
SCAN_TARGET="${SCAN_TARGET:-}"

for pred in "${PREDS[@]}"; do
    echo "Running predictions with SCAN_PREDICTOR_TYPE=$pred SCAN_TARGET=${SCAN_TARGET:-<objective>}"
    SCAN_PREDICTOR_TYPE="$pred" \
    SCAN_TARGET="$SCAN_TARGET" \
        python -m src.plotting.prediction_plots
done
