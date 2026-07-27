#!/usr/bin/env bash
set -euo pipefail

IMAGE="scaling-studies:test"
RESULTS_DIR="/Users/nastaranalipour/Downloads/work/dlc2workfs2/alipourn-scan"
BENCHMARK="llm"

# Trained ensemble_xgb/ensemble_xgb surrogates get cached as .joblib files
# inside scan_benchmark's own package dir (see base_performance_benchmark.py's
# _prepare_surrogate_for_target). That path lives inside the container's
# filesystem, so without this mount every fresh container retrains from
# scratch. Sharing one host dir across containers lets them reuse each
# other's trained models instead of redoing the same training every run.
MODEL_CACHE_DIR="/Users/nastaranalipour/Documents/scaling_studies_algorithm/.docker_model_cache_llm"
MODEL_CACHE_CONTAINER_PATH="/usr/local/lib/python3.11/site-packages/scan_benchmark/llm/performance_surrogate/saved_models"
mkdir -p "$MODEL_CACHE_DIR"

SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(chinchilla_approach1 chinchilla_approach2 random_search carbs)
PREDS=(ensemble_xgb)

for optimizer in "${OPTIMIZERS[@]}"; do
    container_name="ensemble_xgb-${BENCHMARK}-${optimizer}"

    docker run -d \
        --name "$container_name" \
        -v "${RESULTS_DIR}:${RESULTS_DIR}" \
        -v "${MODEL_CACHE_DIR}:${MODEL_CACHE_CONTAINER_PATH}" \
        "$IMAGE" \
        bash -c "
SEEDS=(${SEEDS[*]})
PREDS=(${PREDS[*]})
for pred in \"\${PREDS[@]}\"; do
    for seed in \"\${SEEDS[@]}\"; do
        echo \"=== optimizer=${optimizer} pred=\$pred seed=\$seed ===\"
        python -m src.run --optimizer ${optimizer} --benchmark ${BENCHMARK} --seed \"\$seed\" --pred \"\$pred\"
    done
done
"

    echo "started ${container_name}"
done

echo
echo "follow all logs with:  for c in ${OPTIMIZERS[*]/#/ensemble_xgb-}; do docker logs -f \$c & done"
echo "check status with:     docker ps --filter name=ensemble_xgb-"
echo "stop all with:         docker stop ${OPTIMIZERS[*]/#/ensemble_xgb-}"
