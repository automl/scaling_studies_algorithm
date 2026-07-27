#!/usr/bin/env bash
set -euo pipefail

IMAGE="scaling-studies:test"
RESULTS_DIR="/Users/nastaranalipour/Downloads/work/dlc2workfs2/alipourn-scan"
BENCHMARK="openclip"

# Trained ensemble_xgb/autogluon surrogates get cached as .joblib files
# inside scan_benchmark's own package dir (see base_performance_benchmark.py's
# _prepare_surrogate). That path lives inside the container's filesystem, so
# without this mount every fresh container retrains from scratch. Sharing one
# host dir across containers lets them reuse each other's trained models
# instead of redoing the same training every run. All SEEDS for a given
# (optimizer, pred) are passed to a single `pred_cache_gen.py` invocation
# below (not one per seed), so the surrogate is loaded/trained once per
# container and reused across every seed, not reloaded per seed.
MODEL_CACHE_DIR="/Users/nastaranalipour/Documents/scaling_studies_algorithm/.docker_model_cache"
MODEL_CACHE_CONTAINER_PATH="/usr/local/lib/python3.11/site-packages/scan_benchmark/vlm/performance_surrogate/saved_models"
mkdir -p "$MODEL_CACHE_DIR"

SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(chinchilla_approach1 chinchilla_approach2 random_search carbs)
PREDS=(autogluon ensemble_mix ensemble_lightgbm ensemble_xgb tabpfn)

# One container per (optimizer, pred) pair -- 20 total -- each looping over
# all seeds sequentially. This host has 12 CPU cores and ~11.7GB allocated to
# Docker (check with `docker info`); several of the PREDS backends default to
# using every visible core internally (n_jobs=-1), and each container loads
# its own copy of the trained surrogate into memory. Launching all 20 at once
# would oversubscribe both cores and RAM -- likely slower (context-switch
# thrashing) or an OOM kill, not faster. Launch in capped-size batches
# instead, and cap each container's own CPU/RAM so a batch can't oversubscribe
# either. Tune MAX_PARALLEL/CPUS_PER_CONTAINER/MEM_PER_CONTAINER to what you
# actually observe with `docker stats` -- these are conservative starting
# points, not measured values.
MAX_PARALLEL=1
CPUS_PER_CONTAINER=12
MEM_PER_CONTAINER="2500m"

batch=()

start_container() {
    local optimizer="$1" pred="$2"
    local container_name="cache-${pred}-${optimizer}"

    docker run -d \
        --name "$container_name" \
        --cpus "$CPUS_PER_CONTAINER" \
        --memory "$MEM_PER_CONTAINER" \
        -v "${RESULTS_DIR}:${RESULTS_DIR}" \
        -v "${MODEL_CACHE_DIR}:${MODEL_CACHE_CONTAINER_PATH}" \
        "$IMAGE" \
        python -m src.plotting.pred_cache_gen \
            --optimizer "$optimizer" --benchmark "$BENCHMARK" --pred "$pred" \
            --seed "${SEEDS[@]}" \
        >/dev/null

    echo "started ${container_name}"
    batch+=("$container_name")
    all_containers+=("$container_name")
}

all_containers=()
for pred in "${PREDS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        start_container "$optimizer" "$pred"

        if (( ${#batch[@]} == MAX_PARALLEL )); then
            echo "  ...waiting for batch to finish: ${batch[*]}"
            docker wait "${batch[@]}" >/dev/null
            batch=()
        fi
    done
done

if (( ${#batch[@]} > 0 )); then
    echo "  ...waiting for final batch: ${batch[*]}"
    docker wait "${batch[@]}" >/dev/null
fi

echo
echo "all ${#all_containers[@]} containers finished"
echo "follow a log with:      docker logs -f <container_name>"
echo "check status with:      docker ps -a --filter name=cache-"
echo "remove finished ones:   docker rm ${all_containers[*]}"

# python -m src.plotting.pred_cache_gen --optimizer chinchilla_approach1 --benchmark openclip --seed 41 --pred autogluon
