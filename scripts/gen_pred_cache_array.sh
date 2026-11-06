#!/bin/bash

#SBATCH --partition=mlstudentdlc2_gpu-l40s
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --job-name=scan_pred_cache
#SBATCH --output=logs/%x-%a-%j.log
#SBATCH --array=0-87

# Warms the predictor cache for every acquired study: one array task per
# (benchmark, optimizer, seed, pred) combination -- the same grid as
# scripts/eval_with_seeds_array.sh, which must have finished first.
#
#   sbatch --partition=<your_gpu_partition> scripts/gen_pred_cache_array.sh
#
# `scripts/gen_pred_cache_array.sh --count` prints the array size for the lists
# below.
#
# Optional environment:
#   OVERRIDE=1      refit everything and overwrite the cache instead of reusing it
#   SCAN_TARGET=... cache a downstream target instead of each study's own
#                   objective (openclip only, and only after
#                   scripts/down_eval_with_seeds_array.sh has written it)
#   SCAN_RESULT_DIR results root (see scripts/_common.sh)
# Pass them through with `sbatch --export=ALL,OVERRIDE=1 ...`.

set -euo pipefail

BENCHMARKS=(llm openclip)
SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(chinchilla_approach1 chinchilla_approach2 random_search carbs)
PREDS=(tabpfn)

NUM_BENCHMARKS=${#BENCHMARKS[@]}
NUM_OPTIMIZERS=${#OPTIMIZERS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_PREDS=${#PREDS[@]}
TOTAL=$(( NUM_BENCHMARKS * NUM_OPTIMIZERS * NUM_SEEDS * NUM_PREDS ))

if [[ "${1:-}" == "--count" ]]; then
    echo "$TOTAL combinations -> use #SBATCH --array=0-$(( TOTAL - 1 ))"
    exit 0
fi

# sbatch copies the batch script into Slurm's spool dir, so BASH_SOURCE no longer
# points at scripts/. Fall back to the directory the job was submitted from.
_common=""
for _dir in "$(dirname "${BASH_SOURCE[0]}")" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/scripts"; do
    if [[ -n "$_dir" && -f "$_dir/_common.sh" ]]; then _common="$_dir/_common.sh"; break; fi
done
if [[ -z "$_common" ]]; then
    echo "error: cannot locate scripts/_common.sh (submit from the repo root or scripts/)" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$_common"

idx=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset -- submit with sbatch}
if (( idx >= TOTAL )); then
    echo "Task $idx is outside the 0-$(( TOTAL - 1 )) grid; nothing to do."
    exit 0
fi

bench_idx=$(( idx % NUM_BENCHMARKS )); idx=$(( idx / NUM_BENCHMARKS ))
opt_idx=$((   idx % NUM_OPTIMIZERS )); idx=$(( idx / NUM_OPTIMIZERS ))
seed_idx=$((  idx % NUM_SEEDS ));      idx=$(( idx / NUM_SEEDS ))
pred_idx=$idx

benchmark=${BENCHMARKS[$bench_idx]}
optimizer=${OPTIMIZERS[$opt_idx]}
seed=${SEEDS[$seed_idx]}
pred=${PREDS[$pred_idx]}

OVERRIDE_FLAG=()
if [[ "${OVERRIDE:-0}" == "1" ]]; then
    OVERRIDE_FLAG=(--override)
fi

echo "Task $SLURM_ARRAY_TASK_ID: benchmark=$benchmark optimizer=$optimizer seed=$seed pred=$pred override=${OVERRIDE:-0}"

python -m src.plotting.pred_cache_gen \
    --optimizer "$optimizer" \
    --benchmark "$benchmark" \
    --seed "$seed" \
    --pred "$pred" \
    "${OVERRIDE_FLAG[@]}"
