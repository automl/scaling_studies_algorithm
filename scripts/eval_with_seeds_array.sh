#!/bin/bash

#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --job-name=scan_acquire
#SBATCH --output=logs/%x-%a-%j.log
#SBATCH --array=0-87

# Data acquisition sweep: benchmark x optimizer x seed x predictor, one array
# task per combination. Both surrogate benchmarks (llm, openclip) are covered
# here, so this single array replaces the old per-benchmark scripts.
#
#   sbatch --partition=<your_cpu_partition> scripts/eval_with_seeds_array.sh
#
# Results land under $SCAN_RESULT_DIR (see scripts/_common.sh); submit with
# `SCAN_RESULT_DIR=/my/path sbatch --export=ALL,SCAN_RESULT_DIR ...` to override.

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

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

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

echo "Task $SLURM_ARRAY_TASK_ID: benchmark=$benchmark optimizer=$optimizer seed=$seed pred=$pred"

python -m src.run --benchmark "$benchmark" --optimizer "$optimizer" --seed "$seed" --pred "$pred"
