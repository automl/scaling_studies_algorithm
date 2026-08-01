#!/bin/bash

#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --job-name=openclip_downstream
#SBATCH --output=logs/%x-%a-%j.log
#SBATCH --array=0-43

# Post-hoc downstream evaluation: for every finished study (optimizer x seed x
# predictor) this re-queries the stored configs with a surrogate bound to a
# downstream target and records it under `extra.predictions.<target>` in each
# report.yaml. Requires the corresponding runs of eval_with_seeds_array.sh to
# have finished -- it only reads the configs they wrote.
#
# The array size must be NUM_OPTIMIZERS * NUM_SEEDS * NUM_PREDS * NUM_TARGETS - 1.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEEDS=(40 41 42 43 44 45 46 47 48 49 50)
OPTIMIZERS=(chinchilla_approach1 chinchilla_approach2 random_search carbs)
PREDS=(tabpfn)
BENCHMARK=openclip
# Downstream targets, see AVAILABLE_TARGETS in src/benchmarks/openclip_space.py
# (e.g. cifar10_mean_per_class_recall, vtab_cifar100_mean_per_class_recall, mnist_acc1).
TARGETS=(imagenet1k_acc1)

NUM_OPTIMIZERS=${#OPTIMIZERS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_PREDS=${#PREDS[@]}
NUM_TARGETS=${#TARGETS[@]}

NUM_TASKS=$(( NUM_OPTIMIZERS * NUM_SEEDS * NUM_PREDS * NUM_TARGETS ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_TASKS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID is out of range (only $NUM_TASKS combinations), nothing to do."
    exit 0
fi

idx=$SLURM_ARRAY_TASK_ID

opt_idx=$(( idx % NUM_OPTIMIZERS ))
idx=$(( idx / NUM_OPTIMIZERS ))

seed_idx=$(( idx % NUM_SEEDS ))
idx=$(( idx / NUM_SEEDS ))

pred_idx=$(( idx % NUM_PREDS ))
idx=$(( idx / NUM_PREDS ))

target_idx=$idx

optimizer=${OPTIMIZERS[$opt_idx]}
seed=${SEEDS[$seed_idx]}
pred=${PREDS[$pred_idx]}
target=${TARGETS[$target_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: optimizer=$optimizer seed=$seed pred=$pred benchmark=$BENCHMARK target=$target"

python -m src.posthoc_downstream_eval --optimizer "$optimizer" --benchmark "$BENCHMARK" --seed "$seed" --pred "$pred" --target "$target"
