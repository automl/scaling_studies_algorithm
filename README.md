# Scaling Analysis

Scaling-law analysis for hyperparameter/compute-allocation search: this repo brings the existing
compute-allocation baselines together under one benchmark harness and budget accounting so they can be
compared systematically. Baselines covered: **Chinchilla approach 1**, **Chinchilla approach 2**,
**CARBS**, and **random search**.

## Install

Requires Python 3.11 and git. The install is CPU-only — nothing in the pipeline needs a GPU, the
"training" is a surrogate lookup.

```bash
conda create -n scaling python=3.11 -y
conda activate scaling
pip install --upgrade pip build

git clone <SCAN_ANALYSIS_REPO_HERE>
cd scaling_studies_algorithm

git submodule update --init

mkdir -p dep
git clone https://github.com/imbue-ai/carbs dep/carbs

pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

pip install ./subs ./dep/carbs

(cd dep/scan_bench_suite && python -m build --wheel)
pip install scan-bench
```

(See [Dockerfile](Dockerfile) for a containerized version of the same install.)

## Configuration

All entrypoints are configured through environment variables — nothing needs editing in the source to
run this somewhere else:

| Variable | Default | Meaning |
|---|---|---|
| `SCAN_RESULT_DIR` | `/work/dlc2workfs2/alipourn-scan` (`RESULTS_BASE_DIR` in [src/benchmarks/configs.py](src/benchmarks/configs.py)) | Root for run directories, predictor caches and plots. Runs are stored under `$SCAN_RESULT_DIR/<pred>/<benchmark>/<optimizer>/seed_<n>/`. |
| `SCAN_PREDICTOR_TYPE` | `tabpfn` | Surrogate's internal performance predictor. Set by `--pred` on the entrypoints; only needed directly for the plotting scripts, which take no `--pred` flag. |
| `SCAN_TARGET` | unset (each study's own objective) | Plot/cache a post-hoc downstream target instead, e.g. `imagenet1k_acc1`. Requires step 3 below to have run. |
| `CONDA_ENV` | `scaling` | Which conda env `scripts/*.sh` activate (see [scripts/_common.sh](scripts/_common.sh)). |

To set `SCAN_RESULT_DIR` explicitly:

```bash
export SCAN_RESULT_DIR=/my/scratch/scan
```

The examples below spell it out inline instead, to make the dependency obvious.

## Workflow

The four steps are strictly ordered: plots and tables read the predictor cache, the cache reads the run
directories, and the run directories are written by acquisition. Each step is shown twice — a single
combination you can run locally to check your install, and the Slurm array that fans it out over the
full grid.

### 1. Data acquisition

Run an optimizer against a benchmark surrogate, logging every trial (config, loss, FLOPs) to a `neps`
run directory:

```bash
SCAN_RESULT_DIR=/my/scratch/scan \
    python -m src.run --benchmark llm --optimizer random_search --pred tabpfn --seed 40
```

`--benchmark` selects the surrogate task (`llm`, `openclip`, `tabpfn`), `--optimizer` the baseline
(`chinchilla_approach1`, `chinchilla_approach2`, `random_search`, `carbs`, ...), `--pred` the
surrogate's internal performance predictor (`tabpfn`, `ensemble_xgb`, `ensemble_lightgbm`,
`ensemble_mix`). One combination takes a few minutes on CPU.

Full grid — **both** benchmarks × N optimizers × M seeds × K predictor = 2 * M * N * K array tasks, one per
`SLURM_ARRAY_TASK_ID`:

```bash
mkdir -p logs                     # sbatch --output writes here
sbatch --partition=<your_cpu_partition> \
       --export=ALL,SCAN_RESULT_DIR=/my/scratch/scan \
       scripts/eval_with_seeds_array.sh
```

### 2. Warm the predictor cache

The three scaling-law predictors are refit at every intermediate trial count, which is far too slow to
do inline in the plotting scripts. [src/plotting/pred_cache_gen.py](src/plotting/pred_cache_gen.py)
walks a run directory and fits (and caches) all of them up front:

```bash
SCAN_RESULT_DIR=/my/scratch/scan \
    python -m src.plotting.pred_cache_gen --benchmark llm --optimizer random_search --seed 40 --pred tabpfn
```

Same task grid as step 1:

```bash
sbatch --partition=<your_partition> \
       --export=ALL,SCAN_RESULT_DIR=/my/scratch/scan \
       scripts/gen_pred_cache_array.sh
```

Add `OVERRIDE=1` to the `--export` list to refit and overwrite instead of reusing what is cached.

### 3. Post-hoc downstream evaluation (optional, openclip only)

Re-queries the configs already stored in step 1 with a surrogate bound to a downstream target
(e.g. ImageNet top-1 instead of the training objective) and records it under
`extra.predictions.<target>` in each `report.yaml`:

```bash
sbatch --partition=<your_partition> \
       --export=ALL,SCAN_RESULT_DIR=/my/scratch/scan \
       scripts/down_eval_with_seeds_array.sh
```

Only needed if you want to pass `SCAN_TARGET=imagenet1k_acc1` to the plotting steps below.

### 4. Plots and tables

Metrics over acquisition — how loss/regret evolves as each optimizer spends its FLOPs budget:

```bash
SCAN_RESULT_DIR=/my/scratch/scan python -m src.plotting.standalone_plots
```

Scaling-law prediction plots and tables — how well each baseline extrapolates to unseen target compute
budgets:

```bash
SCAN_RESULT_DIR=/my/scratch/scan python -m src.plotting.prediction_plots
SCAN_RESULT_DIR=/my/scratch/scan python -m src.plotting.table
```

These aggregate over the whole grid rather than one run, so they take no `--benchmark`/`--seed` flags —
they read every optimizer × seed listed in [src/plotting/consts.py](src/plotting/consts.py).

`scripts/run_all_standalone.sh`, `scripts/run_all_predictions.sh` and `scripts/run_all_table.sh` sweep
these over all four surrogate predictor types:

```bash
SCAN_RESULT_DIR=/my/scratch/scan CONDA_ENV=scaling scripts/run_all_predictions.sh
```

## Acquisition baselines

| Baseline | Source | Description |
|---|---|---|
| Chinchilla approach 1 | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) | Fits a scaling law using Iso-Parameter trials. |
| Chinchilla approach 2 | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) | Same fitting procedure as approach 1 but samples configs on a fixed isoFLOP grid rather than fx model paramaters list. |
| CARBS | [src/baselines/carbs_utils.py](src/baselines/carbs_utils.py) | Wraps the CARBS cost-aware Bayesian-optimization library to jointly search hyperparameters and compute allocation. |
| Random search | [src/baselines/configs.py](src/baselines/configs.py) (`OptimizerEnum.RANDOM_SEARCH`, built on `neps.optimizers.algorithms.random_search`) | Samples configs uniformly at random subject to the FLOPs budget constraint; the naive baseline. |

`src/baselines/configs.py` is the dispatcher (`get_optimizer_wrapper`) that maps an `OptimizerEnum` value
to its concrete optimizer instance for `src/run.py`.

## Scaling-law predictors

Once trials are acquired, three predictors extrapolate different quantities to unseen target FLOPs:

- **[src/hp_predictor.py](src/hp_predictor.py)** — predicts the *hyperparameter configuration* (width,
  depth, batch size, etc.) that a baseline would choose at a target compute budget, via per-parameter
  log-log power-law fits on the Pareto front or a GP-posterior draw.
- **[src/loss_predictor.py](src/loss_predictor.py)** — predicts the *achievable loss* at a target compute
  budget (`KAPLAN` independent power-law fit or `CHINCHILLA3` joint parametric fit).
- **[src/scale_param_predictor.py](src/scale_param_predictor.py)** — predicts the optimal *model size
  (parameter count)* at a target compute budget, either fit directly (`POWER_LAW_FIT`) or derived from the
  Chinchilla joint fit `CHINCHILLA3`

### Caching fits

All three predictors are expensive to fit (they refit on every trial count seen during a run), so each one
routes its fit results through **[src/predictor_cache.py](src/predictor_cache.py)**, a small
`.npz`-backed cache. On a cache hit the fit is loaded instead of recomputed; on a miss it's fit once and
saved. Step 2 of the workflow above is what populates it.

## Reproducibility

**Seeds.** The canonical seed list is defined in [`src/plotting/consts.py`](src/plotting/consts.py)
(`all_seeds = range(40, 50)`, i.e. **10 seeds, 40–49**). Note that the acquisition and cache scripts run
seeds 40–50 (11 seeds), so seed 50 is acquired but not aggregated — widen `all_seeds` to `range(40, 51)`
if you want it included. Aggregation is a real statistical step: `standalone_plots.py`,
`prediction_plots.py`, and `table.py` all compute per-seed arrays including over derived quantities like
optimizer rank and Pareto-front hypervolume, not just raw loss.

**Predictor-type ablation.** `--pred {tabpfn, ensemble_xgb, ensemble_lightgbm, ensemble_mix}` swaps the
*surrogate benchmark's* internal performance-prediction model (the thing that stands in for actually
training a model at a given config). This is a robustness axis: rerunning the full
optimizer comparison under all four surrogates (`run_all_predictions.sh`, `run_all_standalone.sh`,
`run_all_table.sh`) checks that baseline rankings aren't an artifact of one surrogate's biases,
preempting the question of "are these conclusions surrogate-specific?".

**Compute budget.** Each benchmark's total optimizer-visible FLOPs budget (`max_evaluation_flops`) and
target range (`min_flops`/`max_flops`) are fixed per benchmark in
[`src/benchmarks/configs.py`](src/benchmarks/configs.py).

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

For usage and licensing terms, see the LICENSE file.

