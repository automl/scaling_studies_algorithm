# Scaling Analysis

Scaling-law analysis for hyperparameter/compute-allocation search: this repo brings the existing
compute-allocation baselines together under one benchmark harness and budget accounting so they can be
compared systematically. Baselines covered: **Chinchilla approach 1**, **Chinchilla approach 2**,
**CARBS**, and **random search**.

## Install

```bash
conda create -n scaling python=3.11
conda activate scaling

# local packages (surrogate benchmarks, neps fork, carbs)
pip install ./subs ./dep/carbs
cd $PASS_TO_SCAN_BENCH_SUITE_HERE && python -m build --wheel && pip install dist/scan_benchmark-0.1.0-py3-none-any.whl
```

(See [Dockerfile](Dockerfile) for a containerized, CPU-only version of the same install.)

## Workflow

1. **Data acquisition** — run an optimizer against a benchmark surrogate, logging every
   trial (config, loss, FLOPs) to a `neps` run directory:

   ```bash
   python -m src.run --benchmark llm --optimizer chinchilla_approach1 --pred tabpfn --seed 40
   ```

   `--optimizer` selects the baseline (`chinchilla_approach1`, `chinchilla_approach2`, `random_search`,
   `carbs`, ...), `--benchmark` selects the surrogate task, `--pred` selects the surrogate's internal
   performance predictor (e.g. `tabpfn`, `ensemble_xgb`). Omit `--optimizer` to run the default baseline
   set back to back.

2. **Metrics-over-acquisition plots** — visualize how loss/regret evolves as each optimizer spends its
   FLOPs budget:

   ```bash
   python -m src.plotting.standalone_plots
   ```

   (`scripts/run_all_standalone.sh` sweeps this over all surrogate predictor types.)

3. **Scaling-law prediction plots and tables** — fit scaling laws on the acquired data and plot/tabulate
   how well each baseline extrapolates to unseen compute budgets:

   ```bash
   python -m src.plotting.prediction_plots
   python -m src.plotting.table
   ```

   (`scripts/run_all_predictions.sh` and `scripts/run_all_table.sh` sweep these over all surrogate
   predictor types.)

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
`.npz`-backed cache. On a cache hit the fit is loaded instead of
recomputed; on a miss it's fit once and saved.

**[src/plotting/pred_cache_gen.py](src/plotting/pred_cache_gen.py)** is the script that *warms* this
cache: for a given `--optimizer`/`--benchmark`/`--seed`/`--pred` combination it walks the run directory and
fits (and caches) the HP/loss/scale-param predictors at every intermediate trial count, so that later
plotting/table scripts (`prediction_plots.py`, `table.py`) just read the cache instead of refitting.

Run it directly:

```bash
python -m src.plotting.pred_cache_gen --optimizer chinchilla_approach1 --benchmark openclip --seed 40 --pred tabpfn
```

or fan it out over the full grid of optimizers × seeds × predictor types with the Slurm array script
**[scripts/gen_pred_cache_array.sh](scripts/gen_pred_cache_array.sh)**, which maps each
`SLURM_ARRAY_TASK_ID` to one `(optimizer, seed, pred)` combination and calls `pred_cache_gen` for it:

```bash
sbatch scripts/gen_pred_cache_array.sh
```

## Reproducibility

**Seeds.** Headline plots/tables aggregate over **11 seeds (40–50)**, the set defined in
[`src/plotting/consts.py`](src/plotting/consts.py) (`all_seeds = range(40, 50)`) — this is the canonical
seed list. Aggregation is a real statistical step: `standalone_plots.py`,
`prediction_plots.py`, and `table.py` all compute per-seed arrays including over
derived quantities like optimizer rank and Pareto-front hypervolume, not just raw loss.


**Predictor-type ablation.** `--pred {tabpfn, ensemble_xgb, ensemble_lightgbm, ensemble_mix}` swaps the
*surrogate benchmark's* internal performance-prediction model (the thing that stands in for actually
training a model at a given config). This is a robustness axis: rerunning the full
optimizer comparison under all four surrogates (`run_all_predictions.sh`, `run_all_standalone.sh`,
`run_all_table.sh`) checks that baseline rankings aren't an artifact of one surrogate's biases. Preempting the question of "are these
conclusions surrogate-specific?

**Compute budget.** Each benchmark's total optimizer-visible FLOPs budget (`max_evaluation_flops`) and
target range (`min_flops`/`max_flops`) are fixed per benchmark in
[`src/benchmarks/configs.py`](src/benchmarks/configs.py).
