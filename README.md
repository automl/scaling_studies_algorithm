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
| Chinchilla approach 1 | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) | Fits a scaling law on the current Pareto front after every trial and samples the next config along it (isoFLOP-style, loss-curve aware). |
| Chinchilla approach 2 | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) | Same fitting procedure as approach 1 but samples configs on a fixed isoFLOP grid rather than refitting after each trial. |
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
`.npz`-backed cache keyed by `(predictor slug, n_trials)`. On a cache hit the fit is loaded instead of
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
seed list. Aggregation is a real statistical step, not just overlaid runs: `standalone_plots.py`,
`prediction_plots.py`, and `table.py` all compute per-seed arrays and reduce them with
`np.nanmean`/`np.nanstd` (and stderr = std/√n for the shaded `fill_between` bands), including over
derived quantities like optimizer rank and Pareto-front hypervolume, not just raw loss.

> Two other seed counts appear elsewhere and should be reconciled to the 10-seed canonical set above:
> `src/plotting/pred_cache_gen.py`'s default (no `--seed` given) warms **8 seeds (40–47)**, and
> `scripts/gen_pred_cache_array.sh`'s Slurm array covers **11 seeds (40–50)**. Neither mismatch produces
> wrong numbers (a cache miss just triggers an on-the-fly refit, and an extra cached seed is simply
> unused), but they should be pointed at the same `all_seeds` source of truth before submission so the
> paper's seed count is unambiguous everywhere it's derived from code.

**Predictor-type ablation.** `--pred {tabpfn, ensemble_xgb, ensemble_lightgbm, ensemble_mix}` swaps the
*surrogate benchmark's* internal performance-prediction model (the thing that stands in for actually
training a model at a given config). This is a robustness axis, not a tuning knob: rerunning the full
optimizer comparison under all four surrogates (`run_all_predictions.sh`, `run_all_standalone.sh`,
`run_all_table.sh`) checks that baseline rankings aren't an artifact of one surrogate's biases. Preempting the question of "are these
conclusions surrogate-specific?

**Compute budget.** Each benchmark's total optimizer-visible FLOPs budget (`max_evaluation_flops`) and
target range (`min_flops`/`max_flops`) are fixed per benchmark in
[`src/benchmarks/configs.py`](src/benchmarks/configs.py); see also
[`scripts/flops_estimation.py`](scripts/flops_estimation.py) and
[`scripts/total_flops_calculator.py`](scripts/total_flops_calculator.py) for how those numbers are
derived. Reporting the exact budget, hardware, and wall-clock/GPU-hours per full sweep (10 seeds × 4
baselines × 4 predictor types × N benchmarks) in the paper satisfies the NeurIPS reproducibility
checklist's compute-cost item.

**Known gaps for an outside reproduction:**
- `dep/carbs` and `dep/scan_bench_suite` are gitignored (see `.gitignore`) and not vendored, submoduled,
  or published anywhere referenced in this repo — a fresh clone cannot install them. Either vendor/release
  them (license permitting) or document them as internal dependencies with a fallback reproduction path.
- The `subs` git submodule ([`.gitmodules`](.gitmodules)) uses an absolute host path
  (`/home/alipourn/scaling_studies_algorithm/neps`) instead of a relative one and points at
  `automl/neps`, while the code imports custom-looking symbols
  (`neps.optimizers.scaling_law_guided.ScalingLawGuidedOptimizer`) — confirm whether this needs a specific
  fork/branch/commit pinned rather than upstream `main`, and fix the submodule path so it isn't tied to
  one machine's directory layout.
- `RESULTS_BASE_DIR` in `src/benchmarks/configs.py`, and the `SCAN_BENCH_MODELS_DIR_LLM`/`_VLM` env vars
  used throughout `scripts/*.sh`, point at an internal cluster filesystem
  (`/work/dlc2workfs2/alipourn-scan/...`). The trained surrogate benchmark checkpoints that live there are
  never described as a releasable artifact — without them, no baseline can be run at all. This is worth
  resolving before submission, e.g. by releasing the checkpoints or a smaller synthetic stand-in
  (`src/benchmarks/synthetic_transformer_space.py`) that reviewers can run without cluster access.
- `environment.yaml` exists but is empty; the install steps above are reconstructed from
  [`Dockerfile`](Dockerfile), which is the only place dependency versions are actually pinned. Fill in
  `environment.yaml` (or a `requirements.txt`/lockfile) so there's one canonical, versioned install path.
- The `scripts/*.sh` files are Slurm jobs hardcoded to specific partitions and absolute home-dir paths —
  useful as a record of how the full sweep was launched, but not runnable as-is outside this cluster. The
  single-run `python -m src.run ...` command above is the actual portable reproduction path; the README
  should keep that distinction explicit rather than implying the sbatch scripts are how an outside reader
  reproduces results.
