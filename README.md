# Scaling Analysis in ScAn Framework

Code for the scaling-analysis baseline experiments of
[ScAn-Bench: Evaluating Scaling Analysis Methodology](https://openreview.net/forum?id=q6rnd1qRBN), accepted at
NeurIPS 2026. It runs the compute-allocation baselines (**Chinchilla approach 1 and 2**, **CARBS**, **random search**)
on the LLM and OpenCLIP surrogates of ScAn-Bench under one FLOPs budget accounting, and scores how well the scaling
laws fitted to their trials extrapolate to larger compute budgets.

## Install

Python 3.11, CPU only (the "training" is a surrogate lookup).

```bash
git clone --recurse-submodules https://github.com/automl/scaling_studies_algorithm.git
cd scaling_studies_algorithm

conda create -n scaling python=3.11 -y && conda activate scaling
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision
pip install ./subs "carbs @ git+https://github.com/imbue-ai/carbs@8f7fee08658ad567d9dd6b6233247bc37d9962ea"
pip install scan-bench==0.4.0
```

`subs` is the NePS submodule (`scaling-neps` branch) that provides the run directories and scaling-law
optimizers. The last step lifts networkx past NePS' `<3` pin, as autogluon requires; the experiments run fine with
networkx 3.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SCAN_RESULT_DIR` | cluster path in [src/benchmarks/configs.py](src/benchmarks/configs.py) | Results root: runs in `<pred>/<benchmark>/<optimizer>/seed_<n>/`, caches, plots. |
| `SCAN_PREDICTOR_TYPE` | `tabpfn` | Surrogate's performance predictor; `--pred` on the entrypoints. |
| `SCAN_TARGET` | unset | Downstream target instead of the objective, e.g. `imagenet1k_acc1` (needs step 3). |
| `CONDA_ENV` | `scaling` | Conda env activated by `scripts/*.sh`. |

## Reproducing the experiments

The steps run in order. Steps 1–3 are Slurm arrays over the full grid (2 benchmarks × 4 optimizers × 11 seeds),
submitted with `sbatch --partition=<your_partition> --export=ALL,SCAN_RESULT_DIR=/my/scratch/scan scripts/<script>.sh`;
the step 4 scripts loop over the surrogate predictors in their `PREDS` list.

| Step | Single run | Full grid (`scripts/`) |
|---|---|---|
| 1. Acquire trials | `python -m src.run --benchmark llm --optimizer random_search --seed 40` | `eval_with_seeds_array.sh` |
| 2. Fit and cache the predictors | `python -m src.plotting.pred_cache_gen --benchmark llm --optimizer random_search --seed 40` | `gen_pred_cache_array.sh` |
| 3. Downstream targets (optional, OpenCLIP) | | `down_eval_with_seeds_array.sh` |
| 4. Figures and tables | `python -m src.plotting.standalone_plots`, `.prediction_plots`, `.table` | `run_all_{standalone,predictions,table}.sh` |

Step 4 aggregates every optimizer × seed listed in [src/plotting/consts.py](src/plotting/consts.py). Set
`OVERRIDE=1` to refit cached fits in step 2.

## Acquisition baselines

| Baseline | `--optimizer` | Source |
|---|---|---|
| Chinchilla approach 1 (Iso-Parameter trials) | `chinchilla_approach1` | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) |
| Chinchilla approach 2 (isoFLOP grid) | `chinchilla_approach2` | [src/baselines/chinchilla_guided_scaling.py](src/baselines/chinchilla_guided_scaling.py) |
| CARBS (cost-aware BO) | `carbs` | [src/baselines/carbs_utils.py](src/baselines/carbs_utils.py) |
| Random search under the FLOPs constraint | `random_search` | [src/baselines/configs.py](src/baselines/configs.py) |

## Scaling-law predictors

Fitted on every trial prefix and cached by [src/predictor_cache.py](src/predictor_cache.py):

- [src/loss_predictor.py](src/loss_predictor.py): achievable loss (`KAPLAN` power law, `CHINCHILLA3` joint fit).
- [src/scale_param_predictor.py](src/scale_param_predictor.py): compute-optimal model size (`POWER_LAW_FIT`, `CHINCHILLA3`).
- [src/hp_predictor.py](src/hp_predictor.py): full HP configuration (per-HP power laws on the Pareto front, or a GP posterior).

## Reproducibility

| Dependency | Version |
|---|---|
| NePS | `subs` submodule, commit `244aa10` |
| CARBS | commit `8f7fee0` |
| scan-bench | `0.4.0` |
| torch | `2.9.1` |

Seeds 40–49 are aggregated ([src/plotting/consts.py](src/plotting/consts.py)); the Slurm arrays also acquire seed 50.
Budgets and target ranges are fixed per benchmark in [src/benchmarks/configs.py](src/benchmarks/configs.py). To check
that rankings are not surrogate-specific, rerun with another `--pred` (`ensemble_xgb`, `ensemble_lightgbm`,
`ensemble_mix`, `autogluon`).

## Citation

```bibtex
@inproceedings{sermaxhaj2026scanbench,
  title={ScAn-Bench: Evaluating Scaling Analysis Methodology},
  author={A. Sermaxhaj and N. Alipour and D. Sinani and J. Hog and N. Mallik and S. Adriaensen and J. Jitsev and D. Stoll},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Evaluations and Datasets Track},
  url={https://arxiv.org/abs/2609.35707}
}
```

## Contributing

Issues and pull requests are welcome. License: [LICENSE](LICENSE).
