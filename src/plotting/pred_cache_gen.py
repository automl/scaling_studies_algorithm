"""Pre-generate the prediction fit cache for all optimizers/seeds.

Run once per optimizer/benchmark/seed job to avoid recomputing expensive
scaling-law/GP fits every time plots or tables are generated.

Fitting is target_flop-independent (see get_loss_fit_trajectory /
get_scale_param_fit_trajectory / get_hp_fit_trajectory in loss_predictor.py,
scale_param_predictor.py, hp_predictor.py): each (predictor_type, n_trials)
is fit once and cached, then reused to extrapolate to any target_flop cheaply
at plot/table time. Warming therefore doesn't sweep a target_flop/ratio axis.

Usage examples
--------------
# All optimizers, all benchmarks, all seeds (sequential — slow)
python -m src.plotting.pred_cache_gen

# One optimizer + benchmark, all seeds
python -m src.plotting.pred_cache_gen --optimizer bo_left_budget_modified_jes --benchmark llm

# One job per seed (for SLURM array parallelism)
python -m src.plotting.pred_cache_gen --optimizer bo_left_budget_modified_jes --benchmark llm --seed 42

# Several explicit seeds in one process -- loads/trains the benchmark
# surrogate once and reuses it across all of them, instead of once per seed
python -m src.plotting.pred_cache_gen --optimizer bo_left_budget_modified_jes --benchmark llm --seed 40 41 42 43

# Use scripts/gen_pred_cache_array.sh to launch one SLURM job per seed automatically.
"""

if __name__ == "__main__":
    import argparse

    from scan_bench import PerformancePredictorType
    from src.benchmarks.configs import BenchmarkEnum, set_predictor_type

    # src.plotting.consts resolves the results dir and benchmark configs at import
    # time, so the predictor has to be exported before it is imported.
    pred_parser = argparse.ArgumentParser(add_help=False)
    pred_parser.add_argument(
        "--pred", "-p",
        choices=[pred.value for pred in PerformancePredictorType],
        default=PerformancePredictorType.TABPFN.value,
        help="Performance predictor used by the benchmark surrogate.",
    )
    predictor_type = PerformancePredictorType(pred_parser.parse_known_args()[0].pred)
    set_predictor_type(predictor_type)

    from src.loss_predictor import LossPredictorTypes
    from src.hp_predictor import HPPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    from src.plotting.prediction_plots import warm_prediction_cache
    from src.benchmarks.configs import get_results_dir, BenchmarkEnum

    from src.plotting.consts import loss_key  # honours SCAN_TARGET, like the plots do

    results_dir = get_results_dir(predictor_type)
    all_seeds = list(set(range(40, 48)))
    optimizers_dict = {
        "random_search": all_seeds,
        "chinchilla_approach1": all_seeds,
        "chinchilla_approach2": all_seeds,
        "carbs": all_seeds,
    }

    all_benchmarks = [BenchmarkEnum.OPENCLIP.value, BenchmarkEnum.LLM.value, BenchmarkEnum.TABPFN.value]
    all_optimizer_names = list(optimizers_dict.keys())

    parser = argparse.ArgumentParser(description="Pre-warm the prediction fit cache.", parents=[pred_parser])
    parser.add_argument(
        "--optimizer", "-o",
        choices=all_optimizer_names,
        default=None,
        help="Single optimizer to cache (default: all).",
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=all_benchmarks,
        default=None,
        help="Single benchmark to cache (default: all).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more seeds to cache (default: all seeds for the optimizer). "
            "Pass multiple to process them in one process -- the benchmark "
            "surrogate (e.g. a trained autogluon model) is loaded once at "
            "import time and reused across every seed, instead of reloading "
            "it once per --seed invocation."
        ),
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Ignore any cached fits and recompute + overwrite them instead of skipping.",
    )
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else all_benchmarks

    if args.optimizer:
        base_seeds: list[int | str] = list(args.seed) if args.seed is not None else list(optimizers_dict[args.optimizer])
        optimizers: dict[str, list[int | str]] = {args.optimizer: base_seeds}
    else:
        if args.seed is not None:
            optimizers = {opt: list(args.seed) for opt in all_optimizer_names}
        else:
            optimizers = {opt: list(seeds) for opt, seeds in optimizers_dict.items()}

    warm_prediction_cache(
        base_path=results_dir,
        benchmarks=benchmarks,
        optimizers=optimizers,
        loss_predictor_types=[
            LossPredictorTypes.KAPLAN,
            LossPredictorTypes.CHINCHILLA3,
            # LossPredictorTypes.GP_POSTERIOR,
        ],
        nparam_predictor_types=[
            ScaleParamPredictorTypes.POWER_LAW_FIT,
            ScaleParamPredictorTypes.CHINCHILLA3,
            # ScaleParamPredictorTypes.GP_POSTERIOR,
        ],
        hp_predictor_types=[
            HPPredictorTypes.LINEAR_REGRESSION,
            # HPPredictorTypes.GP_POSTERIOR,
        ],
        key_to_extract="cost",
        loss_key=loss_key,
        override=args.override,
    )
