"""Pre-generate the prediction trajectory cache for all optimizers/seeds.

Run once per optimizer/benchmark job to avoid recomputing expensive CHINCHILLA3
fits every time plots are generated.  Launch one job per optimizer to parallelise:

    # all optimizers, all benchmarks (sequential)
    python -m src.plotting.pred_cache_gen

    # one job per optimizer — run these in parallel (e.g. SLURM array)
    python -m src.plotting.pred_cache_gen --optimizer chinchilla_approach1 --benchmark openclip
    python -m src.plotting.pred_cache_gen --optimizer chinchilla_approach1 --benchmark llm
    python -m src.plotting.pred_cache_gen --optimizer carbs --benchmark openclip
    ...
"""

if __name__ == "__main__":
    import argparse

    from src.benchmarks.configs import BenchmarkEnum
    from src.loss_predictor import LossPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    from src.plotting.consts import (
        results_dir,
        optimizers_dict,
        target_flop_per_benchmark,
        loss_key,
    )
    from src.plotting.prediction_plots import warm_prediction_cache

    all_benchmarks = [BenchmarkEnum.OPENCLIP.value, BenchmarkEnum.LLM.value]
    all_optimizers = list(optimizers_dict.keys())

    parser = argparse.ArgumentParser(description="Pre-warm prediction trajectory cache.")
    parser.add_argument(
        "--optimizer", "-o",
        choices=all_optimizers,
        default=None,
        help="Single optimizer to cache (default: all).",
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=all_benchmarks,
        default=None,
        help="Single benchmark to cache (default: all).",
    )
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else all_benchmarks
    optimizers = (
        {args.optimizer: optimizers_dict[args.optimizer]}
        if args.optimizer
        else optimizers_dict
    )

    warm_prediction_cache(
        base_path=results_dir,
        benchmarks=benchmarks,
        optimizers=optimizers,
        target_flop_per_benchmark=target_flop_per_benchmark,
        loss_predictor_types=[
            LossPredictorTypes.GP_POSTERIOR,
        ],
        nparam_predictor_types=[
            ScaleParamPredictorTypes.GP_POSTERIOR,
        ],
        
        key_to_extract="cost",
        loss_key=loss_key,
    )
