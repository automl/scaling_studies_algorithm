import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
from pathlib import Path

from neps.state.neps_state import TrialRepo
from neps.utils.files import deserialize, serialize

from scan_benchmark.base_performance_benchmark import PerformancePredictorType
from src.baselines.configs import OptimizerEnum
from src.benchmarks.configs import get_benchmark, get_results_dir, get_target, set_predictor_type, BenchmarkEnum


def evaluate_target(root_dir: str, query, target: str) -> int:
    """Predict ``target`` for every evaluated config of a finished study.

    The surrogate of a study is bound to a single target, so `report.yaml` only
    holds predictions for that one. Here we re-query the stored configs with a
    benchmark bound to another (downstream) target and record the result under
    ``extra.predictions.<target>``, leaving everything else in the report intact.
    """
    configs_dir = Path(root_dir) / "configs"
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"No configs directory at {configs_dir}")

    config_dirs = sorted(configs_dir.glob("config_*"), key=lambda p: int(p.name.split("_")[-1]))
    n_updated = 0
    for config_dir in config_dirs:
        config_path = config_dir / "config.yaml"
        report_path = config_dir / "report.yaml"
        if not (config_path.exists() and report_path.exists()):
            print(f"{config_dir.name}: no report yet, skipping")
            continue

        config = deserialize(config_path)
        info_dict = query(**config)
        predictions = info_dict["predictions"]

        report = deserialize(report_path)
        extra = report.get("extra") or {}
        extra_predictions = extra.get("predictions") or {}
        extra_predictions[target] = predictions
        extra["predictions"] = extra_predictions
        report["extra"] = extra
        serialize(report, report_path)
        n_updated += 1

        if predictions is None:
            print(f"{config_dir.name}: {target} = failed")
        else:
            print(f"{config_dir.name}: {target} = {predictions['mean']} +- {predictions['uncertainty']}")

    # The pickled trial cache shadows the reports we just rewrote, so rebuild it from disk.
    TrialRepo(configs_dir).latest(refresh_cache=True)
    return n_updated


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict a downstream target for a finished scaling study')
    parser.add_argument('--benchmark', type=str, required=True, choices=[bench.value for bench in BenchmarkEnum],
                        help='Name of the study/root_dir created with neps')

    parser.add_argument('--optimizer', type=str, required=True, choices=[opt.value for opt in OptimizerEnum],
                        help='Name of the optimizer the study was run with')

    parser.add_argument('--pred', type=str, default=PerformancePredictorType.TABPFN.value,
                        choices=[pred.value for pred in PerformancePredictorType],
                        help='Performance predictor used by the benchmark surrogate')

    parser.add_argument('--seed', type=int, default=40,
                        help='Seed the study was run with')

    parser.add_argument('--use-priors', action='store_true',
                        help='The study used priors for the sampling approach')

    parser.add_argument('--target', type=str, required=True,
                        help='Downstream target to predict, e.g. imagenet1k_acc1')

    args = parser.parse_args()

    predictor_type = PerformancePredictorType(args.pred)
    set_predictor_type(predictor_type)
    results_dir = get_results_dir(predictor_type)

    optimizer_name = OptimizerEnum(args.optimizer)
    target = get_target(args.benchmark, args.target)  # validates --target against the benchmark
    benchmark = get_benchmark(BenchmarkEnum(args.benchmark), lc_aware=False, predictor_type=predictor_type,
                              target=target)

    seed = args.seed
    root_dir = str(results_dir / args.benchmark / f"{optimizer_name + '_with_priors' if args.use_priors else optimizer_name}" / f"seed_{seed}")
    print(f"Evaluating {args.target} for {optimizer_name} with seed {seed} in {root_dir}...")
    try:
        n_updated = evaluate_target(root_dir, benchmark.query, args.target)
    except Exception as e:
        print(f"Error occurred while running downstream task evaluation for {optimizer_name} with seed {seed}: {e}")
        raise e
    print(f"Finished downstream task evaluation for {optimizer_name}: updated {n_updated} configs!")
