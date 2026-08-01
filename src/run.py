import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import numpy as np
import socket
import torch
import random
import neps

from scan_benchmark.base_performance_benchmark import PerformancePredictorType
from src.baselines.configs import OptimizerEnum, get_optimizer_wrapper
from src.benchmarks.configs import get_benchmark, get_results_dir, set_predictor_type, BenchmarkEnum

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run scaling study on the benchmark')
    parser.add_argument('--benchmark', type=str, required=True, choices=[bench.value for bench in BenchmarkEnum],
                        help='Name of the study/root_dir to be created with neps')
    
    parser.add_argument('--optimizer', type=str, required=True, choices=[opt.value for opt in OptimizerEnum],
                        help='Name of the optimizer to be used')
    
    parser.add_argument('--pred', type=str, default=PerformancePredictorType.TABPFN.value,
                        choices=[pred.value for pred in PerformancePredictorType],
                        help='Performance predictor used by the benchmark surrogate')

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for reproducability')
    
    parser.add_argument('--use-priors', action='store_true',
                        help='Use prior for the sampling approach')
    
    args = parser.parse_args()

    predictor_type = PerformancePredictorType(args.pred)
    set_predictor_type(predictor_type)
    results_dir = get_results_dir(predictor_type)

    optimizers = [
        # OptimizerEnum.BO_SPACE_EXPANSION,
        # OptimizerEnum.BO_LEFT_BUDGET,
        OptimizerEnum.CHINCHILLA_APPROACH1, 
        OptimizerEnum.CHINCHILLA_APPROACH2,
        # OptimizerEnum.CHINCHILLA_GEMSTONES, 
        OptimizerEnum.RANDOM_SEARCH,
        OptimizerEnum.CARBS,
        ] if args.optimizer == "" else [OptimizerEnum(args.optimizer)]

    for optimizer_name in optimizers:
        lc_aware = optimizer_name in [OptimizerEnum.CHINCHILLA_APPROACH1.value, OptimizerEnum.CHINCHILLA_GEMSTONES.value]
        benchmark = get_benchmark(BenchmarkEnum(args.benchmark), lc_aware=lc_aware, predictor_type=predictor_type)
        get_optimizer = get_optimizer_wrapper(
            pipeline_space=benchmark.space,
            evaluate_pipeline=benchmark.evaluate_pipeline,
            param_estimator=benchmark.params_estimator,
            seen_datapoints_estimator=benchmark.seen_datapoints_estimator,
            flops_estimator=benchmark.flops_estimator,
            aspect_ratio_fn=benchmark.aspect_ratio_fn,
            data_parameter_name=benchmark.data_parameter_name,
            benchmark_type=BenchmarkEnum(args.benchmark),
            # type: ignore
        )
        seed_range = [40] if args.seed == 0 else [args.seed]
        for seed in seed_range:
            print(f"Running {optimizer_name} with seed {seed}...")
            # set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            root_dir = str(results_dir / args.benchmark / f"{optimizer_name + '_with_priors' if args.use_priors else optimizer_name}" / f"seed_{seed}")
            optimizer, kwargs = get_optimizer(
                OptimizerEnum(optimizer_name), 
                root_dir=root_dir,
                max_evaluation_flops=benchmark.max_evaluation_flops, 
                max_flops=int(benchmark.max_flops),
                seed=seed,
                use_priors=args.use_priors,
            )
            if 'evaluate_pipeline' not in kwargs:
                kwargs['evaluate_pipeline'] = benchmark.evaluate_pipeline
            try:
                neps.run(
                    pipeline_space=benchmark.space,
                    root_directory=root_dir,
                    cost_to_spend=benchmark.max_evaluation_flops,
                    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
                    overwrite_root_directory=True,
                    optimizer=optimizer,
                    **kwargs,
                )
            except Exception as e:
                print(f"Error occurred while running {optimizer_name} with seed {seed}: {e}")
                raise e
        print(f"Finished {optimizer_name}!")
            
