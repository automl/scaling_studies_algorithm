from enum import StrEnum
from collections.abc import Callable
from typing import Any


from neps.optimizers.algorithms import random_search
from neps.optimizers.optimizer import AskFunction
from neps.optimizers.algorithms import custom, CustomOptimizer

from src.baselines.chinchilla_guided_scaling import chinchilla_guided_scaling

class OptimizerEnum(StrEnum):
    CHINCHILLA_APPROACH1 = "chinchilla_approach1"
    CHINCHILLA_APPROACH2 = "chinchilla_approach2"
    RANDOM_SEARCH = "random_search"
    CARBS = "carbs"

def get_optimizer_wrapper(pipeline_space, evaluate_pipeline, param_estimator, seen_datapoints_estimator, flops_estimator, aspect_ratio_fn, data_parameter_name, benchmark_type, size_per_numerical_dimension=10):
    def get_optimizer(optimizer_name: OptimizerEnum, root_dir, max_evaluation_flops, max_flops, seed = 1, use_priors = False) -> tuple[Callable[..., Any] | CustomOptimizer, dict[str, Any]]:
        kwargs = {}
        match optimizer_name:
            case OptimizerEnum.CHINCHILLA_APPROACH1:
                return chinchilla_guided_scaling(
                    evaluate_pipeline=evaluate_pipeline,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    max_flops=max_flops,
                    strategy="approach1",
                    use_priors=use_priors,
                    benchmark_type=benchmark_type,
                ), kwargs
        
            case OptimizerEnum.CHINCHILLA_APPROACH2: 
                return chinchilla_guided_scaling(
                    evaluate_pipeline=evaluate_pipeline,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    max_flops=max_flops,
                    strategy="approach2",
                    use_priors=use_priors,
                    benchmark_type=benchmark_type,
                ), kwargs
        
            case OptimizerEnum.RANDOM_SEARCH: 
                return lambda space: random_search(
                    space,
                    use_priors=use_priors,
                    ignore_fidelity=True,
                    constraints_func= lambda conf: max_flops - flops_estimator(**conf)
                ), kwargs
            case OptimizerEnum.CARBS: 
                from src.baselines.carbs_utils import get_carbs_optimizer, evaluate_pipeline_wrapper
                optimizer, carbs_opt = get_carbs_optimizer(
                    pipeline_space, 
                    seed, use_priors,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    max_evaluation_flops=max_evaluation_flops,
                    max_flops=max_flops,
                    data_parameter_name=data_parameter_name,
                    evaluate_pipeline=evaluate_pipeline,
                    root_dir=root_dir,
                    )
                return optimizer, kwargs | {"evaluate_pipeline": evaluate_pipeline_wrapper(carbs_opt, evaluate_pipeline, flops_estimator, max_flops)}
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return get_optimizer