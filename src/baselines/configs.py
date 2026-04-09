from enum import StrEnum
from collections.abc import Callable
from typing import Any


from neps.optimizers.algorithms import bo_guided_scaling, random_search, grid_search
from neps.optimizers.optimizer import AskFunction
from neps.optimizers.algorithms import custom, CustomOptimizer

from src.baselines.chinchilla_guided_scaling import chinchilla_guided_scaling

class OptimizerEnum(StrEnum):
    BO_SPACE_EXPANSION = "bo_space_expansion"
    BO_LEFT_BUDGET = "bo_left_budget"
    CHINCHILLA_APPROACH1 = "chinchilla_approach1"
    CHINCHILLA_APPROACH2 = "chinchilla_approach2"
    CHINCHILLA_GEMSTONES = "chinchilla_gemstones"
    RANDOM_SEARCH = "random_search"
    CARBS = "carbs"
    GRID_SEARCH = "grid_search"

def get_optimizer_wrapper(pipeline_space, evaluate_pipeline, param_estimator, seen_datapoints_estimator, flops_estimator, aspect_ratio_fn, data_parameter_name, size_per_numerical_dimension=10):
    def get_optimizer(optimizer_name: OptimizerEnum, max_evaluation_flops, max_target_flops, seed) -> tuple[Callable[..., Any] | CustomOptimizer, dict[str, Any]]:
        kwargs = {}
        match optimizer_name:
            case OptimizerEnum.BO_SPACE_EXPANSION: 
                return lambda space: bo_guided_scaling(
                    space,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    max_evaluation_flops=max_evaluation_flops,
                    max_target_flops=max_target_flops,
                    device='cpu',
                    sampling_strategy="space_expansion",
                ), kwargs
        
            case OptimizerEnum.BO_LEFT_BUDGET: 
                return lambda space: bo_guided_scaling(
                    space,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    max_evaluation_flops=max_evaluation_flops,
                    max_target_flops=max_target_flops,
                    device='cpu',
                    sampling_strategy="left_budget_contraction",
                ), kwargs
        
        
            case OptimizerEnum.CHINCHILLA_APPROACH1:
                return chinchilla_guided_scaling(
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    strategy="approach1",
                ), kwargs
        
            case OptimizerEnum.CHINCHILLA_APPROACH2: 
                return chinchilla_guided_scaling(
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    strategy="approach2",
                ), kwargs
        
            case OptimizerEnum.CHINCHILLA_GEMSTONES: 
                return chinchilla_guided_scaling(
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    strategy="gemstones",
                ), kwargs
        
            case OptimizerEnum.RANDOM_SEARCH: 
                return lambda space: random_search(
                    space,
                    use_priors=True,
                    ignore_fidelity=True,
                    constraints_func= lambda conf: max_evaluation_flops - flops_estimator(**conf)
                ), kwargs
            case OptimizerEnum.CARBS: 
                from src.baselines.carbs_utils import get_carbs_optimizer, evaluate_pipeline_wrapper
                optimizer, carbs_opt = get_carbs_optimizer(pipeline_space, seed)
                return optimizer, kwargs | {"evaluate_pipeline": evaluate_pipeline_wrapper(carbs_opt, evaluate_pipeline)}
            case OptimizerEnum.GRID_SEARCH:
                return lambda space: grid_search(space, size_per_numerical_dimension=size_per_numerical_dimension), kwargs
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return get_optimizer