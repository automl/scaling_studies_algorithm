from enum import StrEnum
from collections.abc import Callable
from typing import Any


from neps.optimizers.algorithms import bo_guided_scaling, random_search
from neps.optimizers.optimizer import AskFunction
from neps.optimizers.algorithms import custom, CustomOptimizer

from src.baselines.chinchilla_guided_scaling import chinchilla_guided_scaling

class OptimizerEnum(StrEnum):
    BO_SPACE_EXPANSION_MES = "bo_space_expansion_mes"
    BO_SPACE_EXPANSION_EI = "bo_space_expansion_ei"
    BO_SPACE_EXPANSION_EI_COST_AWARE = "bo_space_expansion_ei_cost_aware"
    BO_SPACE_EXPANSION_JES = "bo_space_expansion_jes"
    BO_LEFT_BUDGET_ZERO_MEAN_EI = "bo_left_budget_zero_mean_ei"
    BO_LEFT_BUDGET_MES = "bo_left_budget_mes"
    BO_LEFT_BUDGET_EI = "bo_left_budget_ei"
    BO_LEFT_BUDGET_JES = "bo_left_budget_jes"
    BO_SPACE_EXPANSION_MODIFIED_JES = "bo_space_expansion_modified_jes"
    BO_LEFT_BUDGET_MODIFIED_JES = "bo_left_budget_modified_jes"
    CHINCHILLA_APPROACH1 = "chinchilla_approach1"
    CHINCHILLA_APPROACH2 = "chinchilla_approach2"
    CHINCHILLA_GEMSTONES = "chinchilla_gemstones"
    RANDOM_SEARCH = "random_search"
    CARBS = "carbs"

def get_optimizer_wrapper(pipeline_space, evaluate_pipeline, param_estimator, seen_datapoints_estimator, flops_estimator, aspect_ratio_fn, data_parameter_name, benchmark_type, size_per_numerical_dimension=10):
    def get_optimizer(optimizer_name: OptimizerEnum, root_dir, max_evaluation_flops, max_flops, seed = 1, use_priors = False) -> tuple[Callable[..., Any] | CustomOptimizer, dict[str, Any]]:
        kwargs = {}
        match optimizer_name:
            case (OptimizerEnum.BO_SPACE_EXPANSION_MES | OptimizerEnum.BO_SPACE_EXPANSION_EI
                  | OptimizerEnum.BO_SPACE_EXPANSION_JES | OptimizerEnum.BO_SPACE_EXPANSION_MODIFIED_JES):
                _acq_se = {
                    OptimizerEnum.BO_SPACE_EXPANSION_MES: "IT-MES",
                    OptimizerEnum.BO_SPACE_EXPANSION_EI: "EI",
                    OptimizerEnum.BO_SPACE_EXPANSION_JES: "IT-JES",
                    OptimizerEnum.BO_SPACE_EXPANSION_MODIFIED_JES: "IT-modified-JES",
                }
                return lambda space: bo_guided_scaling(
                    space,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    max_evaluation_flops=max_evaluation_flops,
                    max_target_flops=max_flops,
                    device='cpu',
                    use_priors=use_priors,
                    sampling_strategy="space_expansion",
                    acquisition_func_type=_acq_se[optimizer_name],
                    cost_aware="log" if optimizer_name == OptimizerEnum.BO_SPACE_EXPANSION_EI_COST_AWARE else False,
                ), kwargs

            case (OptimizerEnum.BO_LEFT_BUDGET_MES | OptimizerEnum.BO_LEFT_BUDGET_EI
                  | OptimizerEnum.BO_LEFT_BUDGET_JES | OptimizerEnum.BO_LEFT_BUDGET_ZERO_MEAN_EI
                  | OptimizerEnum.BO_LEFT_BUDGET_MODIFIED_JES):
                _acq_lb = {
                    OptimizerEnum.BO_LEFT_BUDGET_MES: "IT-MES",
                    OptimizerEnum.BO_LEFT_BUDGET_EI: "EI",
                    OptimizerEnum.BO_LEFT_BUDGET_ZERO_MEAN_EI: "EI",
                    OptimizerEnum.BO_LEFT_BUDGET_JES: "IT-JES",
                    OptimizerEnum.BO_LEFT_BUDGET_MODIFIED_JES: "IT-modified-JES",
                }
                return lambda space: bo_guided_scaling(
                    space,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    max_evaluation_flops=max_evaluation_flops,
                    use_priors=use_priors,
                    max_target_flops=max_flops,
                    use_gp_scaling_mean=optimizer_name != OptimizerEnum.BO_LEFT_BUDGET_ZERO_MEAN_EI,
                    device='cpu',
                    sampling_strategy="left_budget_contraction",
                    acquisition_func_type=_acq_lb[optimizer_name],
                ), kwargs
        
        
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
        
            case OptimizerEnum.CHINCHILLA_GEMSTONES: 
                return chinchilla_guided_scaling(
                    evaluate_pipeline=evaluate_pipeline,
                    params_estimator=param_estimator,
                    seen_datapoints_estimator=seen_datapoints_estimator,
                    flops_estimator=flops_estimator,
                    aspect_ratio_fn=aspect_ratio_fn,
                    max_evaluation_flops=max_evaluation_flops,
                    data_parameter_name=data_parameter_name,
                    max_flops=max_flops,
                    strategy="gemstones",
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
                    benchmark_type=benchmark_type,
                    )
                return optimizer, kwargs | {"evaluate_pipeline": evaluate_pipeline_wrapper(carbs_opt, evaluate_pipeline, flops_estimator, max_flops)}
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return get_optimizer