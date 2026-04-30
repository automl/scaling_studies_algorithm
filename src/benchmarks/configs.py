from dataclasses import dataclass
import neps
from typing import Callable
from enum import Enum
import src.benchmarks.openclip_space as openclip
import src.benchmarks.llm_space as llm
import src.benchmarks.tabpfn_space as tabpfn

class BenchmarkEnum(str, Enum):
    OPENCLIP = "openclip"
    LLM = "llm"
    TABPFN = "tabpfn"

@dataclass
class BenchmarkConfig:
    space: neps.PipelineSpace
    params_estimator: Callable
    seen_datapoints_estimator: Callable
    flops_estimator: Callable
    evaluate_pipeline: Callable
    max_evaluation_flops: float
    aspect_ratio_fn: Callable
    data_parameter_name: str
    max_flops: float
    min_flops: float
    best_loss_guess: float

def get_benchmark(benchmark: BenchmarkEnum | str, lc_aware: bool) -> BenchmarkConfig:
    if isinstance(benchmark, str):
        benchmark = BenchmarkEnum(benchmark)

    match benchmark:
        case BenchmarkEnum.OPENCLIP: 
            bench = openclip.get_benchmark()
            min_flops = 3.600000e+04
            max_flops = 5.970000e+08
            grid_diagonal = 9 + 9 # vision_width and text_width have 9 possible values each
            total_eval_flops = max_flops * 6.52
            return BenchmarkConfig(
                space= openclip.get_pipeline_space(),
                params_estimator=openclip.parameters_estimator_wrapper(bench),
                seen_datapoints_estimator=openclip.seen_datapoints_estimator,
                flops_estimator=openclip.flops_estimator_wrapper(bench),
                evaluate_pipeline=openclip.evaluate_pipeline_wrapper(bench, lc_aware),
                max_evaluation_flops=total_eval_flops,
                aspect_ratio_fn=openclip.aspect_ratio_fn,
                data_parameter_name="train_num_samples",
                max_flops=max_flops,
                min_flops=min_flops,
                best_loss_guess=0.0001
            )
        case BenchmarkEnum.LLM:
            bench = llm.get_benchmark()
            # (min_params, min_data, min_flops), (max_params, max_data, max_flops) = llm.get_macro_params_bounds(bench)
            # print(f"Macro bounds - Params: ({min_params:e}, {max_params:e}), Data: ({min_data:e}, {max_data:e}), Flops: ({min_flops:e}, {max_flops:e})")
            max_flops = 5.312364e+19
            min_flops = 3.600000e+04
            grid_diagonal = 14 + 21 # d_model + n_layers
            total_eval_flops = max_flops * grid_diagonal * 0.5
            return BenchmarkConfig(
                space=llm.get_pipeline_space(),
                params_estimator=llm.parameters_estimator,
                seen_datapoints_estimator=llm.seen_datapoints_estimator,
                flops_estimator=llm.flops_estimator,
                evaluate_pipeline=llm.evaluate_pipeline_wrapper(bench, lc_aware),
                max_evaluation_flops=total_eval_flops,
                aspect_ratio_fn=llm.aspect_ratio_fn,
                data_parameter_name="n_tokens",
                max_flops=max_flops,
                min_flops=min_flops,
                best_loss_guess=1.2,
            )
        case BenchmarkEnum.TABPFN:
            bench = tabpfn.get_benchmark()
            # (min_params, min_data, min_flops), (max_params, max_data, max_flops) = tabpfn.get_macro_params_bounds(bench)
            # print(f"Macro bounds - Params: ({min_params:e}, {max_params:e}), Data: ({min_data:e}, {max_data:e}), Flops: ({min_flops:e}, {max_flops:e})")
            max_flops = 3.850136e+15
            min_flops = 2.921580e+10
            grid_diagonal = 4 + 3 # d_model + n_layers
            total_eval_flops = max_flops * grid_diagonal * 0.5
            return BenchmarkConfig(
                space=tabpfn.get_pipeline_space(),
                params_estimator=tabpfn.parameters_estimator_wrapper(bench),
                seen_datapoints_estimator=tabpfn.seen_datapoints_estimator,
                flops_estimator=tabpfn.flops_estimator_wrapper(bench),
                evaluate_pipeline=tabpfn.evaluate_pipeline_wrapper(bench, lc_aware),
                max_evaluation_flops=total_eval_flops,
                aspect_ratio_fn=tabpfn.aspect_ratio_fn,
                data_parameter_name="total_cells",
                max_flops=max_flops,
                min_flops=min_flops,
                best_loss_guess=0.0001,
            )
        case _:
            raise Exception("Benchmark not found")

