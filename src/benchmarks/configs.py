import os
from dataclasses import dataclass
from pathlib import Path

from scan_bench import PerformancePredictorType
import neps
from typing import Callable
from enum import Enum
import src.benchmarks.openclip_space as openclip
import src.benchmarks.llm_space as llm

class BenchmarkEnum(str, Enum):
    OPENCLIP = "openclip"
    LLM = "llm"

TARGET_ENUMS = {
    BenchmarkEnum.OPENCLIP: openclip.VLMTarget,
    BenchmarkEnum.LLM: llm.LLMTarget,
}


def get_target(benchmark: BenchmarkEnum | str, target: str | None):
    """Resolve a ``--target`` string to the target enum member of that benchmark."""
    if target is None:
        return None
    return TARGET_ENUMS[BenchmarkEnum(benchmark)](target)


RESULTS_DIR_ENV_VAR = "SCAN_RESULT_DIR"
DEFAULT_RESULTS_BASE_DIR = Path("/work/dlc2workfs2/alipourn-scan")

# Root under which every run directory, predictor cache and plot lands. Override
# per invocation with `SCAN_RESULT_DIR=/my/path python -m src.run ...`; the path
# below is only the cluster default.
RESULTS_BASE_DIR = Path(
    os.environ.get(RESULTS_DIR_ENV_VAR) or DEFAULT_RESULTS_BASE_DIR
).expanduser()

PREDICTOR_TYPE_ENV_VAR = "SCAN_PREDICTOR_TYPE"
DEFAULT_PREDICTOR_TYPE = PerformancePredictorType.TABPFN


def get_predictor_type() -> PerformancePredictorType:
    """Predictor selected via the ``--pred`` flag of the entrypoint, defaulting to TabPFN.

    Entrypoints export it through the environment so that modules building
    constants at import time (e.g. src.plotting.consts) see the same choice.
    """
    return PerformancePredictorType(
        os.environ.get(PREDICTOR_TYPE_ENV_VAR, DEFAULT_PREDICTOR_TYPE.value)
    )


def set_predictor_type(predictor_type: PerformancePredictorType) -> None:
    os.environ[PREDICTOR_TYPE_ENV_VAR] = predictor_type.value


def get_results_dir(predictor_type: PerformancePredictorType | None = None) -> Path:
    """Results root for a predictor: runs of different predictors never share a directory."""
    return RESULTS_BASE_DIR / (predictor_type or get_predictor_type()).value

@dataclass
class BenchmarkConfig:
    space: neps.PipelineSpace
    params_estimator: Callable
    seen_datapoints_estimator: Callable
    flops_estimator: Callable
    evaluate_pipeline: Callable
    query: Callable
    max_evaluation_flops: float
    aspect_ratio_fn: Callable
    data_parameter_name: str
    max_flops: float
    min_flops: float
    best_loss_guess: float

def get_benchmark(benchmark: BenchmarkEnum | str, lc_aware: bool, predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN, lazy=False, target=None) -> BenchmarkConfig:
    if isinstance(benchmark, str):
        benchmark = BenchmarkEnum(benchmark)

    match benchmark:
        case BenchmarkEnum.OPENCLIP: 
            bench = openclip.get_benchmark(predictor_type=predictor_type, target=target) if not lazy else None
            # 3.569725e+08 parameters, 2.000000e+08 data points, total flops: 7.815374e+18
            # 2.013121e+06 parameters, 6.000000e+05 data points, total flops: 3.605023e+13
            min_flops = 3.600000e+13
            max_flops = 2e+16
            total_eval_flops = max_flops * 10
            return BenchmarkConfig(
                space= openclip.get_pipeline_space(),
                params_estimator=openclip.parameters_estimator_wrapper(bench),
                seen_datapoints_estimator=openclip.seen_datapoints_estimator,
                flops_estimator=openclip.flops_estimator_wrapper(bench),
                evaluate_pipeline=openclip.evaluate_pipeline_wrapper(bench, lc_aware),
                query=openclip.query_wrapper(bench),
                max_evaluation_flops=total_eval_flops,
                aspect_ratio_fn=openclip.aspect_ratio_fn,
                data_parameter_name="train_num_samples",
                max_flops=max_flops,
                min_flops=min_flops,
                best_loss_guess=0.0001
            )
        case BenchmarkEnum.LLM:
            bench = llm.get_benchmark(predictor_type=predictor_type, target=target) if not lazy else None
            # 9.266657e+08 parameters, 4.000000e+10 data points, total flops: 2.676758e+20
            # 1.628109e+07 parameters, 2.000000e+08 data points, total flops: 2.456771e+16
            min_flops = 2.456771e+16
            max_flops = 1e+19
            total_eval_flops = max_flops * 10
            return BenchmarkConfig(
                space=llm.get_pipeline_space(),
                params_estimator=llm.parameters_estimator,
                seen_datapoints_estimator=llm.seen_datapoints_estimator,
                flops_estimator=llm.flops_estimator,
                evaluate_pipeline=llm.evaluate_pipeline_wrapper(bench, lc_aware),
                query=llm.query_wrapper(bench),
                max_evaluation_flops=total_eval_flops,
                aspect_ratio_fn=llm.aspect_ratio_fn,
                data_parameter_name="n_tokens",
                max_flops=max_flops,
                min_flops=min_flops,
                best_loss_guess=1.2,
            )
        case _:
            raise Exception("Benchmark not found")

