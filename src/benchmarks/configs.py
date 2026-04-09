from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    name: str
    # space: neps.PipelineSpace
    # params_estimator: Callable
    # seen_datapoints_estimator: Callable
    # flops_estimator: Callable
    # max_evaluation_flops: int

BENCHMARKS = {
    "synthetic_transformer": BenchmarkConfig(
        name="Synthetic Transformer Benchmark",
        # space=SyntheticMLPSpace(),
        # params_estimator=get_number_of_parameters,
        # seen_datapoints_estimator=seen_datapoints_estimator,
        # flops_estimator=get_total_flops,
        # max_evaluation_flops=max_target_flops * 32
    ),
    
    "openclip": BenchmarkConfig(
        name="OpenCLIP Benchmark",
        # space=SyntheticMLPSpace(),
        # params_estimator=get_number_of_parameters,
        # seen_datapoints_estimator=seen_datapoints_estimator,
        # flops_estimator=get_total_flops,
        # max_evaluation_flops=max_target_flops * 32
    ),
    
    "dummy": BenchmarkConfig(
        name="Dummy Benchmark",
        # space=DummySpace(),
        # params_estimator=get_number_of_parameters,
        # seen_datapoints_estimator=seen_datapoints_estimator,
        # flops_estimator=get_total_flops,
        # max_evaluation_flops=max_target_flops * 32
    )
}