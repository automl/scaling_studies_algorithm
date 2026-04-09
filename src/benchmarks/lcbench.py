from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
import sys

import numpy as np

from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue import Benchmark
import mfpbench

mfp_logger = logging.getLogger(__name__)

class Objective(str):
    VAL_ACCURACY = "val_accuracy"
    TIME = "time"
    VAL_CROSS_ENTROPY = "val_cross_entropy"
    VAL_BALANCED_ACCURACY = "val_balanced_accuracy"


_lcbench_task_ids = (
    "3945",
    "7593",
    "34539",
    "126025",
    "126026",
    "126029",
    "146212",
    "167104",
    "167149",
    "167152",
    "167161",
    "167168",
    "167181",
    "167184",
    "167185",
    "167190",
    "167200",
    "167201",
    "168329",
    "168330",
    "168331",
    "168335",
    "168868",
    "168908",
    "168910",
    "189354",
    "189862",
    "189865",
    "189866",
    "189873",
    "189905",
    "189906",
    "189908",
    "189909",
)

def lcbench_surrogate(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the LCBench surrogate Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in the LCBench surrogate Benchmark.
    """
    if datadir is not None and "yahpo" in os.listdir(datadir):  # noqa: PTH208
        datadir = datadir / "yahpo"

    env = Env(
        name="py310-mfpbench-1.10-yahpo",
        requirements=(
            "mf-prior-bench==1.10.0",
            "xgboost>=1.7"
        ),
    )
    with HiddenPrints():        # NOTE: To stop yahpo-lcbench from printing garbage
        for task_id in _lcbench_task_ids:
            yield BenchmarkDescription(
                name=f"yahpo-lcbench-{task_id}",
                config_space=mfpbench.get(
                    "lcbench",
                    task_id=task_id,
                    datadir=datadir,
                ).space,
                load=partial(
                    _get_surrogate_benchmark,
                    benchmark_name="lcbench",
                    datadir=datadir,
                    task_id=task_id
                ),
                metrics={
                    "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                    "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                    "val_balanced_accuracy": Measure.metric((0, 1), minimize=False),
                },
                test_metrics={
                    "test_balanced_accuracy": Measure.test_metric((0, 1), minimize=False),
                    "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
                },
                costs={
                    "time": Measure.cost((0, np.inf), minimize=True),
                },
                fidelities={
                    "epoch": RangeFidelity.from_tuple((1, 52, 1), supports_continuation=True),
                },
                env=env,
                mem_req_mb=4096,
            )

import numpy as np

# TODO: find the input_dim and num_classes, dataset_size for each dataset in LCBench
# def estimate_lcbench_flops(config, epoch, input_dim=100, num_classes=10, dataset_size=10000):
#     """
#     Estimates theoretical FLOPS for a funnel-shaped MLP in the LCBench setting.
    
#     Args:
#         config (dict): Dictionary containing:
#             - max_units (int): Width of the first hidden layer
#             - num_layers (int): Number of hidden layers
#             - batch_size (int): Batch size used during training
#         epoch (float/int): Number of epochs trained.
#         input_dim (int): Number of input features (dataset dependent). 
#                          LCBench datasets vary (e.g., Fashion-MNIST=784).
#         num_classes (int): Number of output classes (dataset dependent).

#     Returns:
#         float: Total estimated FLOPS for the training run.
#     """
#     # 1. Reconstruct the "Funnel" Architecture
#     # The first layer has 'max_units'. Subsequent layers typically halve in size.
#     # We enforce a minimum width (e.g., 16) to avoid tiny layers in deep nets.
#     units = []
#     current_width = config['max_units']
    
#     for _ in range(config['num_layers']):
#         units.append(current_width)
#         current_width = max(16, int(current_width / 2)) # Funnel decay
    
#     # 2. Calculate MACs (Multiply-Accumulates) for one forward pass
#     total_macs = 0
    
#     # Input -> First Hidden
#     total_macs += input_dim * units[0]
    
#     # Hidden -> Hidden
#     for i in range(len(units) - 1):
#         total_macs += units[i] * units[i+1]
        
#     # Last Hidden -> Output
#     total_macs += units[-1] * num_classes
    
#     # 3. Convert to FLOPS
#     # Standard approximation: 1 MAC ≈ 2 FLOPS (multiply + add)
#     forward_flops = 2 * total_macs
    
#     # Backward pass is roughly 2x the cost of forward pass
#     # (calculating gradients for weights + gradients for input)
#     total_flops_per_sample = forward_flops * 3 
    
#     # 4. Scale by Budget (Epochs * Dataset Size)
#     # Since we don't know the exact dataset size (num_samples) from just the config,
#     # we usually estimate Total FLOPS per epoch. 
#     # If you want "Cost per Epoch", we assume a fixed dataset size (e.g. 10k samples).
#     # OR if you strictly follow the config's 'batch_size' and assume 'epoch' means 
#     # "processed X batches", you need to know how many batches make an epoch.
    
#     # Assuming standard definition: 1 Epoch = Full pass over N samples.
    
#     total_training_flops = total_flops_per_sample * dataset_size * epoch
    
#     return total_training_flops

def estimate_lcbench_flops(bench: Benchmark, config, epoch) -> int:
    runtime =  bench.query(config, epoch=epoch, objectives=[Objective.TIME])[0]
    # Assume an average hardware with 1 TFLOPS performance
    return int(runtime * 1e12)
    

def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:

    with HiddenPrints():        # NOTE: To stop yahpo-lcbench from printing garbage
        if datadir is not None:
            datadir = Path(datadir).absolute().resolve()
            kwargs["datadir"] = datadir
        bench = mfpbench.get(benchmark_name, **kwargs)
    query_function = partial(_neps_mfpbench_eval_fn, benchmark=bench)
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.space,
        query=query_function,
    )

def _neps_mfpbench_eval_fn(
    benchmark: mfpbench.Benchmark,
    objectives: list[str] | None = None,
    **config: dict[str, Any]
) -> dict[str, Any]:
    fid_str = benchmark.fidelity_name
    
    fid_value = config.pop(fid_str) if fid_str in config else benchmark.fidelity_range[1]
    result = benchmark.query(config, at=fid_value).as_dict()
    if objectives is not None:
        result = {obj: result[obj] for obj in objectives if obj in result}
    return list(result.values())

class HiddenPrints:  # noqa: D101
    def __enter__(self):
        self._original_stdout = sys.stdout
        from pathlib import Path  # noqa: PLC0415
        sys.stdout = Path(os.devnull).open("w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def mfpbench_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for various MF-Prior-Bench.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each benchmark.
    """
    if datadir is None:
        datadir=Path(__file__).parent.parent.parent.absolute() / "data" / "data"
    yield from lcbench_surrogate(datadir)


mfpbench_benches = {
    desc.name: desc for desc in mfpbench_benchmarks()
}