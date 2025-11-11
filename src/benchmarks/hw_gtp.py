"""Script to interface with the HW-GPT-Bench library."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

if TYPE_CHECKING:
    from hpoglue.query import Query
    from hwgpt.api import HWGPT


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

size_maps = {
    "s": 192,
    "m": 256,
    "l": 320,
}

def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    predictor: Literal["mlp", "supernet"] = "mlp",
    space_size: Literal["s", "m", "l"] = "s",
    use_supernet_surrogate: bool = False,
    datadir: Path | str | None = None,
) -> SurrogateBenchmark:
    """Creates a SurrogateBenchmark from HW-GPT-Bench."""
    from hwgpt.api import HWGPT  # noqa: PLC0415

    if datadir is not None:
        datadir = Path(datadir).absolute().resolve()
    bench = HWGPT(
        search_space=space_size,
        use_supernet_surrogate=use_supernet_surrogate,
        base_path=datadir,
    )
    query_function = partial(
        _hwgptbench_surrogate_query_function,
        benchmark=bench,
        predictor=predictor,
    )
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.space,
        query=query_function,
    )


def _hwgptbench_surrogate_query_function(
        query: Query,
        benchmark: HWGPT,
        space_size: Literal["s", "m", "l"] = "s",
        predictor: Literal["mlp", "supernet"] = "mlp",
    ) -> Result:
    benchmark.set_arch(query.config.values)
    perplexity = benchmark.query(metric="perplexity", predictor=predictor)
    hwmetrics = benchmark.query(predictor=predictor).as_dict()
    all_results = {"perplexity": perplexity}
    all_results.update(hwmetrics)
    if query.fidelity is not None:
        assert isinstance(query.fidelity, tuple)
        _, fid_value = query.fidelity
    else:
        fid_value = 3 # The max in our space is 3
    fid_value = size_maps[space_size] * 2**(fid_value - 1)
    return Result(
        query=query,
        values=benchmark.query(
            query.config.values,
            at=fid_value,
        ).as_dict(),
        fidelity=query.fidelity,
    )


def hwgpt_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the HW-GPT-Bench Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each HW-GPT-Bench benchmark.
    """
    from hwgpt.api import HWGPT  # noqa: PLC0415

    hwgptbench = partial(
        HWGPT,
        use_supernet_surrogate=False,
        base_path=datadir,
    )

    env = Env(
        name="py310-hwgptbench-0.1",
        python_version="3.10",
        requirements=None,
        post_install=None,
    )
    yield BenchmarkDescription(
        name="hwgptbench-s",
        config_space=hwgptbench(search_space="s").space,
        load=partial(
            _get_surrogate_benchmark,
            predictor="mlp",
            space_size="s",
            use_supernet_surrogate=True,
            datadir=datadir,
        ),
        metrics={
            "perplexity": Measure.metric(bounds=(0, 1), minimize=True),
            "energies": Measure.metric(bounds=(0, np.inf), minimize=True),
            "latencies": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_head_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=True)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="hwgptbench-m",
        config_space=hwgptbench(search_space="m").space,
        load=partial(
            _get_surrogate_benchmark,
            predictor="mlp",
            space_size="m",
            use_supernet_surrogate=True,
            datadir=datadir,
        ),
        metrics={
            "perplexity": Measure.metric(bounds=(0, 1), minimize=True),
            "energies": Measure.metric(bounds=(0, np.inf), minimize=True),
            "latencies": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_head_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=True)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="hwgptbench-l",
        config_space=hwgptbench(search_space="l").space,
        load=partial(
            _get_surrogate_benchmark,
            predictor="mlp",
            space_size="l",
            use_supernet_surrogate=True,
            datadir=datadir,
        ),
        metrics={
            "perplexity": Measure.metric(bounds=(0, 1), minimize=True),
            "energies": Measure.metric(bounds=(0, np.inf), minimize=True),
            "latencies": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_head_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=True)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )


def hwgptbench(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for various HW-GPT-Bench.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each benchmark.
    """
    if datadir is None:
        datadir = (
            Path(__file__).parent.parent.parent.parent.parent.absolute()
            / "HW-GPT-Bench" / "data_collection"
        )
    yield from hwgpt_benchmarks(datadir=datadir)