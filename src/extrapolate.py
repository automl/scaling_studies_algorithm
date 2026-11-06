import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _openclip_arch_stats(df: pd.DataFrame) -> tuple["pd.Series", "pd.Series"]:
    """Exact (params, total_flops) for OpenCLIP rows, from the benchmark's arch table.

    data.csv rounds `mparams` to two decimals and reports a measured
    `compute_seen`, which puts the front ~0.15% off in params and ~3% in FLOPs.
    `flops_params.json` is what `VLMBenchmark.get_model_stats` reads, so taking
    the numbers from there makes the reference front exactly commensurate with
    the `extra.n_param` / `extra.flops` of the trials plotted against it.
    """
    table = json.loads(files("scan_bench.vlm").joinpath("flops_params.json").read_text())
    keys = "text_" + df["text_width"].astype(int).astype(str) + "_vision_" + df["vision_width"].astype(int).astype(str)

    params = keys.map(lambda k: (table.get(k) or {}).get("params"))
    flops_per_sample = keys.map(lambda k: (table.get(k) or {}).get("flops_per_sample"))
    total_flops = flops_per_sample * df["total_samples_planned"] * df["training_progress"]
    return params, total_flops


def _openclip_config(row: "pd.Series") -> dict[str, Any]:
    """Row of vlm/data.csv as an OpenCLIP pipeline-space config.

    Widths go back to their ordinal index, the unit the search space uses.
    """
    from src.benchmarks.openclip_space import WIDTH_CHOICES

    return {
        "beta1": float(row["beta1"]),
        "beta2": float(row["beta2"]),
        "eps": float(row["eps"]),
        "lr": float(row["lr"]),
        "warmup_fraction": float(row["warmup_fraction"]),
        "wd": float(row["wd"]),
        "text_width": WIDTH_CHOICES.index(int(row["text_width"])),
        "vision_width": WIDTH_CHOICES.index(int(row["vision_width"])),
        "train_num_samples": float(row["total_samples_planned"]),
    }


def _llm_config(row: "pd.Series") -> dict[str, Any]:
    """Row of the LLM splits as a config, in the csv's own (actual) model dims."""
    return {
        "beta1": float(row["beta1"]),
        "beta2": float(row["beta2"]),
        "cooldown_steps": float(row["cooldown_steps"]),
        "d_model": int(row["d_model"]),
        "lr": float(row["lr"]),
        "n_heads": int(row["n_heads"]),
        "n_layers": int(row["n_layers"]),
        "n_tokens": float(row["n_data"]),
        "weight_decay": float(row["weight_decay"]),
    }


class _CsvSpec:
    """How to read one benchmark's measured runs into (n_params, n_data, loss, flops).

    The reference front is built from the same measurements the benchmark's own
    surrogate is fitted on, so the "Empirical Best Loss" line is a measurement
    rather than a model output.
    """

    def __init__(
        self,
        package: str,
        csv_names: list[str],
        default_target: str,
        completed: Callable[[pd.DataFrame], "pd.Series"],
        stats: Callable[[pd.DataFrame], tuple["pd.Series", "pd.Series"]],
        n_data: Callable[[pd.DataFrame], "pd.Series"],
        config_of_row: Callable[["pd.Series"], dict[str, Any]],
        monotonic_params: bool = True,
    ):
        self.package = package
        self.csv_names = csv_names
        self.default_target = default_target
        self.completed = completed
        self.stats = stats  # df -> (n_params, flops)
        self.n_data = n_data
        self.config_of_row = config_of_row
        # Whether the front may only shrink in model size toward lower compute.
        self.monotonic_params = monotonic_params

    def load(self) -> pd.DataFrame:
        # Resolve through the installed package, the way the benchmarks load their
        # own splits, rather than through the dep/ checkout.
        frames = [pd.read_csv(str(files(self.package).joinpath(name))) for name in self.csv_names]
        return pd.concat(frames, ignore_index=True)


CSV_SPECS: dict[str, _CsvSpec] = {
    "openclip": _CsvSpec(
        package="scan_bench.vlm",
        csv_names=["data.csv"],
        default_target="val_loss",
        completed=lambda df: df["training_progress"] >= 1.0,
        stats=_openclip_arch_stats,
        n_data=lambda df: df["samples_seen"],
        config_of_row=_openclip_config,
    ),
    # Exactly what LLMBenchmark.__init__ loads.
    "llm": _CsvSpec(
        package="scan_bench.llm",
        csv_names=["splits/train.csv", "splits/test.csv"],
        default_target="valid_loss",
        completed=lambda df: df["eval_step"] == df["final_step"],
        # These splits already carry exact params/flops, no arch table needed.
        stats=lambda df: (df["n_param"], df["flops_so_far"]),
        n_data=lambda df: df["n_data"],
        config_of_row=_llm_config,
    ),
}


def _enforce_monotonic_params(pareto_front: list[tuple]) -> list[tuple]:
    if len(pareto_front) <= 1:
        return pareto_front

    descending = sorted(pareto_front, key=lambda row: row[3], reverse=True)
    monotonic = [descending[0]]
    prev_n = descending[0][0]
    for n_params, n_data, loss, flops in descending[1:]:
        if n_params <= prev_n:
            monotonic.append((n_params, n_data, loss, flops))
            prev_n = n_params
    return monotonic[::-1]


def measured_runs_from_csv(
    benchmark_name: str, target: str | None = None
) -> tuple[_CsvSpec, pd.DataFrame, pd.DataFrame]:
    """The benchmark's usable measured runs, at their final checkpoint.

    Returns (spec, raw rows, frame of n_params/n_data/loss/flops indexed like the
    raw rows). `target` names a column of the benchmark's csv -- every
    `VLMTarget` / `LLMTarget` value is one -- and defaults to that benchmark's
    validation loss. Accuracy-like targets are flipped to `1 - acc` so the front
    minimizes, the same convention `process_seed` uses for the trajectories
    plotted against it.
    """
    from src.plotting.utils import target_loss_key, to_minimizable

    spec = CSV_SPECS.get(benchmark_name)
    if spec is None:
        raise ValueError(
            f"No measured-run csv is registered for '{benchmark_name}'"
            f" (have: {', '.join(sorted(CSV_SPECS))})"
        )

    column = target or spec.default_target
    df = spec.load()
    if column not in df.columns:
        raise ValueError(f"'{column}' is not a column of the {benchmark_name} csv(s)")

    df = df[spec.completed(df)]
    n_params, flops = spec.stats(df)
    frame = pd.DataFrame({
        "n_params": pd.to_numeric(n_params, errors="coerce"),
        "n_data": pd.to_numeric(spec.n_data(df), errors="coerce"),
        "loss": pd.to_numeric(df[column], errors="coerce"),
        "flops": pd.to_numeric(flops, errors="coerce"),
    })
    frame["loss"] = frame["loss"].map(lambda v: to_minimizable(v, target_loss_key(column)))
    frame = frame[frame.apply(np.isfinite).all(axis=1) & (frame["flops"] > 0) & (frame["loss"] > 0)]
    if frame.empty:
        raise ValueError(f"No completed run of {benchmark_name} has a usable '{column}'")
    return spec, df, frame


def empirical_pareto_front_from_csv(benchmark_name: str, target: str | None = None) -> dict[str, list]:
    """Pareto front of the benchmark's measured runs (see `measured_runs_from_csv`)."""
    spec, df, frame = measured_runs_from_csv(benchmark_name, target)
    column = target or spec.default_target

    rows =list(frame.itertuples(index=True, name=None))  # (idx, n_params, n_data, loss, flops)
    # Same non-domination the optimizer uses, so this front is directly comparable
    # to the ones computed from a study's trials.
    pareto = ScalingLawGuidedOptimizer._compute_pareto_front([r[1:] for r in rows])
    if spec.monotonic_params:
        pareto = _enforce_monotonic_params(pareto)
    pareto = sorted(pareto, key=lambda row: row[3])  # ascending flops, as the monotonic pass returns it

    row_index = {r[1:]: r[0] for r in rows}
    configs = [spec.config_of_row(df.loc[row_index[point]]) for point in pareto]

    logger.info(
        "%s empirical front (%s): %d points, loss %.4f..%.4f, flops %.3e..%.3e",
        benchmark_name, column, len(pareto),
        min(p[2] for p in pareto), max(p[2] for p in pareto),
        min(p[3] for p in pareto), max(p[3] for p in pareto),
    )

    return {
        "loss": [float(p[2]) for p in pareto],
        "flops": [float(p[3]) for p in pareto],
        "n_params": [float(p[0]) for p in pareto],
        "n_data": [float(p[1]) for p in pareto],
        "configs": configs,
    }


def _pareto_info_file(output_dir: Path | str, target: str | None = None) -> Path:
    """One cached front per target: the objective's and a downstream target's
    Pareto fronts are different point sets and must not share a file."""
    if target is None:
        return Path(output_dir) / "pareto_info.json"
    from src.plotting.utils import loss_key_slug, target_loss_key

    return Path(output_dir) / f"pareto_info_{loss_key_slug(target_loss_key(target))}.json"


def save_pareto_info(pareto_info: dict, output_dir: Path | str, target: str | None = None) -> None:
    """Save Pareto front information to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = _pareto_info_file(output_dir, target)
    with open(output_file, 'w') as f:
        json.dump(pareto_info, f, indent=2)
    logger.info(f"Saved pareto_info to {output_file}")


def load_pareto_info(output_dir: Path | str, target: str | None = None) -> dict[str, list] | None:
    """Load Pareto front information from JSON file if it exists."""
    output_file = _pareto_info_file(output_dir, target)

    if output_file.exists():
        with open(output_file, 'r') as f:
            logger.info(f"Loaded pareto_info from {output_file}")
            return json.load(f)
    return None


def empirical_pareto_dir(benchmark_name: str) -> Path:
    """Cache location of a benchmark's empirical fronts.

    Deliberately *not* under a predictor directory: the front comes from measured
    runs, so it is the same whichever surrogate the plots are using.
    """
    from src.benchmarks.configs import RESULTS_BASE_DIR

    return RESULTS_BASE_DIR / "empirical_pareto" / benchmark_name


def load_or_generate_pareto_info(benchmark_name: str, target: str | None = None) -> dict[str, list] | None:
    """Empirical reference front of a benchmark's measured runs, cached as JSON.

    `target` selects a metric column (see `empirical_pareto_front_from_csv`);
    without it, the benchmark's validation loss. Returns None when the benchmark
    has no measured-run csv, so plots degrade to "no reference line" rather than
    silently drawing another metric's front.
    """
    summary_dir = empirical_pareto_dir(benchmark_name)

    pareto_info = load_pareto_info(summary_dir, target)
    if pareto_info is None:
        try:
            pareto_info = empirical_pareto_front_from_csv(benchmark_name, target=target)
        except ValueError as e:
            logger.warning(f"No empirical reference front for {benchmark_name} / '{target}': {e}")
            return None
        save_pareto_info(pareto_info, summary_dir, target)
    return pareto_info
