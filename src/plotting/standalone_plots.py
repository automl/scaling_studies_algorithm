import os
import sys

if sys.platform == "darwin":
    # Avoid segfaults from conflicting OpenMP runtimes (torch/xgboost/lightgbm)
    # and macOS's post-fork Objective-C safety check.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import logging
from pathlib import Path
from typing import Any, cast
import json
from copy import deepcopy
import string

import neps
import numpy as np
import matplotlib.pyplot as plt
from neps.state.trial import State

from src.benchmarks.configs import get_benchmark, get_predictor_type
from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit
from src.plotting.plot_styles import BENCHMARK_TITLES
from src.plotting.utils import (
    get_optimizer_color,
    get_optimizer_linestyle,
    get_optimizer_label,
    _split_optimizers_by_priors,
    _placeholder_plot,
    _composite_empty_placeholder,
    _get_global_flop_bounds,
    _normalize_flop_counts,
    _get_hypervolume_bounds,
    _get_common_xlim,
    _replot_lines,
    split_by_priors,
)

# Empirical Pareto front computed from scripts/data.csv (loss=val/val_loss, flops=total_flops,
# n_param=parameters), non-dominated points sorted by FLOPs. Hardcoded here since this is the
# comprehensive ground-truth dataset, replacing the single-seed grid_search-derived front.
_EMPIRICAL_PARETO_FLOPS = np.array([
    10727293440.0, 15250965504.0, 29219162112.0, 46698451968.0, 46872168960.0,
    124048359936.0, 633350240256.0, 637579564032.0, 1805750685696.0, 2316040848384.0,
    7168117874688.0, 7892801894400.0, 14382041579520.0, 27741553582080.0, 28489145041920.0,
    31342624558080.0, 57084027856896.0, 62903361411072.0, 66544722470400.0, 133554345581568.0,
    228938713448448.0, 252576154598400.0, 442440100988928.0, 456389110984704.0, 459788497035264.0,
    877991642818560.0, 1091691656832000.0, 1762967723753472.0, 1794858024357888.0, 1848333694390272.0,
    3695874609924096.0, 7046782151347200.0, 7106065163904000.0, 2.81844992704512e+16,
])
_EMPIRICAL_PARETO_LOSS = np.array([
    2.1019073362307004, 1.8585961785477896, 1.3148221568553708, 1.29471607444187, 1.1672652121051217,
    1.1392895887547638, 1.0693727648443505, 1.0468241217458851, 1.0270189261051808, 1.0215362563806896,
    1.002541976248419, 0.9815379456995288, 0.9795148752455134, 0.9750420943528296, 0.9692769562704294,
    0.9633090722127235, 0.953561184771388, 0.9342086057295091, 0.9249831785006488, 0.9154967823706102,
    0.9131693685073212, 0.907456222892506, 0.9049312442427132, 0.9009358285851098, 0.8935840906536517,
    0.8775050381348896, 0.8697598909808827, 0.8619308802262822, 0.8547532442959588, 0.8425618700451499,
    0.820608968657325, 0.8122687703344127, 0.8078278559475317, 0.8017667817515152,
])
_EMPIRICAL_PARETO_NPARAM = np.array([
    930.0, 930.0, 6202.0, 930.0, 930.0,
    3002.0, 930.0, 930.0, 6202.0, 3002.0,
    6202.0, 3002.0, 10602.0, 39626.0, 19402.0,
    10058.0, 10602.0, 5354.0, 3002.0, 3002.0,
    37002.0, 3002.0, 141578.0, 10602.0, 10602.0,
    73610.0, 10058.0, 73610.0, 73610.0, 37002.0,
    37002.0, 141578.0, 277514.0, 141578.0,
])

logging.getLogger("fontTools").setLevel(logging.WARNING)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 20,
    "figure.figsize": (5.5, 3.5),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ProcessSeedResult = tuple[
    list[float],
    list[float],
    float,
    list[tuple[float, float, float, float]],
    list[list[tuple[float, float, float, float]]],
    list[dict[str, Any]],
]
_PROCESS_SEED_CACHE: dict[str, ProcessSeedResult] = {}


def _serialize_process_seed_input(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None,
    loss_key: str,
    return_incumbent: bool,
    flop_range: tuple[float, float] | None,
) -> str:
    normalized_flop_range = None if flop_range is None else [float(flop_range[0]), float(flop_range[1])]
    return json.dumps(
        {
            "path": str(Path(path).expanduser().resolve()),
            "seed": None if seed is None else str(seed),
            "key_to_extract": key_to_extract,
            "loss_key": loss_key,
            "return_incumbent": bool(return_incumbent),
            "flop_range": normalized_flop_range,
        },
        sort_keys=True,
    )


def _get_best_loss_guess(benchmark_name: str) -> float:
    return get_benchmark(benchmark_name, lc_aware=False, predictor_type=get_predictor_type(), lazy=True).best_loss_guess


def _get_log_scale_upper_clip(values: list[np.ndarray], quantile: float | None) -> float | None:
    if quantile is None:
        return None
    if not 0.0 < quantile <= 1.0:
        raise ValueError("y_clip_quantile must be in (0, 1] or None")

    finite_positive_values = []
    for value_array in values:
        arr = np.asarray(value_array, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) > 0:
            finite_positive_values.append(arr)

    if not finite_positive_values:
        return None

    all_values = np.concatenate(finite_positive_values)
    if len(all_values) < 3:
        return None

    return float(10 ** np.nanquantile(np.log10(all_values), quantile))


def _filter_log_scale_spikes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 2,
    max_log10_deviation: float = 0.75,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop isolated order-of-magnitude spikes before log-scale interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]

    if len(y) < min_points:
        return x, y

    log_y = np.log10(y)
    keep = np.ones(len(log_y), dtype=bool)

    for idx, value in enumerate(log_y):
        start = max(0, idx - window)
        stop = min(len(log_y), idx + window + 1)
        neighbor_mask = np.ones(stop - start, dtype=bool)
        neighbor_mask[idx - start] = False
        neighbors = log_y[start:stop][neighbor_mask]
        neighbors = neighbors[np.isfinite(neighbors)]

        if len(neighbors) < 2:
            continue

        local_median = float(np.median(neighbors))
        local_mad = float(np.median(np.abs(neighbors - local_median)))
        robust_threshold = max(max_log10_deviation, 6.0 * 1.4826 * local_mad)

        if abs(float(value) - local_median) > robust_threshold:
            keep[idx] = False

    return x[keep], y[keep]


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    return_incumbent: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> tuple[
    list[float],
    list[float],
    float,
    list[tuple[float, float, float, float]],
    list[list[tuple[float, float, float, float]]],
    list[dict[str, Any]],
]:
    cache_key = _serialize_process_seed_input(
        path=path,
        seed=seed,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        return_incumbent=return_incumbent,
        flop_range=flop_range,
    )
    if cache_key in _PROCESS_SEED_CACHE:
        return deepcopy(_PROCESS_SEED_CACHE[cache_key])

    path = Path(path)

    # flop_scale = 1e9 if "openclip" in str(path).lower() else 1.0
    flop_scale = 1.0

    if seed is not None:
        path = path / f"seed_{seed}"

    _fulldf, _summary = neps.status(path, print_summary=False)
    if _fulldf.empty:
        raise ValueError(f"No trials found in {path}")

    _fulldf = _fulldf.sort_values("time_sampled")

    def get_cost(idx: str | int) -> float:
        row = _fulldf.loc[idx]
        if key_to_extract and key_to_extract in row:
            return float(row[key_to_extract]) * flop_scale
        return float(np.nan)

    losses = []
    costs = []
    rows: list[tuple[float, float, float, float]] = []
    learning_curve_rows: list[list[tuple[float, float, float, float]]] = []
    configs: list[dict[str, Any]] = []

    max_cost: float = -1.0

    for config_id, config_result in _fulldf.iterrows():
        if config_result["state"] != State.SUCCESS:
            continue

        cost = get_cost(config_id)

        n_params = float(config_result.get("extra.n_param", np.nan))
        tokens = float(config_result.get("extra.n_data", np.nan))

        loss = config_result[loss_key]
        try:
            loss = float(loss)
            # Keep NaN/inf values as NaN for post-processing
            if np.isinf(loss):
                loss = np.nan
            # elif '_acc' in loss_key or 'recall' in loss_key:
            #     loss = 1.0 - loss
        except (ValueError, TypeError): # Handle NaN, NA, and inf values
            loss = np.nan
        losses.append(loss)
        costs.append(cost)
        rows.append((n_params, tokens, float(loss), float(cost)))

        config_dict: dict[str, Any] = {}
        for col_name, col_value in config_result.items():
            if isinstance(col_name, str) and col_name.startswith("config."):
                config_dict[col_name.removeprefix("config.")] = col_value
        configs.append(config_dict)

        learning_curve = config_result.get("learning_curve", None)
        if isinstance(learning_curve, (list, tuple)) and len(learning_curve) >= 2:
            current_curve_rows: list[tuple[float, float, float, float]] = []
            for lc_point in learning_curve:
                if not isinstance(lc_point, (list, tuple)) or len(lc_point) < 3:
                    continue

                try:
                    lc_n_data = float(lc_point[0])
                    lc_flop = float(lc_point[1]) * flop_scale
                    lc_loss = float(lc_point[2])
                    if np.isinf(lc_loss):
                        lc_loss = np.nan
                except (ValueError, TypeError):
                    continue

                current_curve_rows.append((n_params, lc_n_data, lc_loss, lc_flop))

            if len(current_curve_rows) >= 2:
                learning_curve_rows.append(current_curve_rows)
    
    cumsum_costs = list(np.cumsum(costs))

    if flop_range is not None:
        upper_bound = float(flop_range[1])
        valid_indices = [i for i, cost in enumerate(cumsum_costs) if cost <= upper_bound]
        if valid_indices:
            last_idx = valid_indices[-1] + 1
            losses = losses[:last_idx]
            costs = costs[:last_idx]
            rows = rows[:last_idx]
            cumsum_costs = cumsum_costs[:last_idx]
            configs = configs[:last_idx]

    max_cost = cumsum_costs[-1] if cumsum_costs else -1.0
    
    print(f"Processed seed {seed} for {path}, accumulated cost: {max_cost:.2e} FLOPs")
    if return_incumbent:
        result = (list(np.fmin.accumulate(losses)), cumsum_costs, max_cost, rows, learning_curve_rows, configs)
    else:
        result = (losses, cumsum_costs, max_cost, rows, learning_curve_rows, configs)
    _PROCESS_SEED_CACHE[cache_key] = deepcopy(result)
    return result


def process_optimizer_per_seed(
    *,
    base_path: str | Path,
    optimizer_name: str,
    seeds: list[int | str],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    interpolate: bool = True,
    return_incumbent: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Process all seeds for an optimizer, keeping per-seed data.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizer_name: Name of the optimizer directory
        seeds: List of seed values to process
        key_to_extract: Key to extract from results as cost (typically 'flopcount')
        log_scale: Whether to interpolate on log scale for flop counts
        interpolate: Whether to interpolate losses onto a shared 100-point FLOP grid
        return_incumbent: Whether to convert losses to incumbent trajectory
            (cumulative minimum). If False, keeps per-trial raw losses.
    
    Returns:
        flop_grid: Common flop count grid (x-axis). If interpolate=False, this is
            the raw cumulative-cost axis for the first valid seed.
        per_seed_losses_dict: Dict mapping seed -> losses (interpolated or raw)
        per_seed_costs_dict: Dict mapping seed -> per-sample cumulative costs
    """
    base_path = Path(base_path)
    optimizer_path = base_path / optimizer_name
    
    all_seeds_data = []
    valid_seeds = []
    
    # Process each seed
    for seed in seeds:
        # try:
        losses, costs, max_cost, _rows, _learning_curve_rows, _configs = process_seed(
            path=optimizer_path,
            seed=seed,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            return_incumbent=return_incumbent,
            flop_range=flop_range,
        )
        all_seeds_data.append({"losses": losses, "costs": costs, "max_cost": max_cost})
        valid_seeds.append(seed)
        # except Exception as e:
        #     print(f"Warning: Could not process {optimizer_name} seed {seed}: {e}")
        #     continue
    
    if not all_seeds_data:
        raise ValueError(f"No valid seeds found for optimizer {optimizer_name}")
    
    per_seed_losses = {}
    per_seed_costs = {}

    if interpolate:
        # Create common grid for interpolation
        max_cost_overall = max(d["max_cost"] for d in all_seeds_data)

        if log_scale:
            # Log scale interpolation
            flop_grid = np.logspace(
                np.log10(min(c[0] for d in all_seeds_data for c in [d["costs"]] if c)),
                np.log10(max_cost_overall),
                num=100
            )
        else:
            # Linear scale interpolation
            flop_grid = np.linspace(0, max_cost_overall, num=100)
    else:
        flop_grid = np.array([], dtype=float)

    for seed, seed_data in zip(valid_seeds, all_seeds_data):
        costs = np.array(seed_data["costs"])
        losses = np.array(seed_data["losses"])

        if interpolate:
            finite_mask = np.isfinite(losses)
            if not np.any(finite_mask):
                continue  # all trials failed for this seed
            interp_losses = np.interp(
                flop_grid,
                costs[finite_mask],
                losses[finite_mask],
                left=losses[finite_mask][0],
                right=losses[finite_mask][-1],
            )
            per_seed_losses[seed] = interp_losses
        else:
            per_seed_losses[seed] = losses
        per_seed_costs[seed] = costs

    if not interpolate:
        # Keep exact trial count/order for downstream uses such as extrapolation.
        flop_grid = per_seed_costs[valid_seeds[0]].astype(float)
    
    return flop_grid, per_seed_losses, per_seed_costs


def process_optimizer(
    *,
    base_path: str | Path,
    optimizer_name: str,
    seeds: list[int] | list[int | str],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    interpolate: bool = True,
    return_incumbent: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flop_grid, per_seed_losses, per_seed_costs = process_optimizer_per_seed(
        base_path=base_path,
        optimizer_name=optimizer_name,
        seeds=seeds,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
        interpolate=interpolate,
        return_incumbent=return_incumbent,
        flop_range=flop_range,
    )

    # Stack per-seed losses and calculate mean/standard error
    if interpolate:
        interpolated_losses_array = np.array(list(per_seed_losses.values()))
    else:
        # Align raw per-seed losses by trial index to preserve exact trial count.
        # Different seeds may have different numbers of trials; trim to shared prefix.
        min_len = min(len(v) for v in per_seed_losses.values())
        interpolated_losses_array = np.array([v[:min_len] for v in per_seed_losses.values()])
        first_seed = next(iter(per_seed_costs))
        flop_grid = per_seed_costs[first_seed][:min_len]
    
    # Use nanmean and nanstd to ignore NaN values
    mean_losses = np.nanmean(interpolated_losses_array, axis=0)
    std_losses = np.nanstd(interpolated_losses_array, axis=0)
    
    # Count valid (non-NaN) values for each position
    valid_counts = np.sum(~np.isnan(interpolated_losses_array), axis=0)
    # Avoid division by zero
    valid_counts = np.maximum(valid_counts, 1)
    se_losses = std_losses / np.sqrt(valid_counts)
    
    # Convert to regular float arrays to ensure matplotlib compatibility
    return flop_grid.astype(float), mean_losses.astype(float), se_losses.astype(float)


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs" if kw.get("normalize_flops", True) else "Cumulative FLOPs",
        ylabel="Incumbent Loss",
    )
)
def plot_optimizer_loss(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
) -> Any:
    fig, ax = plt.subplots(figsize=figsize)

    normalized_range = (
        flop_range is not None
        and 0.0 <= flop_range[0] <= 1.0
        and 0.0 <= flop_range[1] <= 1.0
    )
    effective_flop_range = flop_range if (not normalize_flops or normalized_range) else None
    
    # First pass: collect all data.
    all_data = []
    all_flop_grids: list[np.ndarray] = []
    
    for opt_name, seeds in optimizers.items():
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        
        flop_counts, mean_losses, std_losses = process_optimizer(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            return_incumbent=True,
            flop_range=flop_range,
        )
        
        if len(flop_counts) > 0:
            all_data.append({
                'opt_name': opt_name,
                'color': color,
                'linestyle': linestyle,
                'flop_counts': flop_counts,
                'mean_losses': mean_losses,
                'std_losses': std_losses,
            })
            all_flop_grids.append(flop_counts)

    if normalize_flops:
        bounds = _get_global_flop_bounds(all_flop_grids)
        if bounds is not None:
            min_flop, max_flop = bounds
            for data in all_data:
                data['flop_counts'] = _normalize_flop_counts(data['flop_counts'], min_flop, max_flop)

    # Optional trim after potential normalization.
    if effective_flop_range is not None:
        for data in all_data:
            flop_counts = data['flop_counts']
            mean_losses = data['mean_losses']
            std_losses = data['std_losses']
            min_flops, max_flops = effective_flop_range
            mask = (flop_counts >= min_flops) & (flop_counts <= max_flops)
            data['flop_counts'] = flop_counts[mask]
            data['mean_losses'] = mean_losses[mask]
            data['std_losses'] = std_losses[mask]

    # Find common start among available curves.
    all_first_flops = []
    for data in all_data:
        flop_counts = data['flop_counts']
        non_zero_flops = flop_counts[flop_counts > 1e-8]
        if len(non_zero_flops) > 0:
            all_first_flops.append(float(non_zero_flops[0]))
    
    # Calculate common starting point (max of all first flops, excluding 0.0)
    # common_start = max(all_first_flops) if all_first_flops else 0
    if effective_flop_range is not None:
        # If zooming in, stretch incumbent lines all the way to the left border
        common_start = effective_flop_range[0]
    else:
        # Otherwise, find the absolute earliest start point
        common_start = min(all_first_flops) if all_first_flops else 0.0
    
    # Second pass: plot with common starting point
    x_lim_min = [0.0]
    x_lim_max = [1.0]
    for data in all_data:
        flop_counts = data['flop_counts']
        mean_losses = data['mean_losses']
        std_losses = data['std_losses']
        
        # Trim data to common_start
        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        mean_losses = mean_losses[mask]
        std_losses = std_losses[mask]
        
        # Ensure all lines start from common_start by prepending first value if needed
        if len(flop_counts) > 0 and flop_counts[0] > common_start + 1e-8:
            flop_counts = np.concatenate([[common_start], flop_counts])
            mean_losses = np.concatenate([[mean_losses[0]], mean_losses])
            std_losses = np.concatenate([[std_losses[0]], std_losses])
        
        if len(flop_counts) > 0:
            # Plot mean loss
            ax.plot(
                flop_counts,
                mean_losses,
                label=get_optimizer_label(data['opt_name']),
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2.5,
            )
            x_lim_min.append(flop_counts[0])
            x_lim_max.append(flop_counts[-1])
            
            # Add confidence band (standard error)
            ax.fill_between(
                flop_counts,
                mean_losses - std_losses,
                mean_losses + std_losses,
                alpha=0.2,
                color=data['color'],
            )
    
    ax.set_xlabel("Normalized FLOPs" if normalize_flops else "Cumulative FLOPs", fontsize=20)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        ax.set_xlim(max(x_lim_min), min(x_lim_max))
    ax.set_ylabel("Incumbent Loss", fontsize=20)
    # ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if not normalize_flops:
        ax.set_xscale("log")
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


def calculate_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    ranking_metric: str = "incumbent",
) -> dict[str, dict]:
    """
    Calculate relative rankings across optimizers at each cost point using per-seed data.
    
    Calculates ranking for each seed, then aggregates (mean/std across seeds).
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        log_scale: Whether to use log scale
    
    Returns:
        rankings: Dict with optimizer names as keys
                Values: dicts with 'flop_counts', 'mean_ranks', 'std_ranks'
    """
    ranking_metric = ranking_metric.lower()
    if ranking_metric not in {"incumbent", "hypervolume"}:
        raise ValueError("ranking_metric must be either 'incumbent' or 'hypervolume'")

    # Get per-seed data for all optimizers
    all_per_seed_data = {}
    for opt_name, seeds in optimizers.items():
        flop_grid, per_seed_losses, _ = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            return_incumbent=(ranking_metric == "incumbent"),
        )

        per_seed_values = per_seed_losses
        if ranking_metric == "hypervolume":
            from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

            per_seed_hvs = {}
            for seed, losses in per_seed_losses.items():
                res = ScalingLawGuidedOptimizer.compute_normalized_search_hypervolume(
                    losses.tolist(),
                    flop_grid.tolist(),
                )
                if res is None:
                    per_seed_hvs[seed] = np.full_like(losses, np.nan, dtype=float)
                else:
                    _, hvs = res
                    per_seed_hvs[seed] = np.array(hvs, dtype=float)
            per_seed_values = per_seed_hvs

        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_values": per_seed_values,
        }
    
    # Calculate rankings per seed, then aggregate
    rankings = {}
    
    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_values = data["per_seed_values"]
        
        # Calculate ranking for each seed
        seed_rankings_list = []
        
        for seed, values in per_seed_values.items():
            seed_ranks = []
            
            for i in range(len(flop_grid)):
                # Get values from all optimizers at this flop index
                values_at_idx = []
                for other_opt in all_per_seed_data.keys():
                    try:
                        other_values = all_per_seed_data[other_opt]["per_seed_values"][seed]
                    except KeyError as e:
                        print(f"other_opt {other_opt} does not have seed {seed}, skipping for ranking at index {i}")
                        raise e
                    if i < len(other_values):
                        values_at_idx.append(other_values[i])
                
                # Rank this optimizer.
                # incumbent: lower is better; hypervolume: higher is better.
                if values_at_idx:
                    valid_values_at_idx = [v for v in values_at_idx if not np.isnan(v)]

                    current_value = values[i]
                    if np.isnan(current_value):
                        rank = len(valid_values_at_idx) + 1
                    elif valid_values_at_idx:
                        if ranking_metric == "incumbent":
                            rank = sum(1 for v in valid_values_at_idx if v < current_value) + 1
                        else:
                            rank = sum(1 for v in valid_values_at_idx if v > current_value) + 1
                    else:
                        rank = 1
                    
                    seed_ranks.append(rank)
            
            seed_rankings_list.append(np.array(seed_ranks))
        
        # Aggregate rankings across seeds
        seed_rankings_array = np.array(seed_rankings_list)
        mean_ranks = np.mean(seed_rankings_array, axis=0)
        std_ranks = np.std(seed_rankings_array, axis=0)
        se_ranks = std_ranks / np.sqrt(len(seed_rankings_list))
        
        rankings[opt_name] = {
            "flop_counts": flop_grid[:len(mean_ranks)],
            "mean_ranks": mean_ranks,
            "std_ranks": se_ranks,
            "seed_ranks": seed_rankings_list
        }
    
    return rankings


def calculate_normalized_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    ranking_metric: str = "incumbent",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
) -> dict[str, dict]:
    """
    Calculate relative rankings with normalized flop counts for cross-benchmark comparison.
    """
    ranking_metric = ranking_metric.lower()
    if ranking_metric not in {"incumbent", "hypervolume", "pareto_area", "cwel"}:
        raise ValueError("ranking_metric must be one of 'incumbent', 'hypervolume', 'pareto_area', or 'cwel'")
    if ranking_metric == "pareto_area" and true_pareto_info is None:
        raise ValueError("ranking_metric='pareto_area' requires true_pareto_info")
    if ranking_metric == "cwel" and true_pareto_info is None:
        raise ValueError("ranking_metric='cwel' requires true_pareto_info")

    # Get raw per-seed data for all optimizers. Ranking must happen after each
    # optimizer/seed metric curve is interpolated onto the same normalized FLOP
    # grid; ranking by raw array index compares different FLOP budgets.
    all_per_seed_data: dict[str, dict[str, Any]] = {}
    all_flop_grids: list[np.ndarray] = []
    for opt_name, seeds in optimizers.items():
        _flop_grid, per_seed_losses, per_seed_costs = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            interpolate=False,
            return_incumbent=(ranking_metric == "incumbent"),
            flop_range=flop_range,
        )

        all_per_seed_data[opt_name] = {
            "per_seed_losses": per_seed_losses,
            "per_seed_costs": per_seed_costs,
        }
        all_flop_grids.extend(np.asarray(costs, dtype=float) for costs in per_seed_costs.values())

    if not all_flop_grids:
        return {}

    # Normalize flop grid for the benchmark using log scale
    min_flop = min(grid[grid > 1e-8][0] for grid in all_flop_grids if len(grid[grid > 1e-8]) > 0)
    max_flop = max(grid.max() for grid in all_flop_grids)
    
    # Handle edge case where min_flop == max_flop
    if np.isclose(min_flop, max_flop):
        # All flops are essentially the same, no normalization needed
        normalized_flop_grid_scaled = np.ones(100) * 0.5  # Middle value
        normalized_flop_grid = np.linspace(min_flop, max_flop, 100)
    else:
        # Create normalized grid in log space
        normalized_flop_grid = np.logspace(
            np.log10(min_flop), np.log10(max_flop), num=100
        )
        # Normalize to 0-1 range for consistency
        normalized_flop_grid_scaled = (np.log10(normalized_flop_grid) - np.log10(min_flop)) / (np.log10(max_flop) - np.log10(min_flop))

    interpolated_values_by_opt_seed: dict[str, dict[int | str, np.ndarray]] = {}

    if ranking_metric in {"hypervolume", "pareto_area", "cwel"}:
        from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

        loss_bounds, hv_flops_bounds = _get_hypervolume_bounds(all_per_seed_data)

        # Extend bounds to include true Pareto so its normalized coordinates stay in [0,1].
        if ranking_metric in {"pareto_area", "cwel"} and true_pareto_info is not None:
            _tp_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
            _tp_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
            _tp_valid = np.isfinite(_tp_flops) & np.isfinite(_tp_losses) & (_tp_flops > 0) & (_tp_losses > 0)
            if np.any(_tp_valid):
                _tp_log_flops = np.log10(np.maximum(1.0, _tp_flops[_tp_valid]))
                if hv_flops_bounds is not None:
                    hv_flops_bounds = (
                        min(hv_flops_bounds[0], float(np.min(_tp_log_flops))),
                        max(hv_flops_bounds[1], float(np.max(_tp_log_flops))),
                    )
                else:
                    hv_flops_bounds = (float(np.min(_tp_log_flops)), float(np.max(_tp_log_flops)))
                if loss_bounds is not None:
                    loss_bounds = (
                        min(loss_bounds[0], float(np.min(_tp_losses[_tp_valid]))),
                        max(loss_bounds[1], float(np.max(_tp_losses[_tp_valid]))),
                    )
                else:
                    loss_bounds = (float(np.min(_tp_losses[_tp_valid])), float(np.max(_tp_losses[_tp_valid])))

        for opt_name, data in all_per_seed_data.items():
            interpolated_values_by_opt_seed[opt_name] = {}
            per_seed_losses = data["per_seed_losses"]
            per_seed_costs = data["per_seed_costs"]

            for seed, losses in per_seed_losses.items():
                seed_costs = np.asarray(per_seed_costs[seed], dtype=float)
                if ranking_metric == "hypervolume":
                    res = ScalingLawGuidedOptimizer.compute_normalized_search_hypervolume(
                        np.asarray(losses, dtype=float).tolist(),
                        seed_costs.tolist(),
                        global_loss_bounds=loss_bounds,
                        global_flops_bounds=hv_flops_bounds,
                    )
                elif ranking_metric == "cwel":
                    # Compute CWEL directly. If log_scale is enabled, both the
                    # interpolation coordinate and the weighting denominator use
                    # log10(FLOPs), while the returned x-axis stays in raw FLOPs.
                    losses_arr = np.asarray(losses, dtype=float)
                    positive_mask = np.isfinite(seed_costs) & (seed_costs > 1e-8) & np.isfinite(losses_arr)
                    if np.sum(positive_mask) < 2:
                        continue

                    seed_costs_pos = seed_costs[positive_mask]
                    losses_pos = losses_arr[positive_mask]

                    true_flops = np.asarray(cast(dict[str, list], true_pareto_info).get("flops", []), dtype=float)
                    true_losses_arr = np.asarray(cast(dict[str, list], true_pareto_info).get("loss", []), dtype=float)
                    true_valid = np.isfinite(true_flops) & np.isfinite(true_losses_arr)
                    true_flops = true_flops[true_valid]
                    true_losses_arr = true_losses_arr[true_valid]
                    sort_idx = np.argsort(true_flops)
                    true_flops = true_flops[sort_idx]
                    true_losses_arr = true_losses_arr[sort_idx]

                    true_positive_mask = true_flops > 1e-8
                    seed_metric_flops = np.log10(np.maximum(seed_costs_pos, 1.0))
                    true_metric_flops = np.log10(np.maximum(true_flops[true_positive_mask], 1.0))
                    true_losses_metric = true_losses_arr[true_positive_mask]

                    if len(seed_metric_flops) < 2 or len(true_metric_flops) < 2:
                        continue

                    true_pareto_losses_at_seed_flops = np.interp(
                        seed_metric_flops,
                        true_metric_flops,
                        true_losses_metric
                    )

                    excess_loss = np.abs(losses_pos - true_pareto_losses_at_seed_flops)

                    raw_compute_weights = seed_costs_pos

                    weighted_errors = raw_compute_weights * excess_loss
                    cumsum_weighted_errors = np.cumsum(weighted_errors)
                    cumsum_compute = np.cumsum(raw_compute_weights)

                    # 4. Final CWEL Calculation
                    cwel_values = cumsum_weighted_errors / np.maximum(cumsum_compute, 1e-10)

                    res = (seed_costs_pos.tolist(), cwel_values.tolist())
                else:  # pareto_area
                    res = ScalingLawGuidedOptimizer.compute_normalized_pareto_line_area_to_true(
                        np.asarray(losses, dtype=float).tolist(),
                        seed_costs.tolist(),
                        true_pareto_info=cast(dict[str, list], true_pareto_info),
                        global_loss_bounds=loss_bounds,
                        global_flops_bounds=hv_flops_bounds,
                    )
                if res is None:
                    continue

                metric_costs, metric_values = res
                original_flop_grid = np.asarray(metric_costs, dtype=float)
                values = np.asarray(metric_values, dtype=float)
                if ranking_metric == "hypervolume":
                    values = np.maximum.accumulate(values)

                positive_mask = original_flop_grid > 1e-8
                finite_mask = positive_mask & np.isfinite(original_flop_grid) & np.isfinite(values)
                if np.sum(finite_mask) < 2:
                    continue

                original_flop_grid = original_flop_grid[finite_mask]
                values = values[finite_mask]
                if np.isclose(min_flop, max_flop):
                    original_normalized_flops = np.ones_like(original_flop_grid) * 0.5
                else:
                    original_normalized_flops = (
                        (np.log10(original_flop_grid) - np.log10(min_flop))
                        / (np.log10(max_flop) - np.log10(min_flop))
                    )

                order = np.argsort(original_normalized_flops)
                original_normalized_flops = original_normalized_flops[order]
                values = values[order]
                original_normalized_flops, unique_idx = np.unique(original_normalized_flops, return_index=True)
                values = values[unique_idx]
                if len(original_normalized_flops) < 2:
                    continue

                interpolated_values_by_opt_seed[opt_name][seed] = np.interp(
                    normalized_flop_grid_scaled,
                    original_normalized_flops,
                    values,
                    left=values[0],
                    right=values[-1],
                )
    else:
        for opt_name, data in all_per_seed_data.items():
            interpolated_values_by_opt_seed[opt_name] = {}
            per_seed_losses = data["per_seed_losses"]
            per_seed_costs = data["per_seed_costs"]

            for seed, values in per_seed_losses.items():
                original_flop_grid = np.asarray(per_seed_costs[seed], dtype=float)
                values = np.asarray(values, dtype=float)

                finite_mask = np.isfinite(original_flop_grid) & np.isfinite(values) & (original_flop_grid > 1e-8)
                if np.sum(finite_mask) < 2:
                    continue

                original_flop_grid = original_flop_grid[finite_mask]
                values = values[finite_mask]
                if np.isclose(min_flop, max_flop):
                    original_normalized_flops = np.ones_like(original_flop_grid) * 0.5
                else:
                    original_normalized_flops = (
                        (np.log10(original_flop_grid) - np.log10(min_flop))
                        / (np.log10(max_flop) - np.log10(min_flop))
                    )

                order = np.argsort(original_normalized_flops)
                original_normalized_flops = original_normalized_flops[order]
                values = values[order]
                original_normalized_flops, unique_idx = np.unique(original_normalized_flops, return_index=True)
                values = values[unique_idx]
                if len(original_normalized_flops) < 2:
                    continue

                interpolated_values_by_opt_seed[opt_name][seed] = np.interp(
                    normalized_flop_grid_scaled,
                    original_normalized_flops,
                    values,
                    left=values[0],
                    right=values[-1],
                )

    rankings = {}
    for opt_name, per_seed_values in interpolated_values_by_opt_seed.items():
        seed_rankings_list = []
        for seed, values in per_seed_values.items():
            seed_ranks = []
            for i in range(len(normalized_flop_grid_scaled)):
                values_at_idx = []
                for other_opt in interpolated_values_by_opt_seed.keys():
                    try:
                        other_values = interpolated_values_by_opt_seed[other_opt][seed]
                        if i < len(other_values):
                            values_at_idx.append(other_values[i])
                    except KeyError:
                        continue
                
                valid_values_at_idx = [v for v in values_at_idx if np.isfinite(v)]
                current_value = values[i]
                if not np.isfinite(current_value):
                    rank = len(valid_values_at_idx) + 1
                elif valid_values_at_idx:
                    if ranking_metric == "hypervolume":
                        rank = sum(1 for v in valid_values_at_idx if v > current_value) + 1
                    else:
                        rank = sum(1 for v in valid_values_at_idx if v < current_value) + 1
                else:
                    rank = 1
                seed_ranks.append(rank)
            
            seed_rankings_list.append(np.asarray(seed_ranks, dtype=float))
        
        if not seed_rankings_list:
            continue

        seed_rankings_array = np.array(seed_rankings_list)
        mean_ranks = np.mean(seed_rankings_array, axis=0)
        std_ranks = np.std(seed_rankings_array, axis=0)
        se_ranks = std_ranks / np.sqrt(len(seed_rankings_list))
        
        rankings[opt_name] = {
            "flop_counts": normalized_flop_grid,
            "flop_counts_scaled": normalized_flop_grid_scaled,
            "mean_ranks": mean_ranks,
            "std_ranks": se_ranks,
            "seed_ranks": seed_rankings_list
        }
        
    return rankings


def calculate_hypervolumes(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """
    Calculate hypervolume with normalized flop counts for cross-benchmark comparison.
    """
    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
    max_hypervolume = 1.1 * 1.1
    
    # First pass: collect raw chronological per-seed data. Hypervolume must be
    # computed before interpolation because the implementation recovers each
    # model's actual cost from adjacent cumulative FLOP values.
    all_flop_grids: list[np.ndarray] = []
    all_per_seed_data: dict[str, dict[str, Any]] = {}
    
    for opt_name, seeds in optimizers.items():
        raw_flop_grid, per_seed_losses, per_seed_costs = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            interpolate=False,
            flop_range=flop_range,
        )
        max_cost = max((float(np.max(costs)) for costs in per_seed_costs.values() if len(costs) > 0), default=0.0)
        positive_min = min(
            (float(np.min(costs[costs > 1e-8])) for costs in per_seed_costs.values() if np.any(costs > 1e-8)),
            default=0.0,
        )
        if log_scale and positive_min > 0 and max_cost > positive_min:
            flop_grid = np.logspace(np.log10(positive_min), np.log10(max_cost), num=100)
        else:
            flop_grid = np.linspace(0.0, max_cost, num=100)

        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_losses": per_seed_losses,
            "per_seed_costs": per_seed_costs,
        }
        for seed_costs in per_seed_costs.values():
            if len(seed_costs) > 0:
                all_flop_grids.append(np.asarray(seed_costs, dtype=float))
        if not per_seed_costs and len(raw_flop_grid) > 0:
            all_flop_grids.append(raw_flop_grid)
        elif not per_seed_costs and len(flop_grid) > 0:
            all_flop_grids.append(flop_grid)

    bounds = _get_global_flop_bounds(all_flop_grids)
    loss_bounds, hv_flops_bounds = _get_hypervolume_bounds(all_per_seed_data)
    
    hvs_dict = {}

    # Second pass: calculate raw per-seed hypervolume and interpolate only the
    # resulting trajectory for aggregation/plotting.
    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        per_seed_costs = data["per_seed_costs"]
        
        seed_hvs_list = []
        
        for seed, losses in per_seed_losses.items():
            seed_costs = np.asarray(per_seed_costs[seed], dtype=float)
            res = ScalingLawGuidedOptimizer.compute_normalized_search_hypervolume(
                np.asarray(losses, dtype=float).tolist(),
                seed_costs.tolist(),
                global_loss_bounds=loss_bounds,
                global_flops_bounds=hv_flops_bounds,
            )
            if res is None:
                continue
            hv_costs, hvs = res
            hvs = np.clip(np.maximum.accumulate(np.asarray(hvs, dtype=float)), 0.0, max_hypervolume)
            seed_hvs = np.interp(
                flop_grid,
                np.asarray(hv_costs, dtype=float),
                hvs,
                left=float(hvs[0]),
                right=float(hvs[-1]),
            )
            seed_hvs_list.append(seed_hvs)

        if not seed_hvs_list:
            continue
        
        # Aggregate across seeds
        seed_hvs_array = np.array(seed_hvs_list)
        mean_hvs = np.nanmean(seed_hvs_array, axis=0)
        mean_hvs = np.clip(np.maximum.accumulate(mean_hvs), 0.0, max_hypervolume)
        std_hvs = np.nanstd(seed_hvs_array, axis=0)
        
        valid_counts = np.sum(~np.isnan(seed_hvs_array), axis=0)
        valid_counts = np.maximum(valid_counts, 1)
        se_hvs = std_hvs / np.sqrt(valid_counts)
        
        hvs_dict[opt_name] = {
            "flop_counts": flop_grid,
            "norm_flops": _normalize_flop_counts(flop_grid, bounds[0], bounds[1]) if bounds is not None else flop_grid,
            "mean_hvs": mean_hvs.astype(float),
            "std_hvs": se_hvs.astype(float),
        }
    
    return hvs_dict


def calculate_pareto_line_areas_to_true(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    true_pareto_info: dict[str, list],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """Calculate mean area between current fitted Pareto line and true Pareto line."""
    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

    all_flop_grids: list[np.ndarray] = []
    all_per_seed_data: dict[str, dict[str, Any]] = {}

    for opt_name, seeds in optimizers.items():
        raw_flop_grid, per_seed_losses, per_seed_costs = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            interpolate=False,
            flop_range=flop_range,
        )
        if flop_range is not None:
            grid_min, grid_max = float(flop_range[0]), float(flop_range[1])
        else:
            grid_min = 0.0
            grid_max = max((float(np.max(costs)) for costs in per_seed_costs.values() if len(costs) > 0), default=0.0)

        positive_min = min(
            (float(np.min(costs[costs > 1e-8])) for costs in per_seed_costs.values() if np.any(costs > 1e-8)),
            default=0.0,
        )
        log_grid_min = max(grid_min, positive_min)
        if log_scale and log_grid_min > 0 and grid_max > log_grid_min:
            flop_grid = np.logspace(np.log10(log_grid_min), np.log10(grid_max), num=100)
        else:
            flop_grid = np.linspace(grid_min, grid_max, num=100)

        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_losses": per_seed_losses,
            "per_seed_costs": per_seed_costs,
        }
        for seed_costs in per_seed_costs.values():
            if len(seed_costs) > 0:
                all_flop_grids.append(np.asarray(seed_costs, dtype=float))
        if not per_seed_costs and len(raw_flop_grid) > 0:
            all_flop_grids.append(raw_flop_grid)
        elif not per_seed_costs and len(flop_grid) > 0:
            all_flop_grids.append(flop_grid)

    bounds = flop_range if flop_range is not None else _get_global_flop_bounds(all_flop_grids)

    area_dict: dict[str, dict[str, np.ndarray]] = {}

    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        per_seed_costs = data["per_seed_costs"]

        seed_area_list = []
        for seed, losses in per_seed_losses.items():
            seed_costs = np.asarray(per_seed_costs[seed], dtype=float)
            res = ScalingLawGuidedOptimizer.compute_normalized_pareto_line_area_to_true(
                np.asarray(losses, dtype=float).tolist(),
                seed_costs.tolist(),
                true_pareto_info=true_pareto_info,
                log_loss_scale=True,
            )
            if res is None:
                continue

            area_costs, areas = res
            areas_arr = np.asarray(areas, dtype=float)
            area_costs_arr = np.asarray(area_costs, dtype=float)
            finite = np.isfinite(area_costs_arr) & np.isfinite(areas_arr)
            if not np.any(finite):
                continue
            area_costs_arr = area_costs_arr[finite]
            areas_arr = areas_arr[finite]
            if len(area_costs_arr) == 0:
                continue

            seed_area = np.interp(
                flop_grid,
                area_costs_arr,
                areas_arr,
                left=float(areas_arr[0]),
                right=float(areas_arr[-1]),
            )
            seed_area_list.append(seed_area)

        if not seed_area_list:
            continue

        seed_area_array = np.asarray(seed_area_list, dtype=float)
        mean_areas = np.nanmean(seed_area_array, axis=0)
        std_areas = np.nanstd(seed_area_array, axis=0)
        valid_counts = np.maximum(np.sum(np.isfinite(seed_area_array), axis=0), 1)
        se_areas = std_areas / np.sqrt(valid_counts)

        area_dict[opt_name] = {
            "flop_counts": flop_grid,
            "norm_flops": _normalize_flop_counts(flop_grid, bounds[0], bounds[1]) if bounds is not None else flop_grid,
            "mean_areas": mean_areas.astype(float),
            "std_areas": se_areas.astype(float),
        }

    return area_dict


def calculate_pareto_fronts(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    acc_flop_ranges: tuple[float, float] | None = None,
) -> dict[str, dict]:
    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

    base_path = Path(base_path)

    all_per_seed_rows: dict[str, dict[int | str, list[tuple[float, float, float, float]]]] = {}

    for opt_name, seeds in optimizers.items():
        seed_rows: dict[int | str, list[tuple[float, float, float, float]]] = {}
        for seed in seeds:
            _losses, _costs, _max_cost, rows, _learning_curve_rows, _configs = process_seed(
                path=base_path / opt_name,
                seed=seed,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                return_incumbent=False,
                flop_range=acc_flop_ranges,
            )
            seed_rows[seed] = rows

        all_per_seed_rows[opt_name] = seed_rows

    pareto_dict = {}
    for opt_name, seed_rows in all_per_seed_rows.items():

        pooled_front_flops = []
        pooled_front_losses = []
        per_seed_fronts: dict[int | str, dict[str, np.ndarray]] = {}

        for seed, rows in seed_rows.items():
            if not rows:
                continue

            rows_arr = np.asarray(rows, dtype=float)
            if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
                continue

            n_params = rows_arr[:, 0]
            tokens = rows_arr[:, 1]
            losses_arr = rows_arr[:, 2]
            actual_trial_flops = rows_arr[:, 3]

            # Ensure we only pass finite values to the Pareto calculator
            mask = (
                np.isfinite(n_params)
                & np.isfinite(tokens)
                & np.isfinite(losses_arr)
                & np.isfinite(actual_trial_flops)
                & (actual_trial_flops > 0)
            )
            if not np.any(mask):
                continue

            res = ScalingLawGuidedOptimizer.compute_search_pareto_front(
                losses_arr[mask].tolist(),
                actual_trial_flops[mask].tolist(),
            )
            
            if res:
                f_points, l_points, idx_points = res
                pooled_front_flops.extend(f_points)
                pooled_front_losses.extend(l_points)
                per_seed_fronts[seed] = {
                    "flop_counts": np.asarray(f_points, dtype=float),
                    "front_losses": np.asarray(l_points, dtype=float),
                }

        if not pooled_front_flops:
            continue

        pooled_flops_arr = np.asarray(pooled_front_flops, dtype=float)
        pooled_losses_arr = np.asarray(pooled_front_losses, dtype=float)
        pareto_dict[opt_name] = {
            "flop_counts": pooled_flops_arr.astype(float),
            "front_losses": pooled_losses_arr.astype(float),
            "per_seed_fronts": per_seed_fronts,
        }

    return pareto_dict


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs" if kw.get("normalize_flops", False) else "Cumulative FLOPs",
        ylabel="Loss",
    )
)
def plot_pareto_front_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Pareto Front Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    pareto_acc_flop_ranges: tuple[float, float] | None = None,
    normalize_flops: bool = False,
    true_pareto_info: dict[str, list] | None = None,
    pareto_target_flop: float | None = None,
    selected_seed: int | str | None = None,
) -> Any:
    """Plot Pareto-front loss over FLOPs for one selected seed per optimizer."""
    if pareto_acc_flop_ranges is not None and pareto_acc_flop_ranges[0] >= pareto_acc_flop_ranges[1]:
        raise ValueError("pareto_acc_flop_ranges must satisfy min < max")

    if pareto_target_flop is not None and pareto_target_flop <= 0:
        raise ValueError("pareto_target_flop must be positive when provided")

    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer

    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)

    all_data = []
    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed = selected_seed if selected_seed is not None else seeds[0]
        if seed not in seeds:
            matching_seeds = [candidate for candidate in seeds if str(candidate) == str(selected_seed)]
            seed = matching_seeds[0] if matching_seeds else seeds[0]

        try:
            _losses, _costs, _max_cost, rows, _learning_curve_rows, _configs = process_seed(
                path=base_path / opt_name,
                seed=seed,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                return_incumbent=False,
                flop_range=pareto_acc_flop_ranges,
            )
        except Exception as e:
            print(f"Warning: could not load rows for {opt_name} seed {seed}: {e}")
            continue

        if not rows:
            continue

        rows_arr = np.asarray(rows, dtype=float)
        if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
            continue

        n_params = rows_arr[:, 0]
        tokens = rows_arr[:, 1]
        losses_arr = rows_arr[:, 2]
        actual_trial_flops = rows_arr[:, 3]
        mask = (
            np.isfinite(n_params)
            & np.isfinite(tokens)
            & np.isfinite(losses_arr)
            & np.isfinite(actual_trial_flops)
            & (actual_trial_flops > 0)
        )
        if not np.any(mask):
            continue

        res = ScalingLawGuidedOptimizer.compute_search_pareto_front(
            losses_arr[mask].tolist(),
            actual_trial_flops[mask].tolist(),
        )
        if not res:
            continue

        flop_counts, front_losses, _idx_points = res
        flop_counts = np.asarray(flop_counts, dtype=float)
        front_losses = np.asarray(front_losses, dtype=float)
        finite_front = np.isfinite(flop_counts) & np.isfinite(front_losses) & (flop_counts > 0) & (front_losses > 0)
        flop_counts = flop_counts[finite_front]
        front_losses = front_losses[finite_front]
        if len(flop_counts) == 0:
            continue

        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        all_data.append({
            "opt_name": opt_name,
            "color": color,
            "linestyle": linestyle,
            "flop_counts": flop_counts,
            "front_losses": front_losses,
            "seed": seed,
        })

    scatter_x_series: list[np.ndarray] = []
    for data in all_data:
        flop_counts = data["flop_counts"]
        front_losses = data["front_losses"]

        if pareto_acc_flop_ranges is not None:
            pareto_mask = (flop_counts >= pareto_acc_flop_ranges[0]) & (flop_counts <= pareto_acc_flop_ranges[1])
            flop_counts = flop_counts[pareto_mask]
            front_losses = front_losses[pareto_mask]

        if len(flop_counts) > 0:
            scatter_x_series.append(flop_counts)

            slope_txt = ""
            fit_result = None
            fit_mask = np.isfinite(flop_counts) & np.isfinite(front_losses) & (flop_counts > 0) & (front_losses > 0)
            if np.sum(fit_mask) >= 2:
                fit_result = fit_power_law(flop_counts[fit_mask], front_losses[fit_mask])
                if fit_result is not None:
                    # slope_txt = f" (m={fit_result.slope:.2f})"
                    fit_grid = np.logspace(
                        np.log10(float(np.min(flop_counts[fit_mask]))),
                        np.log10(float(np.max(flop_counts[fit_mask]))),
                        100,
                    )
                    ax.plot(
                        fit_grid,
                        np.power(10.0, fit_result.intercept) * np.power(fit_grid, fit_result.slope),
                        color=data["color"],
                        linestyle=":" if "_with_priors" in data["opt_name"] else data["linestyle"],
                        linewidth=2.0,
                        alpha=0.9,
                        label="_nolegend_",
                    )

            ax.scatter(
                flop_counts,
                front_losses,
                label=f"{get_optimizer_label(data['opt_name'])}{slope_txt}",
                color=data["color"],
                s=24,
                alpha=0.95,
            )

            if pareto_target_flop is not None:
                if fit_result is not None:
                    predicted_loss, _pred_upper, _pred_lower = predict_from_loglog_fit(
                        fit=fit_result,
                        target_flop=pareto_target_flop,
                    )
                    if np.isfinite(predicted_loss) and predicted_loss > 0:
                        ax.scatter(
                            [pareto_target_flop],
                            [float(predicted_loss)],
                            marker="*",
                            s=140,
                            color=data["color"],
                            edgecolors="black",
                            linewidths=0.6,
                            alpha=0.95,
                            label=f"{get_optimizer_label(data['opt_name'])} Pred loss {float(predicted_loss):.2e}",
                        )

            # band_lower = np.maximum(mean_front_losses - std_front_losses, 1e-12)
            # band_upper = np.maximum(mean_front_losses + std_front_losses, 1e-12)

            # ax.fill_between(
            #     flop_counts,
            #     band_lower,
            #     band_upper,
            #     alpha=0.2,
            #     color=data["color"],
            # )

    ax.set_xlabel("FLOPs", fontsize=20)

    benchmark_xlim: tuple[float, float] | None = None
    finite_benchmark_x = [xs[np.isfinite(xs)] for xs in scatter_x_series if len(xs) > 0]
    finite_benchmark_x = [xs for xs in finite_benchmark_x if len(xs) > 0]
    if finite_benchmark_x:
        left = min(float(np.min(xs)) for xs in finite_benchmark_x)
        right = max(float(np.max(xs)) for xs in finite_benchmark_x)
        if left < right:
            benchmark_xlim = (left, right)

    # Overlay true Pareto-front points in the same FLOP units as optimizer curves.
    plotted_true_reference = False
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)

        # Keep unit consistency with process_seed/process_optimizer_per_seed.


        valid_true = np.isfinite(true_flops) & np.isfinite(true_losses)
        true_flops = true_flops[valid_true]
        true_losses = true_losses[valid_true]

        if len(true_flops) > 0:
            true_x = true_flops

            # Clip true Pareto to benchmark optimizer FLOP range, extended to
            # show empirical points out to 3e16 FLOPs even beyond the optimizers' own range.
            if benchmark_xlim is not None:
                bench_left, bench_right = benchmark_xlim
                bench_mask = (true_x >= bench_left) & (true_x <= bench_right)
                true_x = true_x[bench_mask]
                true_losses = true_losses[bench_mask]

            if len(true_x) > 0:
                order = np.argsort(true_x)
                true_x = true_x[order]
                true_losses = true_losses[order]

                true_fit_result = fit_power_law(
                    np.asarray(true_x, dtype=float),
                    np.asarray(true_losses, dtype=float),
                )
                true_slope_txt = ""
                if true_fit_result is not None:
                    # true_slope_txt = f" (m={true_fit_result.slope:.2f})"
                    true_fit_x = np.asarray(true_x, dtype=float)
                    true_fit_x = true_fit_x[np.isfinite(true_fit_x) & (true_fit_x > 0)]
                    if len(true_fit_x) >= 2:
                        true_fit_grid = np.logspace(
                            np.log10(float(np.min(true_fit_x))),
                            np.log10(float(np.max(true_fit_x))),
                            100,
                        )
                        ax.plot(
                            true_fit_grid,
                            np.power(10.0, true_fit_result.intercept) * np.power(true_fit_grid, true_fit_result.slope),
                            color="black",
                            linestyle=":",
                            linewidth=2.0,
                            alpha=0.9,
                            label="_nolegend_",
                        )

                ax.scatter(
                    true_x,
                    true_losses,
                    color="black",
                    s=24,
                    alpha=0.95,
                    label=f"Empirical Pareto Front{true_slope_txt}",
                )
    if benchmark_xlim is not None:
        if flop_range is not None:
            ax.set_xlim(max(benchmark_xlim[0], flop_range[0]), min(benchmark_xlim[1], flop_range[1]))
        else:
            ax.set_xlim(benchmark_xlim[0], benchmark_xlim[1])
    cur_xlim = ax.get_xlim()
    ax.set_xlim(cur_xlim[0], max(cur_xlim[1], 5e16))
    ax.set_ylabel("Loss", fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def calculate_cwel(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    true_pareto_info: dict[str, list],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    flop_range: tuple[float, float] | None = None,
) -> dict[str, dict]:
    """
    Calculate Cumulative Weighted Error Loss (CWEL) over flops.
    
    CWEL = sum(flop[i] * abs(loss[i] - true_pareto_loss[i])) / cumulative_flops
    
    This metric measures how far the current trajectory deviates from the true pareto front,
    weighted by the FLOPs spent at that point.
    """
    all_flop_grids: list[np.ndarray] = []
    all_per_seed_data: dict[str, dict[str, Any]] = {}
    
    # Extract true pareto data for interpolation
    true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
    true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    true_valid = np.isfinite(true_flops) & np.isfinite(true_losses)
    
    if not np.any(true_valid):
        return {}
    
    true_flops = true_flops[true_valid]
    true_losses = true_losses[true_valid]
    # Sort by flops for interpolation
    sort_idx = np.argsort(true_flops)
    true_flops = true_flops[sort_idx]
    true_losses = true_losses[sort_idx]

    for opt_name, seeds in optimizers.items():
        raw_flop_grid, per_seed_losses, per_seed_costs = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            interpolate=False,
            flop_range=flop_range,
            return_incumbent=False,
        )
        max_cost = max((float(np.max(costs)) for costs in per_seed_costs.values() if len(costs) > 0), default=0.0)
        positive_min = min(
            (float(np.min(costs[costs > 1e-8])) for costs in per_seed_costs.values() if np.any(costs > 1e-8)),
            default=0.0,
        )
        if log_scale and positive_min > 0 and max_cost > positive_min:
            flop_grid = np.logspace(np.log10(positive_min), np.log10(max_cost), num=100)
        else:
            flop_grid = np.linspace(0.0, max_cost, num=100)

        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_losses": per_seed_losses,
            "per_seed_costs": per_seed_costs,
        }
        for seed_costs in per_seed_costs.values():
            if len(seed_costs) > 0:
                all_flop_grids.append(np.asarray(seed_costs, dtype=float))
        if not per_seed_costs and len(raw_flop_grid) > 0:
            all_flop_grids.append(raw_flop_grid)
        elif not per_seed_costs and len(flop_grid) > 0:
            all_flop_grids.append(flop_grid)

    bounds = _get_global_flop_bounds(all_flop_grids)
    cwel_dict: dict[str, dict[str, np.ndarray]] = {}

    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        per_seed_costs = data["per_seed_costs"]

        seed_cwel_list = []
        for seed, losses in per_seed_losses.items():
            seed_costs = np.asarray(per_seed_costs[seed], dtype=float)
            losses_arr = np.asarray(losses, dtype=float)
            
            # Skip if no valid data
            if len(seed_costs) == 0 or len(losses_arr) == 0:
                continue
            
            # Interpolate true pareto loss at each seed flop point
            positive_mask = np.isfinite(seed_costs) & (seed_costs > 1e-8) & np.isfinite(losses_arr)
            true_positive_mask = true_flops > 1e-8
            if np.sum(positive_mask) < 2 or np.sum(true_positive_mask) < 2:
                continue

            metric_costs = seed_costs[positive_mask]
            seed_metric_flops = np.log10(np.maximum(metric_costs, 1.0))
            losses_arr = losses_arr[positive_mask]
            true_metric_flops = np.log10(np.maximum(true_flops[true_positive_mask], 1.0))
            true_metric_losses = true_losses[true_positive_mask]
            true_pareto_losses_at_seed_flops = np.interp(
                seed_metric_flops,
                true_metric_flops,
                true_metric_losses,
                left=np.nan,
                right=np.nan,
            )
            weighted_errors = metric_costs * np.abs(losses_arr - true_pareto_losses_at_seed_flops)
            cumsum_weighted_errors = np.cumsum(weighted_errors)
            cumsum_flops = np.cumsum(metric_costs)
            cwel_values = cumsum_weighted_errors / np.maximum(cumsum_flops, 1e-10)
            seed_costs = metric_costs
            
            # Interpolate to flop_grid
            if len(seed_costs) > 0 and np.any(np.isfinite(cwel_values)):
                valid_mask = np.isfinite(cwel_values)
                if np.any(valid_mask):
                    valid_costs = seed_costs[valid_mask]
                    valid_cwel = cwel_values[valid_mask]
                    seed_cwel = np.interp(
                        flop_grid, valid_costs, valid_cwel,
                        left=np.nan, right=valid_cwel[-1]
                    )
                    seed_cwel_list.append(seed_cwel)

        if not seed_cwel_list:
            continue

        seed_cwel_array = np.asarray(seed_cwel_list, dtype=float)
        mean_cwels = np.nanmean(seed_cwel_array, axis=0)
        std_cwels = np.nanstd(seed_cwel_array, axis=0)
        valid_counts = np.maximum(np.sum(np.isfinite(seed_cwel_array), axis=0), 1)
        se_cwels = std_cwels / np.sqrt(valid_counts)

        cwel_dict[opt_name] = {
            "flop_counts": flop_grid,
            "norm_flops": _normalize_flop_counts(flop_grid, bounds[0], bounds[1]) if bounds is not None else flop_grid,
            "mean_cwels": mean_cwels.astype(float),
            "std_cwels": se_cwels.astype(float),
        }

    return cwel_dict


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs" if kw.get("normalize_flops", True) else "Cumulative FLOPs",
        ylabel="Hypervolume",
    )
)
def plot_hypervolume_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Hypervolume Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
) -> Any:
    fig, ax = plt.subplots(figsize=figsize)
    max_hypervolume = 1.1 * 1.1

    normalized_range = (
        flop_range is not None
        and 0.0 <= flop_range[0] <= 1.0
        and 0.0 <= flop_range[1] <= 1.0
    )
    effective_flop_range = flop_range if (not normalize_flops or normalized_range) else None
    
    hvs_data = calculate_hypervolumes(
        base_path=base_path,
        optimizers=optimizers,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
        flop_range=flop_range,
    )
    
    # First pass: collect all data and find maximum of minimum flops (excluding 0.0)
    all_data = []
    all_first_flops = []
    
    for opt_name, data in hvs_data.items():
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        flop_counts = data["norm_flops"] if normalize_flops else data["flop_counts"]
        mean_hvs = data["mean_hvs"]
        std_hvs = data["std_hvs"]
        
        # Trim data to flop_range if specified
        if effective_flop_range is not None:
            min_flops, max_flops = effective_flop_range
            mask = (flop_counts >= min_flops) & (flop_counts <= max_flops)
            flop_counts = flop_counts[mask]
            mean_hvs = mean_hvs[mask]
            std_hvs = std_hvs[mask]
        
        if len(flop_counts) > 0:
            all_data.append({
                'opt_name': opt_name,
                'color': color,
                'linestyle': linestyle,
                'flop_counts': flop_counts,
                'mean_hvs': mean_hvs,
                'std_hvs': std_hvs,
            })
            # Record first flop value (excluding 0.0)
            non_zero_flops = flop_counts[flop_counts > 1e-8]
            if len(non_zero_flops) > 0:
                all_first_flops.append(non_zero_flops[0])
    
    # Calculate common starting point (max of all first flops, excluding 0.0)
    common_start = max(all_first_flops) if all_first_flops else 0
    
    # Second pass: plot with common starting point
    x_series_for_limits: list[np.ndarray] = []
    for data in all_data:
        flop_counts = data['flop_counts']
        mean_hvs = data['mean_hvs']
        std_hvs = data['std_hvs']

        # Trim data to common_start
        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        mean_hvs = mean_hvs[mask]
        std_hvs = std_hvs[mask]
        
        # Ensure all lines start from common_start by prepending first value if needed
        if len(flop_counts) > 0 and flop_counts[0] > common_start + 1e-8:
            flop_counts = np.concatenate([[common_start], flop_counts])
            mean_hvs = np.concatenate([[mean_hvs[0]], mean_hvs])
            std_hvs = np.concatenate([[std_hvs[0]], std_hvs])
        
        if len(flop_counts) > 0:
            x_series_for_limits.append(flop_counts)
            # Plot mean hypervolume
            ax.plot(
                flop_counts,
                mean_hvs,
                label=get_optimizer_label(data['opt_name']),
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2.5,
            )
            
            # Add std band
            lower_hvs = np.clip(mean_hvs - std_hvs, 0.0, max_hypervolume)
            upper_hvs = np.clip(mean_hvs + std_hvs, 0.0, max_hypervolume)
            ax.fill_between(
                flop_counts,
                lower_hvs,
                upper_hvs,
                alpha=0.2,
                color=data['color'],
            )
    
    ax.set_xlabel("Normalized FLOPs" if normalize_flops else "Cumulative FLOPs", fontsize=20)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)
    ax.set_ylabel("Hypervolume", fontsize=20)
    # ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if not normalize_flops:
        ax.set_xscale("log")
    
    # Automatically scale y-axis to fit data tightly
    ax.autoscale(enable=True, axis='y', tight=True)
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs" if kw.get("normalize_flops", True) else "Cumulative FLOPs",
        ylabel="PER",
    )
)
def plot_pareto_line_area_to_true_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    true_pareto_info: dict[str, list],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Pareto Line Area to True Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
) -> Any:
    """Plot area between fitted current Pareto line and fitted true Pareto line."""
    fig, ax = plt.subplots(figsize=figsize)

    normalized_range = (
        flop_range is not None
        and 0.0 <= flop_range[0] <= 1.0
        and 0.0 <= flop_range[1] <= 1.0
    )
    effective_flop_range = flop_range if (not normalize_flops or normalized_range) else None
    area_data = calculate_pareto_line_areas_to_true(
        base_path=base_path,
        optimizers=optimizers,
        true_pareto_info=true_pareto_info,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
        flop_range=flop_range,
    )

    all_data = []
    all_first_flops = []
    for opt_name, data in area_data.items():
        flop_counts = data["norm_flops"] if normalize_flops else data["flop_counts"]
        mean_areas = data["mean_areas"]
        std_areas = data["std_areas"]

        if effective_flop_range is not None:
            min_flops, max_flops = effective_flop_range
            mask = (flop_counts >= min_flops) & (flop_counts <= max_flops)
            flop_counts = flop_counts[mask]
            mean_areas = mean_areas[mask]
            std_areas = std_areas[mask]

        if len(flop_counts) == 0:
            continue
        all_data.append({
            "opt_name": opt_name,
            "flop_counts": flop_counts,
            "mean_areas": mean_areas,
            "std_areas": std_areas,
        })
        non_zero_flops = flop_counts[flop_counts > 1e-8]
        if len(non_zero_flops) > 0:
            all_first_flops.append(non_zero_flops[0])

    common_start = max(all_first_flops) if all_first_flops else 0.0
    x_series_for_limits: list[np.ndarray] = []
    y_values_for_view: list[np.ndarray] = []
    for data in all_data:
        flop_counts = data["flop_counts"]
        mean_areas = data["mean_areas"]
        std_areas = data["std_areas"]

        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        mean_areas = mean_areas[mask]
        std_areas = std_areas[mask]
        if len(flop_counts) == 0:
            continue

        x_series_for_limits.append(flop_counts)
        finite_mean_areas = mean_areas[np.isfinite(mean_areas)]
        if len(finite_mean_areas) > 0:
            y_values_for_view.append(finite_mean_areas)
        color = get_optimizer_color(data["opt_name"])
        linestyle = get_optimizer_linestyle(data["opt_name"])
        ax.plot(
            flop_counts,
            mean_areas,
            label=get_optimizer_label(data["opt_name"]),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        lower = np.maximum(mean_areas - std_areas, 0.0)
        upper = mean_areas + std_areas
        ax.fill_between(flop_counts, lower, upper, alpha=0.2, color=color)

    ax.set_xlabel("Normalized FLOPs" if normalize_flops else "Cumulative FLOPs", fontsize=20)
    ax.set_ylabel("PER (Quantile 99%)", fontsize=20)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)
    if not normalize_flops:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.autoscale(enable=True, axis='y', tight=True)
    if y_values_for_view:
        finite_y_values = np.concatenate(y_values_for_view)
        finite_y_values = finite_y_values[np.isfinite(finite_y_values) & (finite_y_values >= 0.0)]
        if len(finite_y_values) > 0:
            y_upper = float(np.nanquantile(finite_y_values, 0.99))
            if y_upper > 0.0:
                ax.set_ylim(0.0, y_upper * 1.05)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig, ax


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs" if kw.get("normalize_flops", True) else "Cumulative FLOPs",
        ylabel="CWEL",
    )
)
def plot_cwel(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    true_pareto_info: dict[str, list],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Cumulative Weighted Error Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
) -> Any:
    """Plot Cumulative Weighted Error Loss (CWEL) over flops."""
    fig, ax = plt.subplots(figsize=figsize)

    normalized_range = (
        flop_range is not None
        and 0.0 <= flop_range[0] <= 1.0
        and 0.0 <= flop_range[1] <= 1.0
    )
    effective_flop_range = flop_range if (not normalize_flops or normalized_range) else None
    cwel_data = calculate_cwel(
        base_path=base_path,
        optimizers=optimizers,
        true_pareto_info=true_pareto_info,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
        flop_range=flop_range,
    )

    all_data = []
    all_first_flops = []
    for opt_name, data in cwel_data.items():
        flop_counts = data["norm_flops"] if normalize_flops else data["flop_counts"]
        mean_cwels = data["mean_cwels"]
        std_cwels = data["std_cwels"]

        if effective_flop_range is not None:
            min_flops, max_flops = effective_flop_range
            mask = (flop_counts >= min_flops) & (flop_counts <= max_flops)
            flop_counts = flop_counts[mask]
            mean_cwels = mean_cwels[mask]
            std_cwels = std_cwels[mask]

        if len(flop_counts) == 0:
            continue
        all_data.append({
            "opt_name": opt_name,
            "flop_counts": flop_counts,
            "mean_cwels": mean_cwels,
            "std_cwels": std_cwels,
        })
        non_zero_flops = flop_counts[flop_counts > 1e-8]
        if len(non_zero_flops) > 0:
            all_first_flops.append(non_zero_flops[0])

    common_start = max(all_first_flops) if all_first_flops else 0.0
    x_series_for_limits: list[np.ndarray] = []
    for data in all_data:
        flop_counts = data["flop_counts"]
        mean_cwels = data["mean_cwels"]
        std_cwels = data["std_cwels"]

        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        mean_cwels = mean_cwels[mask]
        std_cwels = std_cwels[mask]
        if len(flop_counts) == 0:
            continue

        x_series_for_limits.append(flop_counts)
        color = get_optimizer_color(data["opt_name"])
        linestyle = get_optimizer_linestyle(data["opt_name"])
        ax.plot(
            flop_counts,
            mean_cwels,
            label=get_optimizer_label(data["opt_name"]),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        lower = np.maximum(mean_cwels - std_cwels, 0.0)
        upper = mean_cwels + std_cwels
        ax.fill_between(flop_counts, lower, upper, alpha=0.2, color=color)

    ax.set_xlabel("Normalized FLOPs" if normalize_flops else "Cumulative FLOPs", fontsize=20)
    ax.set_ylabel("CWEL", fontsize=20)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)
    if not normalize_flops:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.autoscale(enable=True, axis='y', tight=True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig, ax


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs",
        ylabel="Relative Rank",
    )
)
def plot_normalized_relative_ranking_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Optimizer Rankings Over Optimization (Normalized)",
    flop_range: tuple[float, float] | None = None,
    ranking_metric: str = "incumbent",
) -> Any:
    """Plot optimizer rankings with a normalized x-axis for a single benchmark."""
    fig, ax = plt.subplots(figsize=figsize)

    rankings = calculate_normalized_rankings(
        base_path=base_path,
        optimizers=optimizers,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
        ranking_metric=ranking_metric,
    )

    x_series_for_limits: list[np.ndarray] = []
    
    for opt_name, rank_data in rankings.items():
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        flop_counts = rank_data["flop_counts_scaled"]
        mean_ranks = rank_data["mean_ranks"]
        std_ranks = rank_data["std_ranks"]
        
        if len(flop_counts) > 0:
            x_series_for_limits.append(flop_counts)
            ax.plot(
                flop_counts,
                mean_ranks,
                label=get_optimizer_label(opt_name),
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )
            
            ax.fill_between(
                flop_counts,
                mean_ranks - std_ranks,
                mean_ranks + std_ranks,
                alpha=0.2,
                color=color,
            )
    
    ax.set_xlabel("Normalized FLOPs", fontsize=20)
    ax.set_ylabel(f"Relative Rank", fontsize=20)
    # ax.set_title(title, fontsize=14)
    left, right = _get_common_xlim(x_series_for_limits)
    ax.set_xlim(left, right)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()
    
    num_optimizers = len(optimizers)
    ax.set_ylim(num_optimizers + 0.5, 0.5)
    ax.set_yticks(range(1, num_optimizers + 1))
    ax.set_yticklabels(range(1, num_optimizers + 1))
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized FLOPs",
        ylabel="Relative Rank",
    )
)
def plot_optimizer_rankings_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Optimizer Rankings Across Benchmarks",
    ranking_metric: str = "incumbent",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
    true_pareto_info_per_benchmark: dict[str, dict] | None = None,
) -> Any:
    """Plot mean optimizer rankings aggregated across all benchmarks with std bands."""
    fig, ax = plt.subplots(figsize=figsize)

    optimizer_names = list(optimizers.keys())
    
    # Collect all ranking curves for each optimizer across benchmarks
    optimizer_all_ranks = {opt: [] for opt in optimizer_names}
    optimizer_all_flop_counts = {opt: [] for opt in optimizer_names}
    
    for benchmark_name in benchmarks:
        try:
            # Calculate rankings for this benchmark
            rankings = calculate_normalized_rankings(
                base_path=base_path / benchmark_name,
                optimizers=optimizers,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                ranking_metric=ranking_metric,
                flop_range=flop_ranges_per_benchmark.get(benchmark_name) if flop_ranges_per_benchmark else None,
                true_pareto_info=(
                    true_pareto_info_per_benchmark.get(benchmark_name)
                    if true_pareto_info_per_benchmark is not None
                    else None
                ),
            )
            
            # Collect ranking curves for each optimizer from this benchmark
            for opt_name in optimizer_names:
                if opt_name in rankings:
                    seed_ranks = rankings[opt_name]["seed_ranks"]
                    flop_counts = rankings[opt_name]["flop_counts_scaled"]
                    optimizer_all_ranks[opt_name].extend(seed_ranks)
                    optimizer_all_flop_counts[opt_name].append(flop_counts)
        
        except Exception as e:
            print(f"Warning: Could not calculate rankings for {benchmark_name}: {e}")
            continue
    
    # For each optimizer, aggregate ranks across benchmarks
    x_series_for_limits: list[np.ndarray] = []
    for opt_name in optimizer_names:
        if not optimizer_all_ranks[opt_name]:
            print(f"Warning: No ranking data for optimizer {opt_name}")
            continue
        
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        
        # Stack all ranking curves from different benchmarks
        all_ranks_array = np.array(optimizer_all_ranks[opt_name])
        
        # Calculate mean and std across benchmarks
        mean_ranks = np.mean(all_ranks_array, axis=0)
        std_ranks = np.std(all_ranks_array, axis=0)
        se_ranks = std_ranks / np.sqrt(len(optimizer_all_ranks[opt_name]))
        
        # Use normalized FLOP coordinates from the first benchmark as reference.
        # Ranking curves are interpolated onto the same [0, 1] grid.
        flop_counts = optimizer_all_flop_counts[opt_name][0]
        x_series_for_limits.append(flop_counts)
        
        # Plot mean ranking
        ax.plot(
            flop_counts,
            mean_ranks,
            label=get_optimizer_label(opt_name),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        
        # Add confidence band (standard error)
        ax.fill_between(
            flop_counts,
            mean_ranks - se_ranks,
            mean_ranks + se_ranks,
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel("Normalized FLOPs", fontsize=20)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylabel(f"Relative Rank", fontsize=20)
    # ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()  # Invert so rank 1 is at top
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


@split_by_priors(_composite_empty_placeholder)
def plot_cwel_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    true_pareto_info_per_benchmark: dict[str, dict],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
) -> Any:
    """Create composite CWEL figures with all benchmarks + across-benchmark aggregate ranking."""
    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    cols = num_subplots
    figsize = (6 * cols, 4)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Collect handles/labels for shared legend
    shared_handles = None
    shared_labels = None
    # Plot CWEL for each benchmark
    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            
            # Call plot_cwel (with return_split=False to get single figure)
            letter = string.ascii_lowercase[idx + 1]
            fig_bench, ax_bench = cast(tuple[Any, Any], plot_cwel(
                base_path=benchmark_path,
                optimizers=optimizers,
                true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=False,
                figsize=(5.5, 3.5),
                title="",
                flop_range=flop_ranges_per_benchmark[benchmark_name] if flop_ranges_per_benchmark else None,
                normalize_flops=False,
                return_split=False,
            ))
            ax.set_title(f"({letter}) {benchmark_name}", loc='left', fontsize=20)
            
            # Re-plot line data instead of reusing artists from another figure.
            _replot_lines(ax_bench, ax)
            
            # Copy axis properties
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlim(ax_bench.get_xlim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Collect legend handles/labels from first benchmark
            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()
            
            plt.close(fig_bench)
            
        except Exception as e:
            print(f"Warning: Could not plot CWEL for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}", 
                   ha="center", va="center", transform=ax.transAxes)
    
    # Plot across-benchmark aggregate ranking (by CWEL, lower is better) in first subplot
    agg_ax = axes[0]
    try:
        # Call ranking plot function with "cwel" metric for aggregate
        fig_agg, ax_agg = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=optimizers,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=False,
            title="",
            figsize=(5.5, 3.5),
            return_split=False,
            ranking_metric="cwel",
            flop_ranges_per_benchmark=flop_ranges_per_benchmark,
            true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
        ))
        title="Across Benchmarks (CWEL)"
        
        # Re-plot line data instead of reusing artists from another figure.
        _replot_lines(ax_agg, agg_ax)
        
        # Copy axis properties
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.set_title(f"({string.ascii_lowercase[0]}) {title}", loc='left', fontsize=20)
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.close(fig_agg)
        
    except Exception as e:
        print(f"Warning: Could not plot aggregate CWEL ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", 
                   ha="center", va="center", transform=agg_ax.transAxes)

    agg_ax.set_xlim(0.0, 1.0)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add single shared legend at bottom
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                  ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')
    return fig, axes


@split_by_priors(_composite_empty_placeholder)
def plot_loss_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    flop_ranges_per_benchmark: dict[str, tuple[float, float]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
) -> Any:
    """Create composite loss figures with all benchmarks + across-benchmark aggregate."""
    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    cols = num_subplots
    figsize = (6 * cols, 4)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Collect handles/labels for shared legend
    shared_handles = None
    shared_labels = None
    # Plot loss for each benchmark (aggregate panel is first, so benchmarks start at index 1)
    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            
            # Call existing loss plot function (with return_split=False to get single figure)
            letter = string.ascii_lowercase[idx + 1]
            fig_bench, ax_bench = cast(tuple[Any, Any], plot_optimizer_loss(
                base_path=benchmark_path,
                optimizers=optimizers,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=False,
                figsize=(5.5, 3.5),
                title="",
                flop_range=flop_ranges_per_benchmark[benchmark_name],
                normalize_flops=False,
                return_split=False,
            ))
            ax.set_title(f"({letter}) {BENCHMARK_TITLES[benchmark_name]}", loc='left', fontsize=20)
            
            # Re-plot line data instead of reusing artists from another figure.
            _replot_lines(ax_bench, ax)
            
            # Copy axis properties
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlim(ax_bench.get_xlim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            # ax.set_title(benchmark_name, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Collect legend handles/labels from first benchmark
            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()
            
            plt.close(fig_bench)
            
        except Exception as e:
            print(f"Warning: Could not plot loss for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}", 
                   ha="center", va="center", transform=ax.transAxes)
    
    # Plot across-benchmark aggregate (rankings) in the first subplot
    agg_ax = axes[0]
    try:
        # Call ranking plot function for aggregate incumbent ranking
        fig_agg, ax_agg = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=optimizers,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            title="",
            figsize=(5.5, 3.5),
            return_split=False,
            ranking_metric="incumbent",
            flop_ranges_per_benchmark=flop_ranges_per_benchmark,
        ))
        title="Across Benchmarks"
        
        # Re-plot line data instead of reusing artists from another figure.
        _replot_lines(ax_agg, agg_ax)
        
        # Copy axis properties
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.set_title(f"({string.ascii_lowercase[0]}) {title}", loc='left', fontsize=20)
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        # agg_ax.invert_yaxis()
        plt.close(fig_agg)
        
    except Exception as e:
        print(f"Warning: Could not plot aggregate ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", 
                   ha="center", va="center", transform=agg_ax.transAxes)

    axes[0].set_xlim(0.0, 1.0)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add single shared legend at bottom
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                  ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')
    return fig, axes


@split_by_priors(_composite_empty_placeholder)
def plot_hypervolume_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]],
) -> Any:
    """Create composite hypervolume figures with all benchmarks + across-benchmark aggregate."""
    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    cols = num_subplots
    figsize = (6 * cols, 4)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize, sharex=False)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    
    # Collect handles/labels for shared legend
    shared_handles = None
    shared_labels = None
    
    # Plot hypervolume for each benchmark (aggregate panel is first, so benchmarks start at index 1)
    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            
            # Call existing hypervolume plot function (with return_split=False to get single figure)
            fig_bench, ax_bench = cast(tuple[Any, Any], plot_hypervolume_over_flops(
                base_path=benchmark_path,
                optimizers=optimizers,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=False,
                title="",
                figsize=(5.5, 3.5),
                flop_range=flop_ranges_per_benchmark[benchmark_name],
                normalize_flops=False,
                return_split=False,
            ))
            letter = string.ascii_lowercase[idx + 1]
            
            # Re-plot line data instead of reusing artists from another figure.
            _replot_lines(ax_bench, ax)
            
            # Copy axis properties
            ax.set_title(f"({letter}) {BENCHMARK_TITLES[benchmark_name]}", loc='left', fontsize=20)
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            benchmark_x_values: list[np.ndarray] = []
            for line in ax_bench.get_lines():
                if line.get_transform() == ax_bench.get_yaxis_transform():
                    continue
                x_values = np.asarray(line.get_xdata(), dtype=float)
                x_values = x_values[np.isfinite(x_values) & (x_values > 0)]
                if len(x_values) > 0:
                    benchmark_x_values.append(x_values)
            if benchmark_x_values:
                all_benchmark_x = np.concatenate(benchmark_x_values)
                ax.set_xlim(float(np.min(all_benchmark_x)), float(np.max(all_benchmark_x)))
            else:
                ax.set_xlim(ax_bench.get_xlim())
            # ax.set_title(benchmark_name, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Collect legend handles/labels from first benchmark
            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()
            
            plt.close(fig_bench)
            
        except Exception as e:
            print(f"Warning: Could not plot hypervolume for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}", 
                   ha="center", va="center", transform=ax.transAxes)
    
    # Plot across-benchmark aggregate (rankings using hypervolume metric) in the first subplot
    agg_ax = axes[0]
    try:
        # Call ranking plot function for aggregate hypervolume ranking
        fig_agg, ax_agg = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=optimizers,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            flop_ranges_per_benchmark=flop_ranges_per_benchmark,
            title="",
            figsize=(5.5, 3.5),
            return_split=False,
            ranking_metric="hypervolume",
        ))
        title="Across Benchmarks"

        
        # Re-plot line data instead of reusing artists from another figure.
        _replot_lines(ax_agg, agg_ax)
        plt.close(fig_agg)
        
        # Copy axis properties
        agg_ax.set_title(f"({string.ascii_lowercase[0]}) {title}", loc='left', fontsize=20)
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlim(ax_agg.get_xlim())
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        # agg_ax.invert_yaxis()
        
    except Exception as e:
        print(f"Warning: Could not plot aggregate ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", 
                   ha="center", va="center", transform=agg_ax.transAxes)

    # Keep the aggregate ranking panel on normalized FLOPs. Benchmark panels
    # intentionally keep their benchmark-local raw FLOP x-scales.
    axes[0].set_xlim(0.0, 1.0)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add single shared legend at bottom
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                  ncol=min(4, len(optimizers)), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # plt.subplots_adjust(hspace=0.3)
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')
    
    return fig, axes


@split_by_priors(_composite_empty_placeholder)
def plot_pareto_line_area_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    true_pareto_info_per_benchmark: dict[str, dict],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]],
) -> Any:
    """Composite Pareto-line area plots with normalized aggregate and raw benchmark panels."""
    num_subplots = len(benchmarks) + 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 4), sharex=False)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    shared_handles = None
    shared_labels = None

    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            true_info = true_pareto_info_per_benchmark[benchmark_name]
            bench_flop_range = flop_ranges_per_benchmark[benchmark_name]

            fig_bench, ax_bench = cast(tuple[Any, Any], plot_pareto_line_area_to_true_over_flops(
                base_path=benchmark_path,
                optimizers=optimizers,
                true_pareto_info=true_info,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=False,
                title="",
                figsize=(5.5, 3.5),
                flop_range=bench_flop_range,
                normalize_flops=False,
                return_split=False,
            ))

            _replot_lines(ax_bench, ax)
            letter = string.ascii_lowercase[idx + 1]
            ax.set_title(f"({letter}) {BENCHMARK_TITLES[benchmark_name]}", loc='left', fontsize=20)
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            ax.set_xlim(bench_flop_range[0], bench_flop_range[1])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()

            plt.close(fig_bench)

        except Exception as e:
            print(f"Warning: Could not plot Pareto line area for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}",
                    ha="center", va="center", transform=ax.transAxes)

    agg_ax = axes[0]
    try:
        fig_agg, ax_agg = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=optimizers,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            flop_ranges_per_benchmark=flop_ranges_per_benchmark,
            true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
            title="",
            figsize=(5.5, 3.5),
            return_split=False,
            ranking_metric="pareto_area",
        ))
        _replot_lines(ax_agg, agg_ax)
        plt.close(fig_agg)

        agg_ax.set_title(f"({string.ascii_lowercase[0]}) Across Benchmarks", loc='left', fontsize=20)
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlim(0.0, 1.0)
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    except Exception as e:
        print(f"Warning: Could not compute aggregate Pareto line area ranking: {e}")
        agg_ax.text(0.5, 0.5, "Error plotting aggregate",
                    ha="center", va="center", transform=agg_ax.transAxes)

    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                   ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')
    return fig, axes



if __name__ == '__main__':
    from src.plotting.consts import results_dir, all_seeds, acc_flop_ranges_per_benchmark,\
        true_pareto_info_per_benchmark, selected_best_seed_per_benchmark, \
        loss_key, flop_range_to_show_in_pareto_per_benchmark, optimizers_dict
    from src.plotting.utils import merge_split_composite_plots, merge_pareto_plots
    from src.benchmarks.configs import BenchmarkEnum
    
    
    
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value,
        # BenchmarkEnum.TABPFN.value,
    ]

    pareto_merged = {}

    for benchmark_name in benchmarks:

        plot_output_dir = results_dir / benchmark_name / "plots"
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Plotting pareto front loss...")
        # (fig_pareto_all, _), (fig_pareto_with, _), (fig_pareto_without, _) = plot_pareto_front_over_flops(
        selected_seed = selected_best_seed_per_benchmark[benchmark_name]
        for seed in all_seeds:
            (fig_pareto_all, ax) = plot_pareto_front_over_flops(
                base_path=results_dir / benchmark_name,
                optimizers={k: [seed] for k in optimizers_dict.keys()},
                key_to_extract="cost",
                log_scale=False,
                loss_key=loss_key,
                title=f"{BENCHMARK_TITLES[benchmark_name]}",
                flop_range=flop_range_to_show_in_pareto_per_benchmark[benchmark_name],
                # pareto_acc_flop_ranges=acc_flop_ranges_per_benchmark[benchmark_name],
                true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                # pareto_target_flop=target_flop_per_benchmark[benchmark_name],
                return_split=False,
                # selected_seed=seed,
            )
            if seed == selected_seed:
                pareto_merged[BENCHMARK_TITLES[benchmark_name]] = (fig_pareto_all, ax)
            output_path = plot_output_dir / f"pareto_front_all_{seed}.pdf"
            fig_pareto_all.savefig(output_path, dpi=300, bbox_inches="tight")
            # fig_pareto_with.savefig(plot_output_dir / f"pareto_front_with_priors_{seed}.pdf", dpi=300, bbox_inches="tight")
            # fig_pareto_without.savefig(plot_output_dir / f"pareto_front_without_priors_{seed}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig_pareto_all)
            # plt.close(fig_pareto_with)
            # plt.close(fig_pareto_without)
            print(f"    ✓ Saved: {output_path}")
            # print(f"    ✓ Saved: {plot_output_dir / 'pareto_front_with_priors.pdf'}")
            # print(f"    ✓ Saved: {plot_output_dir / 'pareto_front_without_priors.pdf'}")

    pareto_master_fig, _ = merge_pareto_plots(pareto_merged)
    pareto_master_fig.savefig(plots_dir / "selected_pareto_composite_all.pdf", dpi=300, bbox_inches="tight")
    plt.close(pareto_master_fig)
    print(f"✓ Saved: {plots_dir / 'selected_pareto_composite_all.pdf'}")

    print(f"\n{'='*60}")
    print("Generating composite loss plots across all benchmarks...")
    print(f"{'='*60}")
    (fig_loss_comp_all, axes_loss_comp_all), (fig_loss_comp_with, axes_loss_comp_with), (fig_loss_comp_without, axes_loss_comp_without) = plot_loss_composite_across_benchmarks(
        base_path=results_dir,
        benchmarks=benchmarks,
        optimizers=optimizers_dict,
        key_to_extract="cost",
        flop_ranges_per_benchmark=acc_flop_ranges_per_benchmark,
    )
    fig_loss_comp_all.savefig(plots_dir / "loss_composite_all.pdf", dpi=300, bbox_inches="tight")
    fig_loss_comp_with.savefig(plots_dir / "loss_composite_with_priors.pdf", dpi=300, bbox_inches="tight")
    fig_loss_comp_without.savefig(plots_dir / "loss_composite_without_priors.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_loss_comp_all)
    plt.close(fig_loss_comp_with)
    print(f"✓ Saved: {plots_dir / 'loss_composite_all.pdf'}")
    print(f"✓ Saved: {plots_dir / 'loss_composite_with_priors.pdf'}")
    print(f"✓ Saved: {plots_dir / 'loss_composite_without_priors.pdf'}")

    # Plot: Composite hypervolume plots (all benchmarks + aggregate in single figures)
    print(f"\n{'='*60}")
    print("Generating composite hypervolume plots across all benchmarks...")
    print(f"{'='*60}")
    try:
        (fig_hv_comp_all, axes_hv_comp_all), (fig_hv_comp_with, axes_hv_comp_with), (fig_hv_comp_without, axes_hv_comp_without) = plot_hypervolume_composite_across_benchmarks(
            base_path=results_dir,
            benchmarks=benchmarks,
            optimizers=optimizers_dict,
            key_to_extract="cost",
            flop_ranges_per_benchmark=acc_flop_ranges_per_benchmark,
        )
        
        # Save composite hypervolume plots to main plots directory
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_hv_comp_all.savefig(plots_dir / "hypervolume_composite_all.pdf", dpi=300, bbox_inches="tight")
        fig_hv_comp_with.savefig(plots_dir / "hypervolume_composite_with_priors.pdf", dpi=300, bbox_inches="tight")
        fig_hv_comp_without.savefig(plots_dir / "hypervolume_composite_without_priors.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig_hv_comp_all)
        plt.close(fig_hv_comp_with)
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_all.pdf'}")
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_with_priors.pdf'}")
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_without_priors.pdf'}")
    except Exception as e:
        print(f"✗ Error plotting composite hypervolume: {e}")
        import traceback
        traceback.print_exc()
        fig_hv_comp_without = None
        axes_hv_comp_without = None
   
   
    # Plot: Composite pareto area plots (all benchmarks + aggregate in single figures)
    print(f"\n{'='*60}")
    print("Generating composite pareto area plots across all benchmarks...")
    print(f"{'='*60}")
    try:
        (fig_pa_comp_all, axes_pa_comp_all), (fig_pa_comp_with, axes_pa_comp_with), (fig_pa_comp_without, axes_pa_comp_without) = plot_pareto_line_area_composite_across_benchmarks(
            base_path=results_dir,
            benchmarks=benchmarks,
            optimizers=optimizers_dict,
            key_to_extract="cost",
            flop_ranges_per_benchmark=acc_flop_ranges_per_benchmark,
            true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
        )
        
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_pa_comp_all.savefig(plots_dir / "pareto_area_composite_all.pdf", dpi=300, bbox_inches="tight")
        fig_pa_comp_with.savefig(plots_dir / "pareto_area_composite_with_priors.pdf", dpi=300, bbox_inches="tight")
        fig_pa_comp_without.savefig(plots_dir / "pareto_area_composite_without_priors.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig_pa_comp_all)
        plt.close(fig_pa_comp_with)
        print(f"✓ Saved: {plots_dir / 'pareto_area_composite_all.pdf'}")
        print(f"✓ Saved: {plots_dir / 'pareto_area_composite_with_priors.pdf'}")
        print(f"✓ Saved: {plots_dir / 'pareto_area_composite_without_priors.pdf'}")
    except Exception as e:
        print(f"✗ Error plotting composite pareto area: {e}")
        import traceback
        traceback.print_exc()
        fig_pa_comp_without = None
        axes_pa_comp_without = None

    if fig_pa_comp_without is not None and fig_hv_comp_without is not None:
        fig_without_metrics_combined, _ = merge_split_composite_plots(
            (
                (fig_pa_comp_without, axes_pa_comp_without),
                (fig_hv_comp_without, axes_hv_comp_without),
                (fig_loss_comp_without, axes_loss_comp_without),
            ),
            close_sources=True,
        )
        fig_without_metrics_combined.savefig(
            plots_dir / "without_priors_metrics_composite.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_without_metrics_combined)
        print(f"✓ Saved: {plots_dir / 'without_priors_metrics_composite.pdf'}")
    else:
        if fig_pa_comp_without is not None:
            plt.close(fig_pa_comp_without)
        if fig_hv_comp_without is not None:
            plt.close(fig_hv_comp_without)
        plt.close(fig_loss_comp_without)
