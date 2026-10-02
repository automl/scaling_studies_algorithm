from collections import defaultdict
from pathlib import Path
from typing import Any, cast
import json
import logging
from dataclasses import asdict

logging.getLogger("fontTools").setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
import string
from dataclasses import asdict

from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit
from src.benchmarks.configs import get_benchmark
from src.plotting.utils import OPTIMIZER_LABELS, get_optimizer_color,\
    get_optimizer_linestyle, get_optimizer_label, _split_optimizers_by_priors, \
    _placeholder_plot, _composite_empty_placeholder, _get_global_flop_bounds, \
    _normalize_flop_counts, _actual_costs_from_cumulative, _get_hypervolume_bounds, \
    _get_common_xlim, _replot_lines, split_by_priors, _extend_curve_to, \
    get_predictor_label, prediction_cache_dir, target_from_loss_key


from src.plotting.standalone_plots import (
    _get_best_loss_guess,
    _get_exponent_bounds,
    _get_log_scale_upper_clip,
    _filter_log_scale_spikes,
    process_seed,
    process_optimizer_per_seed,
    process_optimizer,
    plot_optimizer_loss,
    calculate_pareto_fronts,
    plot_pareto_front_over_flops,
    calculate_rankings,
    calculate_normalized_rankings,
    calculate_hypervolumes,
    calculate_pareto_line_areas_to_true,
    calculate_cwel,
    plot_cwel,
    plot_cwel_composite_across_benchmarks,
    plot_hypervolume_over_flops,
    plot_pareto_line_area_to_true_over_flops,
    plot_normalized_relative_ranking_over_flops,
    plot_optimizer_rankings_across_benchmarks,
    plot_loss_composite_across_benchmarks,
    plot_hypervolume_composite_across_benchmarks,
    plot_pareto_line_area_composite_across_benchmarks,
)

plt.rcParams.update({
    "font.family": "serif",        # Forces Times-style font
    "font.size": 20,               # Base font size
    "axes.labelsize": 20,          # X and Y labels
    "xtick.labelsize": 15,          # X-axis numbers
    "ytick.labelsize": 15,          # Y-axis numbers
    "legend.fontsize": 20,          # Legend labels
    "figure.figsize": (5.5, 3.5),      # Single-panel default used by composite plots
    "pdf.fonttype": 42,            # Ensures fonts render correctly in the final PDF
    "ps.fonttype": 42
})

@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized Cumulative FLOPs" if kw.get("normalized_flops", True) else "Cumulative FLOPs",
        ylabel=r"$|L - L^*|$" if kw.get("abs_loss_diff", False) else r"$L$",
    )
)
def plot_predicted_loss_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    normalized_flops: bool = True,
    y_clip_quantile: float | None = 0.99,
    abs_loss_diff: bool = False,
    cost_estimator: Any = None,
    search_space: Any = None,
) -> Any:
    """Plot predicted loss over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    from src.loss_predictor import LossPredictorTypes, get_loss_fit_trajectory

    if predictor_type is None:
        predictor_type = LossPredictorTypes.KAPLAN

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)
    true_target_loss: float | None = None

    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
        valid_true = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
        if np.any(valid_true):
            closest_idx = int(np.argmin(np.abs(true_flops[valid_true] - target_flop)))
            true_target_loss = float(true_losses[valid_true][closest_idx])

    if abs_loss_diff and true_target_loss is None:
        raise ValueError("abs_loss_diff=True requires true_pareto_info with finite positive flops and losses")

    # First pass: collect all cumulative FLOP values to determine normalization bounds
    all_flop_arrays: list[np.ndarray] = []
    all_data_per_opt: dict[str, list[dict[str, np.ndarray]]] = {}

    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed_curves: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            try:
                _losses, _cum_costs, _max_cost, rows, _learning_curve_rows, _configs = process_seed(
                    path=base_path / opt_name,
                    seed=seed,
                    key_to_extract=key_to_extract,
                    loss_key=loss_key,
                    return_incumbent=False,
                )
            except Exception as e:
                print(f"Warning: could not load rows for {opt_name} seed {seed}: {e}")
                continue

            if not rows:
                continue

            trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
            trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
            cumulative_flops = np.cumsum(trial_flops)

            predictor_slug = str(getattr(predictor_type, "value", None) or getattr(predictor_type, "name", None) or str(predictor_type))

            # Fit is target_flop-independent and predictor-owned: caching lives
            # inside get_loss_fit_trajectory, keyed by (predictor_type, n_trials).
            _cache_dir = prediction_cache_dir(base_path, opt_name, seed, loss_key)
            _trajectory = get_loss_fit_trajectory(
                predictor_type,
                rows=rows,
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
                space=search_space,
                configs=_configs,
                cost_estimator=cost_estimator,
                cache_dir=_cache_dir,
            )
            x = np.asarray(cumulative_flops, dtype=float)
            y = _trajectory.predict(target_flop)
            if _trajectory.fits:
                try:
                    _, _info = _trajectory.fits[-1].predict(target_flop)
                    # Target-scoped: a downstream target's fit info is a different
                    # quantity and must not overwrite the objective's.
                    _target = target_from_loss_key(loss_key)
                    _fdir = base_path / "plots" / predictor_slug / opt_name / f"seed_{seed}"
                    _fpath = (_fdir / _target if _target else _fdir) / "final_prediction_info.json"
                    _fpath.parent.mkdir(parents=True, exist_ok=True)
                    with open(_fpath, "w+") as _f:
                        json.dump(asdict(_info), _f, indent=2)
                    print(f"Saved final prediction info for {opt_name} seed {seed} to {_fpath}")
                except Exception as _e:
                    print(f"Warning: could not save final prediction info for {opt_name} seed {seed}: {_e}")
            if abs_loss_diff:
                y = np.abs(y - float(true_target_loss))
                y = np.where(y == 0.0, np.finfo(float).tiny, y)
            finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(finite_mask):
                continue

            x = x[finite_mask]
            y = y[finite_mask]

            if flop_range is not None:
                range_mask = (x >= flop_range[0]) & (x <= flop_range[1])
                x = x[range_mask]
                y = y[range_mask]

            if len(x) == 0:
                continue

            all_flop_arrays.append(x)
            seed_curves.append({"flops": x, "losses": y})

        if seed_curves:
            all_data_per_opt[opt_name] = seed_curves

    # Determine normalization bounds if normalized_flops is True
    norm_bounds = None
    global_flop_bounds = _get_global_flop_bounds(all_flop_arrays)
    shared_flop_grid = None
    if normalized_flops:
        norm_bounds = global_flop_bounds
        if norm_bounds is not None:
            shared_flop_grid = np.linspace(0.0, 1.0, num=100)
    else:
        if global_flop_bounds is not None:
            grid_min, grid_max = global_flop_bounds
            if flop_range is not None:
                grid_min = max(grid_min, float(flop_range[0]))
                grid_max = min(grid_max, float(flop_range[1]))
            if grid_min > 0 and grid_max > grid_min:
                shared_flop_grid = np.logspace(np.log10(grid_min), np.log10(grid_max), num=100)

    # Optional reference: true Pareto loss at closest FLOP to target.
    if true_pareto_info is not None and not abs_loss_diff:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
        valid_true = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
        true_flops = true_flops[valid_true]
        true_losses = true_losses[valid_true]
        if len(true_flops) > 0:
            closest_idx = int(np.argmin(np.abs(true_flops - target_flop)))
            closest_flop = float(true_flops[closest_idx])
            closest_loss = float(true_losses[closest_idx])
            ax.axhline(
                y=closest_loss,
                color="black",
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                label=f"Empirical Best Loss",
            )

    plot_records: list[dict[str, Any]] = []

    # Second pass: aggregate all seeds per optimizer on a shared grid.
    for opt_name, seed_curves in all_data_per_opt.items():
        interpolated_seed_losses: list[np.ndarray] = []

        for data in seed_curves:
            x = data["flops"].copy()
            y = data["losses"].copy()

            if normalized_flops and norm_bounds is not None:
                x = _normalize_flop_counts(x, norm_bounds[0], norm_bounds[1])

            if shared_flop_grid is not None:
                order = np.argsort(x)
                x_sorted = x[order]
                y_sorted = y[order]
                x_unique, unique_idx = np.unique(x_sorted, return_index=True)
                y_unique = y_sorted[unique_idx]
                if len(x_unique) >= 2:
                    y_interp = np.interp(shared_flop_grid, x_unique, y_unique)
                    outside = (shared_flop_grid < x_unique[0]) | (shared_flop_grid > x_unique[-1])
                    y_interp[outside] = np.nan
                    interpolated_seed_losses.append(y_interp)
                elif len(x_unique) == 1:
                    y_interp = np.full_like(shared_flop_grid, np.nan, dtype=float)
                    nearest_idx = int(np.argmin(np.abs(shared_flop_grid - x_unique[0])))
                    y_interp[nearest_idx] = y_unique[0]
                    interpolated_seed_losses.append(y_interp)
            else:
                interpolated_seed_losses.append(y)

        if not interpolated_seed_losses:
            continue

        min_len = min(len(values) for values in interpolated_seed_losses)
        seed_loss_array = np.asarray([values[:min_len] for values in interpolated_seed_losses], dtype=float)
        if shared_flop_grid is not None:
            x_plot = shared_flop_grid[:min_len]
        else:
            x_plot = seed_curves[0]["flops"][:min_len].copy()
            if normalized_flops and norm_bounds is not None:
                x_plot = _normalize_flop_counts(x_plot, norm_bounds[0], norm_bounds[1])

        valid_counts = np.sum(np.isfinite(seed_loss_array), axis=0)
        has_values = valid_counts > 0
        if not np.any(has_values):
            continue

        mean_losses = np.full(seed_loss_array.shape[1], np.nan, dtype=float)
        se_losses = np.full(seed_loss_array.shape[1], np.nan, dtype=float)
        mean_losses[has_values] = np.nanmean(seed_loss_array[:, has_values], axis=0)
        std_losses = np.nanstd(seed_loss_array[:, has_values], axis=0)
        se_losses[has_values] = std_losses / np.sqrt(valid_counts[has_values])

        x_plot = x_plot[has_values]
        y = mean_losses[has_values]
        yerr = se_losses[has_values]

        lower = np.maximum(y - yerr, np.finfo(float).tiny)
        upper = y + yerr
        plot_records.append({
            "opt_name": opt_name,
            "x": x_plot,
            "y": y,
            "lower": lower,
            "upper": upper,
        })

    # Curves that run out of data early are held constant to the shared end point.
    common_end = max(
        (float(record["x"][-1]) for record in plot_records if len(record["x"]) > 0),
        default=0.0,
    )
    for record in plot_records:
        record["x"], (record["y"], record["lower"], record["upper"]) = _extend_curve_to(
            record["x"],
            [record["y"], record["lower"], record["upper"]],
            common_end,
        )

    y_clip_upper = _get_log_scale_upper_clip(
        [record["y"] for record in plot_records] + [record["upper"] for record in plot_records],
        y_clip_quantile,
    )

    for record in plot_records:
        opt_name = record["opt_name"]
        x_plot = record["x"]
        y = record["y"]
        lower = record["lower"]
        upper = record["upper"]

        if y_clip_upper is not None:
            y = np.minimum(y, y_clip_upper)
            lower = np.minimum(lower, y_clip_upper)
            upper = np.minimum(upper, y_clip_upper)

        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)

        ax.plot(
            x_plot,
            y,
            label=get_optimizer_label(opt_name),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        ax.fill_between(
            x_plot,
            lower,
            upper,
            alpha=0.2,
            color=color,
        )

        x_series_for_limits.append(x_plot)

    ax.set_xlabel("Normalized Cumulative FLOPs" if normalized_flops else "Cumulative FLOPs", fontsize=20)
    ax.set_ylabel(r"$|L - L^*|$" if abs_loss_diff else r"$L$", fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if not normalized_flops:
        ax.set_xscale("log")
    
    ax.set_yscale("log")

    if y_clip_upper is not None:
        ax.set_ylim(top=y_clip_upper)

    if flop_range is not None and not normalized_flops:
        ax.set_xlim(flop_range[0], flop_range[1])
    elif x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax

def plot_loss_predictor_comparison_for_optimizer(
    *,
    base_path: str | Path,
    optimizer_name: str = "random_search",
    seed: int | str | list[int | str] | tuple[int | str, ...] = 40,
    target_flop: float,
    predictor_types: list[Any] | None = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[float, float] = (6.0, 4.0),
    title: str | None = None,
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
) -> tuple[Any, Any]:
    """Plot all loss predictor types for one optimizer, aggregating over seeds."""
    from src.loss_predictor import LossPredictorTypes, get_loss_fit_trajectory

    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    if predictor_types is None:
        predictor_types = list(LossPredictorTypes)

    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]

    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
        valid = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
        if np.any(valid):
            idx = int(np.argmin(np.abs(true_flops[valid] - target_flop)))
            target_val = float(true_losses[valid][idx])
            ax.axhline(y=target_val, color="black", linestyle="--", alpha=0.6, label="Empirical Best Loss")

    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(predictor_types)))
    predictor_curves: dict[Any, list[tuple[np.ndarray, np.ndarray]]] = {p_type: [] for p_type in predictor_types}
    all_positive_flops: list[np.ndarray] = []

    for seed_value in seeds:
        try:
            _losses, _cum_costs, _max_cost, rows, _learning_curve_rows, _configs = process_seed(
                path=base_path / optimizer_name,
                seed=seed_value,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                return_incumbent=False,
            )
        except Exception as e:
            print(f"Error loading {optimizer_name} seed {seed_value}: {e}")
            continue

        if not rows:
            continue

        trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
        trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
        cumulative_flops = np.cumsum(trial_flops)
        positive_cumulative_flops = cumulative_flops[np.isfinite(cumulative_flops) & (cumulative_flops > 0)]
        if len(positive_cumulative_flops) > 0:
            all_positive_flops.append(positive_cumulative_flops)

        for p_type in predictor_types:
            _cache_dir = prediction_cache_dir(base_path, optimizer_name, seed_value, loss_key)
            _trajectory = get_loss_fit_trajectory(
                p_type, rows=rows, best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds, cache_dir=_cache_dir,
            )
            pred_loss_arr = _trajectory.predict(target_flop)

            x = np.asarray(cumulative_flops, dtype=float)
            y = np.asarray(pred_loss_arr, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

            if flop_range is not None:
                mask = mask & (x >= flop_range[0]) & (x <= flop_range[1])

            if np.any(mask):
                predictor_curves[p_type].append((x[mask], y[mask]))

    shared_flop_grid = None
    if all_positive_flops:
        positive_flops = np.concatenate(all_positive_flops)
        grid_min = float(np.min(positive_flops))
        grid_max = float(np.max(positive_flops))
        if flop_range is not None:
            grid_min = max(grid_min, float(flop_range[0]))
            grid_max = min(grid_max, float(flop_range[1]))
        if grid_min > 0 and grid_max > grid_min:
            shared_flop_grid = np.logspace(np.log10(grid_min), np.log10(grid_max), num=100)

    for p_idx, p_type in enumerate(predictor_types):
        curves = predictor_curves[p_type]
        if not curves:
            continue

        label = get_predictor_label(p_type)
        color = getattr(p_type, "color", None)
        if color is None:
            color = palette[p_idx]

        if shared_flop_grid is None:
            x, y = curves[0]
            ax.plot(x, y, label=f"{label}", color=color, linewidth=2.5)
            continue

        seed_values = []
        for x, y in curves:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            if len(x_unique) == 0:
                continue
            if len(x_unique) == 1:
                seed_values.append(np.full_like(shared_flop_grid, y_unique[0], dtype=float))
            else:
                seed_values.append(np.interp(shared_flop_grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1]))

        if not seed_values:
            continue

        seed_array = np.asarray(seed_values, dtype=float)
        mean_values = np.nanmean(seed_array, axis=0)
        std_values = np.nanstd(seed_array, axis=0)
        valid_counts = np.maximum(np.sum(np.isfinite(seed_array), axis=0), 1)
        stderr_values = std_values / np.sqrt(valid_counts)
        ax.plot(shared_flop_grid, mean_values, label=f"{label}", color=color, linewidth=2.5)
        ax.fill_between(
            shared_flop_grid,
            mean_values - stderr_values,
            mean_values + stderr_values,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Cumulative FLOPs", fontsize=20)
    ax.set_ylabel("Predicted Loss", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)

    if flop_range:
        ax.set_xlim(flop_range)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=3, fontsize=20, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    return fig, ax

def plot_n_param_predictor_comparison_for_optimizer(
    *,
    base_path: str | Path,
    optimizer_name: str = "random_search",
    seed: int | str | list[int | str] | tuple[int | str, ...] = 40,
    target_flop: float,
    predictor_types: list[Any],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[float, float] = (6.0, 4.0),
    title: str | None = None,
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    search_space: Any = None,
    cost_estimator: Any = None,
    params_estimator: Any = None,
) -> tuple[Any, Any]:
    """
    Plots multiple scaling law predictors for one optimizer's trajectory.
    Useful for comparing how different predictors (Kaplan vs Chinchilla vs Power Law) 
    converge toward the truth using the same data. Multiple seeds are aggregated
    as mean with standard-error bands.
    """
    from src.scale_param_predictor import get_scale_param_fit_trajectory

    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]

    # 2. Plot True Pareto Reference (if available)
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_n_params = np.asarray(true_pareto_info.get("n_params", []), dtype=float)
        valid = (true_flops > 0) & np.isfinite(true_n_params)
        if np.any(valid):
            idx = np.argmin(np.abs(true_flops[valid] - target_flop))
            target_val = true_n_params[valid][idx]
            ax.axhline(y=target_val, color="black", linestyle="--", alpha=0.6, label="True Optimal Model Size")

    # 3. Iterate through different predictors
    palette = plt.cm.tab10(np.linspace(0, 1, len(predictor_types)))
    predictor_curves: dict[Any, list[tuple[np.ndarray, np.ndarray]]] = {p_type: [] for p_type in predictor_types}
    all_positive_flops: list[np.ndarray] = []
    
    for seed_value in seeds:
        try:
            _losses, _cum_costs, _max_cost, rows, learning_curve_rows, _configs = process_seed(
                path=base_path / optimizer_name,
                seed=seed_value,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                return_incumbent=False,
            )
        except Exception as e:
            print(f"Error loading {optimizer_name} seed {seed_value}: {e}")
            continue

        if not rows:
            continue

        trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
        trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
        cumulative_flops = np.cumsum(trial_flops)
        positive_cumulative_flops = cumulative_flops[np.isfinite(cumulative_flops) & (cumulative_flops > 0)]
        if len(positive_cumulative_flops) > 0:
            all_positive_flops.append(positive_cumulative_flops)

        for p_type in predictor_types:
            from src.scale_param_predictor import ScaleParamPredictorTypes as _SPT
            use_lc = p_type == _SPT.POWER_LAW_FIT and len(learning_curve_rows) > 0
            source_rows = learning_curve_rows if use_lc else rows
            n_trials = len(source_rows)

            _cache_dir = prediction_cache_dir(base_path, optimizer_name, seed_value, loss_key)
            _trajectory = get_scale_param_fit_trajectory(
                p_type,
                rows=source_rows,
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
                space=search_space,
                configs=_configs,
                cost_estimator=cost_estimator,
                params_estimator=params_estimator,
                cache_dir=_cache_dir,
            )
            pred_n_params_arr = _trajectory.predict(target_flop)[:, 0]

            pred_values = list(pred_n_params_arr)

            x = np.asarray(cumulative_flops[:n_trials], dtype=float)
            y = np.asarray(pred_values, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

            if flop_range is not None:
                mask = mask & (x >= flop_range[0]) & (x <= flop_range[1])

            if np.any(mask):
                predictor_curves[p_type].append((x[mask], y[mask]))

    shared_flop_grid = None
    if all_positive_flops:
        positive_flops = np.concatenate(all_positive_flops)
        grid_min = float(np.min(positive_flops))
        grid_max = float(np.max(positive_flops))
        if flop_range is not None:
            grid_min = max(grid_min, float(flop_range[0]))
            grid_max = min(grid_max, float(flop_range[1]))
        if grid_min > 0 and grid_max > grid_min:
            shared_flop_grid = np.logspace(np.log10(grid_min), np.log10(grid_max), num=100)

    for p_idx, p_type in enumerate(predictor_types):
        curves = predictor_curves[p_type]
        if not curves:
            continue

        label = get_predictor_label(p_type)
        color = getattr(p_type, "color", None)
        if color is None:
            color = palette[p_idx]

        if shared_flop_grid is None:
            x, y = curves[0]
            ax.plot(x, y, label=f"{label}", color=color, linewidth=2.5)
            continue

        seed_values = []
        for x, y in curves:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            if len(x_unique) == 0:
                continue
            if len(x_unique) == 1:
                seed_values.append(np.full_like(shared_flop_grid, y_unique[0], dtype=float))
            else:
                seed_values.append(np.interp(shared_flop_grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1]))

        if not seed_values:
            continue

        seed_array = np.asarray(seed_values, dtype=float)
        mean_values = np.nanmean(seed_array, axis=0)
        std_values = np.nanstd(seed_array, axis=0)
        valid_counts = np.maximum(np.sum(np.isfinite(seed_array), axis=0), 1)
        stderr_values = std_values / np.sqrt(valid_counts)
        lower = np.maximum(mean_values - stderr_values, np.finfo(float).tiny)
        upper = mean_values + stderr_values
        ax.plot(shared_flop_grid, mean_values, label=f"{label}", color=color, linewidth=2.5)
        ax.fill_between(
            shared_flop_grid,
            lower,
            upper,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative FLOPs", fontsize=20)
    ax.set_ylabel("Predicted " + r"$N$", fontsize=20)
    if title: ax.set_title(title, fontsize=20)
    
    if flop_range:
        ax.set_xlim(flop_range)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=3, fontsize=20, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    return fig, ax

@split_by_priors(
    lambda kw: _placeholder_plot(
        figsize=kw.get("figsize", (5.5, 3.5)),
        title=kw.get("title", ""),
        xlabel="Normalized Cumulative FLOPs" if kw.get("normalized_flops", True) else "Cumulative FLOPs",
        ylabel=r"$|N - N^*|$" if kw.get("abs_param_diff", False) else r"$N$",
    )
)
def plot_predicted_n_param_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted n_param Over Optimization",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    normalized_flops: bool = True,
    y_clip_quantile: float | None = 0.99,
    abs_param_diff: bool = False,
    cost_estimator: Any = None,
    params_estimator: Any = None,
    search_space: Any = None,
) -> Any:
    """Plot predicted n_param over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    from src.scale_param_predictor import ScaleParamPredictorTypes, get_scale_param_fit_trajectory

    if predictor_type is None:
        predictor_type = ScaleParamPredictorTypes.POWER_LAW_FIT

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)

    all_flop_arrays: list[np.ndarray] = []
    all_data_per_opt: dict[str, list[dict[str, np.ndarray]]] = {}
    true_target_n_param: float | None = None

    # Optional reference: true n_param at closest FLOP to target.
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_n_params = np.asarray(true_pareto_info.get("n_params", []), dtype=float)
        valid_true = np.isfinite(true_flops) & np.isfinite(true_n_params) & (true_flops > 0) & (true_n_params > 0)
        true_flops = true_flops[valid_true]
        true_n_params = true_n_params[valid_true]
        if len(true_flops) > 0:
            closest_idx = int(np.argmin(np.abs(true_flops - target_flop)))
            closest_flop = float(true_flops[closest_idx])
            closest_n_param = float(true_n_params[closest_idx])
            true_target_n_param = closest_n_param
            if not abs_param_diff:
                ax.axhline(
                    y=closest_n_param,
                    color="black",
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.9,
                    label=f"Empirical Best N",
                )

    if abs_param_diff and true_target_n_param is None:
        raise ValueError("abs_param_diff=True requires true_pareto_info with finite positive flops and n_params")

    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed_curves: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            try:
                _losses, _cum_costs, _max_cost, rows, learning_curve_rows, _configs = process_seed(
                    path=base_path / opt_name,
                    seed=seed,
                    key_to_extract=key_to_extract,
                    loss_key=loss_key,
                    return_incumbent=False,
                )
            except Exception as e:
                print(f"Warning: could not load rows for {opt_name} seed {seed}: {e}")
                continue

            if not rows:
                continue

            trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
            trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
            cumulative_flops = np.cumsum(trial_flops)

            use_learning_curves = (
                predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT
                and len(learning_curve_rows) > 0
            )
            source_rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]]
            if use_learning_curves:
                source_rows = learning_curve_rows
                if len(cumulative_flops) != len(learning_curve_rows):
                    print(f"Warning: Expected cumulative_flops and learning_curve_rows to have the same length, but got {len(cumulative_flops)} and {len(learning_curve_rows)}")
                    continue
                total_trials = len(cumulative_flops)
            else:
                source_rows = rows
                total_trials = len(rows)

            if total_trials <= 0:
                continue

            _nparam_predictor_slug = str(getattr(predictor_type, "value", None) or getattr(predictor_type, "name", None) or str(predictor_type))
            _cache_dir = prediction_cache_dir(base_path, opt_name, seed, loss_key)
            _trajectory = get_scale_param_fit_trajectory(
                predictor_type,
                rows=source_rows,
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
                space=search_space,
                configs=_configs,
                cost_estimator=cost_estimator,
                params_estimator=params_estimator,
                cache_dir=_cache_dir,
            )
            pred_n_params_arr = _trajectory.predict(target_flop)[:, 0]
            print(f"[nparam predictor] {opt_name}/seed_{seed}/{_nparam_predictor_slug} {list(pred_n_params_arr)}")

            x = np.asarray(cumulative_flops[:total_trials], dtype=float)
            y = np.asarray(pred_n_params_arr, dtype=float)
            if abs_param_diff:
                y = np.abs(np.log10(y) - float(np.log10(true_target_n_param)))
                y = np.where(y == 0.0, np.finfo(float).tiny, y)
            finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(finite_mask):
                continue

            x = x[finite_mask]
            y = y[finite_mask]

            if flop_range is not None:
                range_mask = (x >= flop_range[0]) & (x <= flop_range[1])
                x = x[range_mask]
                y = y[range_mask]

            x, y = _filter_log_scale_spikes(x, y)

            if len(x) == 0:
                continue

            all_flop_arrays.append(x)
            seed_curves.append({"flops": x, "n_params": y})

        if seed_curves:
            all_data_per_opt[opt_name] = seed_curves

    norm_bounds = None
    global_flop_bounds = _get_global_flop_bounds(all_flop_arrays)
    shared_flop_grid = None
    if normalized_flops:
        norm_bounds = global_flop_bounds
        if norm_bounds is not None:
            shared_flop_grid = np.linspace(0.0, 1.0, num=100)
    else:
        if global_flop_bounds is not None:
            grid_min, grid_max = global_flop_bounds
            if flop_range is not None:
                grid_min = max(grid_min, float(flop_range[0]))
                grid_max = min(grid_max, float(flop_range[1]))
            if grid_min > 0 and grid_max > grid_min:
                shared_flop_grid = np.logspace(np.log10(grid_min), np.log10(grid_max), num=100)

    plot_records: list[dict[str, Any]] = []

    for opt_name, seed_curves in all_data_per_opt.items():
        interpolated_seed_n_params: list[np.ndarray] = []

        for data in seed_curves:
            x = data["flops"].copy()
            y = data["n_params"].copy()

            if normalized_flops and norm_bounds is not None:
                x = _normalize_flop_counts(x, norm_bounds[0], norm_bounds[1])

            if shared_flop_grid is not None:
                order = np.argsort(x)
                x_sorted = x[order]
                y_sorted = y[order]
                x_unique, unique_idx = np.unique(x_sorted, return_index=True)
                y_unique = y_sorted[unique_idx]
                if len(x_unique) >= 2:
                    y_interp = np.interp(shared_flop_grid, x_unique, np.log10(y_unique))
                    outside = (shared_flop_grid < x_unique[0]) | (shared_flop_grid > x_unique[-1])
                    y_interp[outside] = np.nan
                    interpolated_seed_n_params.append(y_interp)
                elif len(x_unique) == 1:
                    y_interp = np.full_like(shared_flop_grid, np.nan, dtype=float)
                    nearest_idx = int(np.argmin(np.abs(shared_flop_grid - x_unique[0])))
                    y_interp[nearest_idx] = np.log10(y_unique[0])
                    interpolated_seed_n_params.append(y_interp)
            else:
                interpolated_seed_n_params.append(np.log10(y))

        if not interpolated_seed_n_params:
            continue

        min_len = min(len(values) for values in interpolated_seed_n_params)
        seed_n_param_array = np.asarray([values[:min_len] for values in interpolated_seed_n_params], dtype=float)
        if shared_flop_grid is not None:
            x_plot = shared_flop_grid[:min_len]
        else:
            x_plot = seed_curves[0]["flops"][:min_len].copy()
            if normalized_flops and norm_bounds is not None:
                x_plot = _normalize_flop_counts(x_plot, norm_bounds[0], norm_bounds[1])

        valid_counts = np.sum(np.isfinite(seed_n_param_array), axis=0)
        has_values = valid_counts > 0
        if not np.any(has_values):
            continue

        mean_log_n_params = np.full(seed_n_param_array.shape[1], np.nan, dtype=float)
        se_log_n_params = np.full(seed_n_param_array.shape[1], np.nan, dtype=float)
        mean_log_n_params[has_values] = np.nanmean(seed_n_param_array[:, has_values], axis=0)
        std_log_n_params = np.nanstd(seed_n_param_array[:, has_values], axis=0)
        se_log_n_params[has_values] = std_log_n_params / np.sqrt(valid_counts[has_values])

        x_plot = x_plot[has_values]
        y = 10 ** mean_log_n_params[has_values]
        log_yerr = se_log_n_params[has_values]

        lower = np.maximum(10 ** (mean_log_n_params[has_values] - log_yerr), np.finfo(float).tiny)
        upper = 10 ** (mean_log_n_params[has_values] + log_yerr)
        plot_records.append({
            "opt_name": opt_name,
            "x": x_plot,
            "y": y,
            "lower": lower,
            "upper": upper,
        })

    # Curves that run out of data early are held constant to the shared end point.
    common_end = max(
        (float(record["x"][-1]) for record in plot_records if len(record["x"]) > 0),
        default=0.0,
    )
    for record in plot_records:
        record["x"], (record["y"], record["lower"], record["upper"]) = _extend_curve_to(
            record["x"],
            [record["y"], record["lower"], record["upper"]],
            common_end,
        )

    y_clip_upper = _get_log_scale_upper_clip(
        [record["y"] for record in plot_records] + [record["upper"] for record in plot_records],
        y_clip_quantile,
    )

    for record in plot_records:
        opt_name = record["opt_name"]
        x_plot = record["x"]
        y = record["y"]
        lower = record["lower"]
        upper = record["upper"]

        if y_clip_upper is not None:
            y = np.minimum(y, y_clip_upper)
            lower = np.minimum(lower, y_clip_upper)
            upper = np.minimum(upper, y_clip_upper)

        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)

        ax.plot(
            x_plot,
            y,
            label=get_optimizer_label(opt_name),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        ax.fill_between(
            x_plot,
            lower,
            upper,
            alpha=0.2,
            color=color,
        )

        x_series_for_limits.append(x_plot)

    ax.set_xlabel("Normalized Cumulative FLOPs" if normalized_flops else "Cumulative FLOPs", fontsize=20)
    ax.set_ylabel(r"$|\log N - \log N^*|$" if abs_param_diff else r"$N$", fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if not normalized_flops:
        ax.set_xscale("log")
    if not abs_param_diff:
        ax.set_yscale("log")

    if y_clip_upper is not None:
        ax.set_ylim(top=y_clip_upper)

    if flop_range is not None and not normalized_flops:
        ax.set_xlim(flop_range[0], flop_range[1])
    elif x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)

    # ax_table.axis("off")
    # if final_prediction_rows_by_opt:
    #     info_columns: list[str] = []
    #     for seed_rows in final_prediction_rows_by_opt.values():
    #         for row in seed_rows:
    #             for col in row.keys():
    #                 if col not in {"spent_over_target", "pred_n_param", "pred_n_data"} and col not in info_columns:
    #                     info_columns.append(col)

    #     table_columns = ["FLOPs / Target", "Method", "pred_n_param", "pred_n_data"] + info_columns
    #     table_rows: list[list[str]] = []
    #     for opt_name, seed_rows in final_prediction_rows_by_opt.items():
    #         row_values: dict[str, float] = {}
    #         for col in ["spent_over_target", "pred_n_param", "pred_n_data"] + info_columns:
    #             values = np.asarray([row.get(col, np.nan) for row in seed_rows], dtype=float)
    #             values = values[np.isfinite(values)]
    #             row_values[col] = float(np.mean(values)) if len(values) > 0 else float("nan")

    #         table_rows.append(
    #             [str(int(round(row_values["spent_over_target"]))), get_optimizer_label(opt_name)]
    #             + [_format_hp_table_value(row_values[col]) for col in ["pred_n_param", "pred_n_data"] + info_columns]
    #         )

        # display_col_labels = {
        #     "pred_n_param": "Pred. N",
        #     "pred_n_data": "Pred. D",
        #     "slope": "Slope",
        # }
        # table = ax_table.table(
        #     cellText=table_rows,
        #     colLabels=[display_col_labels.get(col, col) for col in table_columns],
        #     cellLoc="center",
        #     rowLoc="center",
        #     loc="center",
        # )
        # table.auto_set_font_size(False)
        # table.set_fontsize(8)
        # table.scale(1.0, 1.25)

        # for (r, _c), cell in table.get_celld().items():
        #     cell.visible_edges = "horizontal"
        #     cell.set_edgecolor("black")
        #     if r == 0:
        #         cell.set_linewidth(1.0)
        #         cell.set_facecolor("#F5F5F5")
        #     else:
        #         cell.set_linewidth(0.45)

        # ax_table.set_title("Final Prediction", fontsize=20, pad=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2),
               ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    # plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def _predictor_linestyle(predictor_type: Any) -> str:
    """Solid for the joint-parametric (Chinchilla) predictor, dashed otherwise.

    Optimizers are distinguished by color alone, so the predictor axis has to be
    carried by the linestyle: CHINCHILLA3 is solid for both the loss and n_param
    ("Chinchilla3") families, every other predictor dashed.
    """
    return "-" if str(getattr(predictor_type, "name", predictor_type)) == "CHINCHILLA3" else "--"


def _plotted_series_end(
    rankings: dict[tuple[str, Any], dict[str, Any]],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    predictor_types: list[Any],
) -> float:
    """Largest x reached by any (optimizer, predictor) curve about to be plotted."""
    ends: list[float] = []
    for opt_name in optimizers.keys():
        for predictor_type in predictor_types:
            rank_data = rankings.get((opt_name, predictor_type))
            if rank_data is None:
                continue
            flop_counts = rank_data["flop_counts_scaled"]
            if len(flop_counts) > 0:
                ends.append(float(flop_counts[-1]))
    return max(ends, default=0.0)


def calculate_predicted_n_param_error_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_types: list[Any],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    cost_estimator: Any = None,
    params_estimator: Any = None,
    search_space: Any = None,
    num_grid_points: int = 100,
) -> dict[tuple[str, Any], dict[str, Any]]:
    """Rank every (optimizer, n_param predictor) pair by model-size prediction error.

    Uses the same per-seed prediction trajectories as
    plot_predicted_n_param_over_flops, converts them to
    |log10 N_pred - log10 N*| error curves on a shared normalized FLOP grid,
    then ranks all pairs against each other at every grid point (rank 1 = the
    smallest error). Ranks are computed per seed and aggregated as mean with
    standard error.
    """
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")
    if not predictor_types:
        raise ValueError("predictor_types must not be empty")

    from src.scale_param_predictor import ScaleParamPredictorTypes, get_scale_param_fit_trajectory

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)

    true_target_n_param: float | None = None
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_n_params = np.asarray(true_pareto_info.get("n_params", []), dtype=float)
        valid_true = np.isfinite(true_flops) & np.isfinite(true_n_params) & (true_flops > 0) & (true_n_params > 0)
        if np.any(valid_true):
            valid_flops = true_flops[valid_true]
            valid_n_params = true_n_params[valid_true]
            closest_idx = int(np.argmin(np.abs(valid_flops - target_flop)))
            true_target_n_param = float(valid_n_params[closest_idx])

    if true_target_n_param is None:
        raise ValueError(
            "true_pareto_info with finite positive flops and n_params is required to rank prediction errors"
        )

    log_true_n_param = float(np.log10(true_target_n_param))

    # Per (optimizer, predictor) -> per seed -> (flops, error) curve.
    error_curves: dict[tuple[str, Any], dict[Any, dict[str, np.ndarray]]] = {}
    all_flop_arrays: list[np.ndarray] = []

    for opt_name, seeds in optimizers.items():
        for seed in seeds:
            try:
                _losses, _cum_costs, _max_cost, rows, learning_curve_rows, _configs = process_seed(
                    path=base_path / opt_name,
                    seed=seed,
                    key_to_extract=key_to_extract,
                    loss_key=loss_key,
                    return_incumbent=False,
                )
            except Exception as e:
                print(f"Warning: could not load rows for {opt_name} seed {seed}: {e}")
                continue

            if not rows:
                continue

            trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
            trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
            cumulative_flops = np.cumsum(trial_flops)

            for predictor_type in predictor_types:
                use_learning_curves = (
                    predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT
                    and len(learning_curve_rows) > 0
                )
                if use_learning_curves:
                    source_rows = learning_curve_rows
                    if len(cumulative_flops) != len(learning_curve_rows):
                        print(
                            f"Warning: Expected cumulative_flops and learning_curve_rows to have the same length, "
                            f"but got {len(cumulative_flops)} and {len(learning_curve_rows)}"
                        )
                        continue
                    total_trials = len(cumulative_flops)
                else:
                    source_rows = rows
                    total_trials = len(rows)

                if total_trials <= 0:
                    continue

                _cache_dir = prediction_cache_dir(base_path, opt_name, seed, loss_key)
                try:
                    _trajectory = get_scale_param_fit_trajectory(
                        predictor_type,
                        rows=source_rows,
                        best_loss_guess=best_loss_guess,
                        exponent_bounds=exponent_bounds,
                        space=search_space,
                        configs=_configs,
                        cost_estimator=cost_estimator,
                        params_estimator=params_estimator,
                        cache_dir=_cache_dir,
                    )
                except Exception as e:
                    print(f"Warning: could not fit {predictor_type} for {opt_name} seed {seed}: {e}")
                    continue

                pred_n_params_arr = _trajectory.predict(target_flop)[:, 0]

                x = np.asarray(cumulative_flops[:total_trials], dtype=float)
                y = np.asarray(pred_n_params_arr, dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if not np.any(valid):
                    continue

                x = x[valid]
                error = np.abs(np.log10(y[valid]) - log_true_n_param)

                if flop_range is not None:
                    range_mask = (x >= flop_range[0]) & (x <= flop_range[1])
                    x = x[range_mask]
                    error = error[range_mask]

                if len(x) == 0:
                    continue

                all_flop_arrays.append(x)
                error_curves.setdefault((opt_name, predictor_type), {})[seed] = {
                    "flops": x,
                    "error": error,
                }

    return _aggregate_error_curves(
        error_curves=error_curves,
        all_flop_arrays=all_flop_arrays,
        num_grid_points=num_grid_points,
    )


def _aggregate_error_curves(
    *,
    error_curves: dict[tuple[str, Any], dict[Any, dict[str, np.ndarray]]],
    all_flop_arrays: list[np.ndarray],
    num_grid_points: int,
) -> dict[tuple[str, Any], dict[str, Any]]:
    """Aggregate per-seed error curves on a shared normalized FLOP grid.

    Returns both the absolute prediction error (mean/median with standard error
    across seeds) and the relative rank of each (optimizer, predictor) pair.
    Shared by the loss and n_param paths so both use identical normalization,
    interpolation, tie handling and aggregation.
    """
    norm_bounds = _get_global_flop_bounds(all_flop_arrays)
    if norm_bounds is None:
        return {}

    normalized_grid = np.linspace(0.0, 1.0, num=num_grid_points)
    # Normalization is linear in log10(FLOPs), so the grid maps back to actual
    # FLOPs exactly -- kept so per-benchmark panels can use a real log FLOP axis
    # while the cross-benchmark rank panel stays on [0, 1].
    actual_flop_grid = 10 ** (
        np.log10(norm_bounds[0])
        + normalized_grid * (np.log10(norm_bounds[1]) - np.log10(norm_bounds[0]))
    )

    # Interpolate every (pair, seed) error curve onto the shared grid.
    interpolated: dict[tuple[str, Any], dict[Any, np.ndarray]] = {}
    for key, per_seed in error_curves.items():
        for seed, data in per_seed.items():
            x = _normalize_flop_counts(data["flops"], norm_bounds[0], norm_bounds[1])
            y = data["error"]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            if len(x_unique) < 2:
                continue

            y_interp = np.interp(normalized_grid, x_unique, y_unique)
            outside = (normalized_grid < x_unique[0]) | (normalized_grid > x_unique[-1])
            y_interp[outside] = np.nan
            interpolated.setdefault(key, {})[seed] = y_interp

    if not interpolated:
        return {}

    all_seed_values = sorted({seed for per_seed in interpolated.values() for seed in per_seed}, key=str)

    # Rank the pairs against each other per seed, at every grid point.
    rank_curves: dict[tuple[str, Any], list[np.ndarray]] = {key: [] for key in interpolated}
    for seed in all_seed_values:
        seed_errors = {key: per_seed[seed] for key, per_seed in interpolated.items() if seed in per_seed}
        if not seed_errors:
            continue

        seed_ranks = {key: np.full_like(normalized_grid, np.nan, dtype=float) for key in seed_errors}
        for grid_idx in range(len(normalized_grid)):
            valid_errors = {
                key: errors[grid_idx]
                for key, errors in seed_errors.items()
                if np.isfinite(errors[grid_idx])
            }
            if not valid_errors:
                continue

            for key in seed_errors:
                if key in valid_errors:
                    current_error = valid_errors[key]
                    seed_ranks[key][grid_idx] = sum(
                        1 for other in valid_errors.values() if other < current_error
                    ) + 1
                else:
                    seed_ranks[key][grid_idx] = len(valid_errors) + 1

        for key, ranks in seed_ranks.items():
            if np.any(np.isfinite(ranks)):
                rank_curves[key].append(ranks)

    rankings: dict[tuple[str, Any], dict[str, Any]] = {}
    for key, curves in rank_curves.items():
        if not curves:
            continue

        rank_array = np.asarray(curves, dtype=float)
        valid_counts = np.sum(np.isfinite(rank_array), axis=0)
        has_values = valid_counts > 0
        if not np.any(has_values):
            continue

        mean_ranks = np.full(rank_array.shape[1], np.nan, dtype=float)
        se_ranks = np.full(rank_array.shape[1], np.nan, dtype=float)
        mean_ranks[has_values] = np.nanmean(rank_array[:, has_values], axis=0)
        std_ranks = np.nanstd(rank_array[:, has_values], axis=0)
        se_ranks[has_values] = std_ranks / np.sqrt(valid_counts[has_values])

        # Absolute errors are aggregated on the same grid points as the ranks so
        # the two views of a pair are directly comparable. Heavy tails are real
        # here (a single diverged seed can move the mean by >1 dex), so the
        # median is carried alongside the mean.
        error_array = np.asarray(
            [values for values in interpolated.get(key, {}).values()], dtype=float
        )
        mean_errors = np.full(rank_array.shape[1], np.nan, dtype=float)
        median_errors = np.full(rank_array.shape[1], np.nan, dtype=float)
        se_errors = np.full(rank_array.shape[1], np.nan, dtype=float)
        if error_array.size:
            error_counts = np.sum(np.isfinite(error_array), axis=0)
            has_errors = error_counts > 0
            with np.errstate(invalid="ignore"):
                mean_errors[has_errors] = np.nanmean(error_array[:, has_errors], axis=0)
                median_errors[has_errors] = np.nanmedian(error_array[:, has_errors], axis=0)
                std_errors = np.nanstd(error_array[:, has_errors], axis=0)
            se_errors[has_errors] = std_errors / np.sqrt(error_counts[has_errors])

        opt_name, predictor_type = key
        rankings[key] = {
            "optimizer": opt_name,
            "predictor_type": predictor_type,
            "flop_counts_scaled": normalized_grid[has_values],
            "flop_counts": actual_flop_grid[has_values],
            "mean_ranks": mean_ranks[has_values],
            "std_ranks": se_ranks[has_values],
            "seed_ranks": curves,
            "mean_errors": mean_errors[has_values],
            "median_errors": median_errors[has_values],
            "std_errors": se_errors[has_values],
        }

    return rankings


def calculate_predicted_loss_error_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_types: list[Any],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    cost_estimator: Any = None,
    search_space: Any = None,
    num_grid_points: int = 100,
) -> dict[tuple[str, Any], dict[str, Any]]:
    """Rank every (optimizer, loss predictor) pair by loss-prediction error.

    The loss counterpart of calculate_predicted_n_param_error_rankings: same
    per-seed trajectories as plot_predicted_loss_over_flops, turned into
    |L_pred - L*| error curves on a shared normalized FLOP grid, then ranked
    across all pairs at every grid point (rank 1 = smallest error) per seed and
    aggregated as mean with standard error.
    """
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")
    if not predictor_types:
        raise ValueError("predictor_types must not be empty")

    from src.loss_predictor import get_loss_fit_trajectory

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name, loss_key)
    exponent_bounds = _get_exponent_bounds(loss_key)

    true_target_loss: float | None = None
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
        valid_true = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
        if np.any(valid_true):
            valid_flops = true_flops[valid_true]
            valid_losses = true_losses[valid_true]
            closest_idx = int(np.argmin(np.abs(valid_flops - target_flop)))
            true_target_loss = float(valid_losses[closest_idx])

    if true_target_loss is None:
        raise ValueError(
            "true_pareto_info with finite positive flops and losses is required to rank prediction errors"
        )

    error_curves: dict[tuple[str, Any], dict[Any, dict[str, np.ndarray]]] = {}
    all_flop_arrays: list[np.ndarray] = []

    for opt_name, seeds in optimizers.items():
        for seed in seeds:
            try:
                _losses, _cum_costs, _max_cost, rows, _learning_curve_rows, _configs = process_seed(
                    path=base_path / opt_name,
                    seed=seed,
                    key_to_extract=key_to_extract,
                    loss_key=loss_key,
                    return_incumbent=False,
                )
            except Exception as e:
                print(f"Warning: could not load rows for {opt_name} seed {seed}: {e}")
                continue

            if not rows:
                continue

            trial_flops = np.asarray([float(r[3]) for r in rows], dtype=float)
            trial_flops = np.where(np.isfinite(trial_flops) & (trial_flops > 0), trial_flops, 0.0)
            cumulative_flops = np.cumsum(trial_flops)

            for predictor_type in predictor_types:
                _cache_dir = prediction_cache_dir(base_path, opt_name, seed, loss_key)
                try:
                    _trajectory = get_loss_fit_trajectory(
                        predictor_type,
                        rows=rows,
                        best_loss_guess=best_loss_guess,
                        exponent_bounds=exponent_bounds,
                        space=search_space,
                        configs=_configs,
                        cost_estimator=cost_estimator,
                        cache_dir=_cache_dir,
                    )
                except Exception as e:
                    print(f"Warning: could not fit {predictor_type} for {opt_name} seed {seed}: {e}")
                    continue

                x = np.asarray(cumulative_flops, dtype=float)
                y = np.asarray(_trajectory.predict(target_flop), dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if not np.any(valid):
                    continue

                x = x[valid]
                error = np.abs(y[valid] - true_target_loss)

                if flop_range is not None:
                    range_mask = (x >= flop_range[0]) & (x <= flop_range[1])
                    x = x[range_mask]
                    error = error[range_mask]

                if len(x) == 0:
                    continue

                all_flop_arrays.append(x)
                error_curves.setdefault((opt_name, predictor_type), {})[seed] = {
                    "flops": x,
                    "error": error,
                }

    return _aggregate_error_curves(
        error_curves=error_curves,
        all_flop_arrays=all_flop_arrays,
        num_grid_points=num_grid_points,
    )


def plot_predictor_error_rankings(
    *,
    rankings: dict[tuple[str, Any], dict[str, Any]],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    predictor_types: list[Any],
    figsize: tuple[float, float] = (5.5, 3.5),
    title: str = "",
) -> tuple[Any, Any]:
    """Plot one relative-ranking curve per (optimizer, predictor) pair.

    Optimizers are distinguished by color (from plot_styles), predictors by
    linestyle: solid for the joint-parametric Chinchilla fit, dashed otherwise.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []
    common_end = _plotted_series_end(rankings, optimizers, predictor_types)

    num_series = 0
    for opt_name in optimizers.keys():
        for predictor_type in predictor_types:
            rank_data = rankings.get((opt_name, predictor_type))
            if rank_data is None:
                continue

            flop_counts = rank_data["flop_counts_scaled"]
            mean_ranks = rank_data["mean_ranks"]
            se_ranks = rank_data["std_ranks"]
            if len(flop_counts) == 0:
                continue

            # Pairs that run out of data early are held constant to the shared end.
            flop_counts, (mean_ranks, se_ranks) = _extend_curve_to(
                flop_counts, [mean_ranks, se_ranks], common_end
            )

            num_series += 1
            color = get_optimizer_color(opt_name)
            predictor_label = get_predictor_label(predictor_type)
            x_series_for_limits.append(flop_counts)

            ax.plot(
                flop_counts,
                mean_ranks,
                label=f"{get_optimizer_label(opt_name)} ({predictor_label})",
                color=color,
                linestyle=_predictor_linestyle(predictor_type),
                linewidth=2.5,
            )
            ax.fill_between(
                flop_counts,
                mean_ranks - se_ranks,
                mean_ranks + se_ranks,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Normalized FLOPs", fontsize=20)
    ax.set_ylabel("Relative Rank", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    left, right = _get_common_xlim(x_series_for_limits)
    ax.set_xlim(left, right)

    num_series = max(num_series, 1)
    ax.invert_yaxis()
    ax.set_ylim(num_series + 0.5, 0.5)
    ax.set_yticks(range(1, num_series + 1))
    ax.set_yticklabels(range(1, num_series + 1))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3),
               ncol=min(2, max(len(handles), 1)), fontsize=16, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def plot_predictor_abs_errors(
    *,
    rankings: dict[tuple[str, Any], dict[str, Any]],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    predictor_types: list[Any],
    ylabel: str,
    figsize: tuple[float, float] = (5.5, 3.5),
    title: str = "",
    log_y: bool = False,
    use_median: bool = False,
) -> tuple[Any, Any]:
    """Plot absolute prediction error per (optimizer, predictor) pair.

    Same series and styling as plot_predictor_error_rankings, but on the error
    itself rather than its rank -- so a pair that "wins" the ranking while still
    being far from the truth is visible as such.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []
    value_key = "median_errors" if use_median else "mean_errors"
    common_end = _plotted_series_end(rankings, optimizers, predictor_types)

    for opt_name in optimizers.keys():
        for predictor_type in predictor_types:
            rank_data = rankings.get((opt_name, predictor_type))
            if rank_data is None:
                continue

            flop_counts = rank_data["flop_counts_scaled"]
            errors = np.asarray(rank_data[value_key], dtype=float)
            se_errors = np.asarray(rank_data["std_errors"], dtype=float)
            if len(flop_counts) == 0 or not np.any(np.isfinite(errors)):
                continue

            # Pairs that run out of data early are held constant to the shared end.
            flop_counts, (errors, se_errors) = _extend_curve_to(
                flop_counts, [errors, se_errors], common_end
            )

            color = get_optimizer_color(opt_name)
            predictor_label = get_predictor_label(predictor_type)
            x_series_for_limits.append(flop_counts)

            ax.plot(
                flop_counts,
                errors,
                label=f"{get_optimizer_label(opt_name)} ({predictor_label})",
                color=color,
                linestyle=_predictor_linestyle(predictor_type),
                linewidth=2.5,
            )
            lower = errors - se_errors
            if log_y:
                lower = np.maximum(lower, np.finfo(float).tiny)
            ax.fill_between(
                flop_counts,
                lower,
                errors + se_errors,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Normalized FLOPs", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if log_y:
        ax.set_yscale("log")

    left, right = _get_common_xlim(x_series_for_limits)
    ax.set_xlim(left, right)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3),
               ncol=min(2, max(len(handles), 1)), fontsize=16, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def plot_prediction_error_composite_across_benchmarks(
    *,
    metric: str,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    predictor_types: list[Any],
    rankings_per_benchmark: dict[str, dict[tuple[str, Any], dict[str, Any]]],
    log_y: bool = False,
    use_median: bool = False,
    y_clip_quantile: float | None = 0.99,
    num_grid_points: int = 100,
) -> tuple[Any, Any]:
    """One panel of absolute error per benchmark, plus an across-benchmark rank panel.

    The per-benchmark panels show |L - L*| (or |log N - log N*|) against actual
    cumulative FLOPs on a log axis, so a pair that is merely the least-bad of a
    bad field is visible as such. The final panel ranks the same pairs against
    each other pooled over benchmarks, on a normalized [0, 1] FLOP axis -- ranks
    are comparable across benchmarks, absolute errors are not, and the benchmarks
    do not share a FLOP range.

    Optimizers are distinguished by color (from plot_styles) and predictors by
    linestyle: solid for the joint-parametric Chinchilla fit, dashed otherwise.
    y_clip_quantile caps the y-axis at that quantile of the plotted values so a
    diverged seed cannot flatten every other curve.
    """
    from src.plotting.plot_styles import BENCHMARK_TITLES

    if metric not in {"loss", "n_param"}:
        raise ValueError("metric must be 'loss' or 'n_param'")

    ylabel = r"$|L - L^*|$" if metric == "loss" else r"$|\log N - \log N^*|$"
    value_key = "median_errors" if use_median else "mean_errors"

    num_panels = len(benchmarks) + 1
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 4), squeeze=False)
    axes = axes[0]

    shared_handles: list[Any] = []
    shared_labels: list[str] = []

    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx]
        rankings = rankings_per_benchmark.get(benchmark_name, {})
        x_series_for_limits: list[np.ndarray] = []

        # Collect first so the clip is computed over everything this panel will
        # draw; a single diverged seed otherwise sets the y-range for all eight.
        panel_records: list[dict[str, Any]] = []
        for opt_name in optimizers.keys():
            for predictor_type in predictor_types:
                rank_data = rankings.get((opt_name, predictor_type))
                if rank_data is None:
                    continue

                flop_counts = np.asarray(rank_data["flop_counts"], dtype=float)
                errors = np.asarray(rank_data[value_key], dtype=float)
                se_errors = np.asarray(rank_data["std_errors"], dtype=float)
                if len(flop_counts) == 0 or not np.any(np.isfinite(errors)):
                    continue

                predictor_label = get_predictor_label(predictor_type)
                panel_records.append({
                    "opt_name": opt_name,
                    "predictor_type": predictor_type,
                    "label": f"{get_optimizer_label(opt_name)} ({predictor_label})",
                    "x": flop_counts,
                    "y": errors,
                    "yerr": se_errors,
                })

        y_clip_upper = _get_log_scale_upper_clip(
            [record["y"] for record in panel_records]
            + [record["y"] + record["yerr"] for record in panel_records],
            y_clip_quantile,
        )

        for record in panel_records:
            flop_counts = record["x"]
            errors = record["y"]
            upper = errors + record["yerr"]
            lower = errors - record["yerr"]
            if y_clip_upper is not None:
                errors = np.minimum(errors, y_clip_upper)
                upper = np.minimum(upper, y_clip_upper)
                lower = np.minimum(lower, y_clip_upper)
            if log_y:
                lower = np.maximum(lower, np.finfo(float).tiny)

            color = get_optimizer_color(record["opt_name"])
            x_series_for_limits.append(flop_counts)

            line, = ax.plot(
                flop_counts,
                errors,
                label=record["label"],
                color=color,
                linestyle=_predictor_linestyle(record["predictor_type"]),
                linewidth=2.5,
            )
            ax.fill_between(flop_counts, lower, upper, alpha=0.2, color=color)

            if record["label"] not in shared_labels:
                shared_handles.append(line)
                shared_labels.append(record["label"])

        letter = string.ascii_lowercase[idx]
        ax.set_title(f"({letter}) {BENCHMARK_TITLES.get(benchmark_name, benchmark_name)}", loc="left", fontsize=20)
        ax.set_xlabel("Cumulative FLOPs", fontsize=20)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=20)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if y_clip_upper is not None:
            ax.set_ylim(top=y_clip_upper)
        if x_series_for_limits:
            left, right = _get_common_xlim(x_series_for_limits)
            ax.set_xlim(left, right)

    # Across-benchmark ranking: pool the per-seed rank curves from every
    # benchmark. Ranks are comparable across benchmarks; the raw errors are not.
    rank_ax = axes[-1]
    normalized_grid = np.linspace(0.0, 1.0, num=num_grid_points)
    num_series = 0

    for opt_name in optimizers.keys():
        for predictor_type in predictor_types:
            pooled: list[np.ndarray] = []
            for benchmark_name in benchmarks:
                rank_data = rankings_per_benchmark.get(benchmark_name, {}).get((opt_name, predictor_type))
                if rank_data is None:
                    continue
                pooled.extend(
                    np.asarray(curve, dtype=float)
                    for curve in rank_data.get("seed_ranks", [])
                    if len(curve) == num_grid_points
                )

            if not pooled:
                continue

            rank_array = np.asarray(pooled, dtype=float)
            counts = np.sum(np.isfinite(rank_array), axis=0)
            has_values = counts > 0
            if not np.any(has_values):
                continue

            mean_ranks = np.nanmean(rank_array[:, has_values], axis=0)
            std_ranks = np.nanstd(rank_array[:, has_values], axis=0)
            se_ranks = std_ranks / np.sqrt(counts[has_values])

            num_series += 1
            color = get_optimizer_color(opt_name)

            rank_ax.plot(
                normalized_grid[has_values],
                mean_ranks,
                color=color,
                linestyle=_predictor_linestyle(predictor_type),
                linewidth=2.5,
            )
            rank_ax.fill_between(
                normalized_grid[has_values],
                mean_ranks - se_ranks,
                mean_ranks + se_ranks,
                alpha=0.2,
                color=color,
            )

    letter = string.ascii_lowercase[len(benchmarks)]
    rank_ax.set_title(f"({letter}) Across Benchmarks", loc="left", fontsize=20)
    rank_ax.set_xlabel("Normalized FLOPs", fontsize=20)
    rank_ax.set_ylabel("Relative Rank", fontsize=20)
    rank_ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    rank_ax.set_xlim(0.0, 1.0)
    num_series = max(num_series, 1)
    rank_ax.invert_yaxis()
    rank_ax.set_ylim(num_series + 0.5, 0.5)
    rank_ax.set_yticks(range(1, num_series + 1))
    rank_ax.set_yticklabels(range(1, num_series + 1))

    if shared_handles:
        fig.legend(
            shared_handles, shared_labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.22),
            ncol=min(4, len(shared_labels)), fontsize=16, frameon=True,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes


def print_predictor_error_rankings(
    rankings: dict[tuple[str, Any], dict[str, Any]],
    *,
    header: str = "",
) -> None:
    """Print absolute prediction error and relative rank per (optimizer, predictor) pair.

    The error columns are what the rank columns hide: a pair can rank first and
    still be far from the truth, and the mean/median gap flags pairs whose error
    is driven by a few diverged seeds.
    """
    if not rankings:
        print("    (no prediction rankings available)")
        return

    rows: list[tuple[str, float, float, float, float, float]] = []
    for (opt_name, predictor_type), rank_data in rankings.items():
        mean_ranks = np.asarray(rank_data["mean_ranks"], dtype=float)
        if len(mean_ranks) == 0:
            continue
        mean_errors = np.asarray(rank_data.get("mean_errors", []), dtype=float)
        median_errors = np.asarray(rank_data.get("median_errors", []), dtype=float)
        predictor_label = get_predictor_label(predictor_type)
        rows.append((
            f"{get_optimizer_label(opt_name)} ({predictor_label})",
            float(np.nanmean(mean_ranks)),
            float(mean_ranks[-1]),
            float(np.nanmean(mean_errors)) if mean_errors.size else float("nan"),
            float(mean_errors[-1]) if mean_errors.size else float("nan"),
            float(median_errors[-1]) if median_errors.size else float("nan"),
        ))

    if not rows:
        print("    (no prediction rankings available)")
        return

    rows.sort(key=lambda row: row[1])

    if header:
        print(header)
    name_width = max(len(row[0]) for row in rows)
    print(
        f"    {'Method'.ljust(name_width)}  {'Avg Rank':>9}  {'Final Rank':>11}"
        f"  {'Avg |err|':>10}  {'Final |err|':>12}  {'Final med':>10}"
    )
    for name, avg_rank, final_rank, avg_err, final_err, final_med in rows:
        print(
            f"    {name.ljust(name_width)}  {avg_rank:>9.2f}  {final_rank:>11.2f}"
            f"  {avg_err:>10.4g}  {final_err:>12.4g}  {final_med:>10.4g}"
        )


def _format_hp_table_value(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(v):
        return "nan"
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-2):
        return f"{v:.3e}"
    if abs(v - round(v)) < 1e-8 and abs(v) < 1e6:
        return f"{int(round(v))}"
    return f"{v:.6g}"



def _get_true_pareto_idx_at_target_flop(
    *,
    true_pareto_info: dict[str, list] | None,
    target_flop: float,
) -> int | None:
    if true_pareto_info is None:
        return None

    true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
    true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    valid = np.isfinite(true_flops) & np.isfinite(true_losses)
    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]
    nearest_in_valid = int(np.argmin(np.abs(true_flops[valid] - float(target_flop))))
    return int(valid_indices[nearest_in_valid])


def _extract_true_pareto_config_row(
    *,
    true_pareto_info: dict[str, list] | None,
    target_flop: float,
) -> dict[str, Any]:
    idx = _get_true_pareto_idx_at_target_flop(
        true_pareto_info=true_pareto_info,
        target_flop=target_flop,
    )
    if idx is None or true_pareto_info is None:
        return {}

    cfg: dict[str, Any] = {}
    maybe_cfgs = true_pareto_info.get("configs", [])
    if isinstance(maybe_cfgs, list) and idx < len(maybe_cfgs) and isinstance(maybe_cfgs[idx], dict):
        cfg = {
            (k[len("config."):] if isinstance(k, str) and k.startswith("config.") else k): v
            for k, v in dict(maybe_cfgs[idx]).items()
        }
    losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    cfg["actual_loss"] = float(losses[idx]) if idx < len(losses) else float("nan")
    return cfg


def _extract_result_value(result: Any, key: str) -> Any:
    if not isinstance(result, dict):
        raise TypeError("evaluate_pipeline must return a dict-like result")
    if key in result:
        return result[key]

    current = result
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current

@split_by_priors(_composite_empty_placeholder)
def plot_predicted_loss_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop_per_benchmark: dict[str, float],
    true_pareto_info_per_benchmark: dict[str, dict],
    cost_estimator_per_benchmark: dict[str, Any],
    search_space_per_benchmark: dict[str, Any],
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted Loss Composite Across Benchmarks",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
    abs_loss_diff: bool = False,
    return_split: bool = False,
) -> Any:
    """Composite predicted-loss figures with all benchmarks + across-benchmark aggregate."""
    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    cols = num_subplots
    figsize = (6 * cols, 4)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    shared_handles = None
    shared_labels = None

    normalized_rank_grid = np.linspace(0.0, 1.0, 100)
    per_optimizer_rank_curves: dict[str, list[np.ndarray]] = {k: [] for k in optimizers.keys()}

    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            bench_flop_range = None
            if flop_ranges_per_benchmark is not None:
                bench_flop_range = flop_ranges_per_benchmark.get(benchmark_name)

            
            true_info = true_pareto_info_per_benchmark[benchmark_name]
            target_flop = target_flop_per_benchmark[benchmark_name]
            cost_estimator=cost_estimator_per_benchmark[benchmark_name]
            search_space=search_space_per_benchmark[benchmark_name]
            fig_bench, ax_bench = cast(tuple[Any, Any], plot_predicted_loss_over_flops(
                base_path=benchmark_path,
                optimizers=optimizers,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=(5.5, 3.5),
                title="",
                flop_range=bench_flop_range,
                true_pareto_info=true_info,
                normalized_flops=False,
                return_split=False,
                abs_loss_diff=abs_loss_diff,
                cost_estimator=cost_estimator,
                search_space=search_space,
            ))

            letter = string.ascii_lowercase[idx + 1]

            _replot_lines(ax_bench, ax)

            ax.set_title(f"({letter}) {BENCHMARK_TITLES.get(benchmark_name, benchmark_name)}", loc='left', fontsize=20)
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlim(ax_bench.get_xlim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            ax.set_yscale(ax_bench.get_yscale())
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()

            # Extract per-optimizer absolute-error curves and rank them on a
            # benchmark-local normalized FLOP grid.
            benchmark_error_curves: dict[str, np.ndarray] = {}
            for opt_name in optimizers.keys():
                label = get_optimizer_label(opt_name)
                pred_xs = np.array([])
                pred_ys = np.array([])
                for line in ax_bench.get_lines():
                    if line.get_label() == label:
                        pred_xs = np.asarray(line.get_xdata(), dtype=float)
                        pred_ys = np.asarray(line.get_ydata(), dtype=float)
                        break

                if len(pred_xs) == 0 or len(pred_ys) == 0:
                    continue

                true_flops = np.asarray(true_info.get("flops", []), dtype=float)
                true_losses = np.asarray(true_info.get("loss", []), dtype=float)
                valid_true = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0)
                if not np.any(valid_true):
                    continue

                valid_flops = true_flops[valid_true]
                valid_losses = true_losses[valid_true]
                closest_idx = int(np.argmin(np.abs(valid_flops - target_flop)))
                true_val = float(valid_losses[closest_idx])

                finite_mask = np.isfinite(pred_xs) & np.isfinite(pred_ys) & (pred_xs > 0)
                if not np.any(finite_mask):
                    continue

                pred_xs = pred_xs[finite_mask]
                pred_ys = pred_ys[finite_mask]
                error_curve = pred_ys if abs_loss_diff else np.abs(pred_ys - true_val)
                norm_bounds = _get_global_flop_bounds([pred_xs])
                if norm_bounds is None:
                    continue

                normalized_xs = _normalize_flop_counts(pred_xs, norm_bounds[0], norm_bounds[1])
                order = np.argsort(normalized_xs)
                normalized_xs = normalized_xs[order]
                error_curve = error_curve[order]
                normalized_xs, unique_idx = np.unique(normalized_xs, return_index=True)
                error_curve = error_curve[unique_idx]
                if len(normalized_xs) < 2:
                    continue

                benchmark_error_curves[opt_name] = np.interp(
                    normalized_rank_grid,
                    normalized_xs,
                    error_curve,
                    left=error_curve[0],
                    right=error_curve[-1],
                )

            benchmark_rank_curves = {
                opt_name: np.full_like(normalized_rank_grid, np.nan, dtype=float)
                for opt_name in benchmark_error_curves.keys()
            }
            for grid_idx in range(len(normalized_rank_grid)):
                valid_errors = {
                    opt_name: errors[grid_idx]
                    for opt_name, errors in benchmark_error_curves.items()
                    if np.isfinite(errors[grid_idx])
                }
                if not valid_errors:
                    continue

                for opt_name in benchmark_error_curves.keys():
                    if opt_name in valid_errors:
                        current_error = valid_errors[opt_name]
                        benchmark_rank_curves[opt_name][grid_idx] = sum(
                            1 for other_error in valid_errors.values() if other_error < current_error
                        ) + 1
                    else:
                        benchmark_rank_curves[opt_name][grid_idx] = len(valid_errors) + 1

            for opt_name, rank_curve in benchmark_rank_curves.items():
                if np.any(np.isfinite(rank_curve)):
                    per_optimizer_rank_curves[opt_name].append(rank_curve)

            plt.close(fig_bench)

        except Exception as e:
            print(f"Warning: Could not plot predicted loss for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}", ha="center", va="center", transform=ax.transAxes)

    # Aggregate ranking panel on the left (axes[0]).
    agg_ax = axes[0]
    try:
        for opt_name in optimizers.keys():
            rank_curves = per_optimizer_rank_curves[opt_name]
            if not rank_curves:
                continue

            rank_array = np.array(rank_curves, dtype=float)
            mean_ranks = np.nanmean(rank_array, axis=0)
            std_ranks = np.nanstd(rank_array, axis=0)
            valid_counts = np.maximum(np.sum(np.isfinite(rank_array), axis=0), 1)
            se_ranks = std_ranks / np.sqrt(valid_counts)

            color = get_optimizer_color(opt_name)
            linestyle = get_optimizer_linestyle(opt_name)

            agg_ax.plot(
                normalized_rank_grid,
                mean_ranks,
                label=get_optimizer_label(opt_name),
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )

            agg_ax.fill_between(
                normalized_rank_grid,
                mean_ranks - se_ranks,
                mean_ranks + se_ranks,
                alpha=0.2,
                color=color,
            )

        agg_ax.set_xlabel("Normalized FLOPs", fontsize=20)
        agg_ax.set_ylabel("Loss Prediction Error Rank", fontsize=20)
        agg_ax.set_title("(a) Across Benchmarks", loc='left', fontsize=20)
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        agg_ax.set_xlim(0.0, 1.0)
        agg_ax.invert_yaxis()
        num_optimizers = len(optimizers)
        agg_ax.set_ylim(num_optimizers + 0.5, 0.5)
        agg_ax.set_yticks(range(1, num_optimizers + 1))
        agg_ax.set_yticklabels(range(1, num_optimizers + 1))

    except Exception as e:
        print(f"Warning: Could not compute aggregate predicted-loss ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", ha="center", va="center", transform=agg_ax.transAxes)

    # Remove individual legends
    for ax in axes[1:]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')

    return fig, axes


@split_by_priors(_composite_empty_placeholder)
def plot_predicted_n_param_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop_per_benchmark: dict[str, float],
    true_pareto_info_per_benchmark: dict[str, dict],
    cost_estimator_per_benchmark: dict[str, Any],
    search_space_per_benchmark: dict[str, Any],
    params_estimator_per_benchmark: dict[str, Any],
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted n_param Composite Across Benchmarks",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
    y_clip_quantile: float | None = 0.99,
    abs_param_diff: bool = False,
    return_split: bool = False,
) -> Any:
    """Composite predicted-n_param figures with per-benchmark panels plus an across-benchmark ranking panel."""
    num_subplots = len(benchmarks) + 1
    cols = num_subplots
    figsize = (6 * cols, 4)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    shared_handles = None
    shared_labels = None
    normalized_rank_grid = np.linspace(0.0, 1.0, 100)
    per_optimizer_rank_curves: dict[str, list[np.ndarray]] = {k: [] for k in optimizers.keys()}

    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx + 1]
        try:
            benchmark_path = base_path / benchmark_name
            bench_flop_range = None
            if flop_ranges_per_benchmark is not None:
                bench_flop_range = flop_ranges_per_benchmark.get(benchmark_name)

            true_info = true_pareto_info_per_benchmark[benchmark_name]
            target_flop = target_flop_per_benchmark[benchmark_name]
            cost_estimator=cost_estimator_per_benchmark[benchmark_name]
            search_space=search_space_per_benchmark[benchmark_name]
            params_estimator=params_estimator_per_benchmark[benchmark_name]
            fig_bench, ax_bench = cast(tuple[Any, Any], plot_predicted_n_param_over_flops(
                base_path=benchmark_path,
                optimizers=optimizers,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=(5.5, 3.5),
                title="",
                flop_range=bench_flop_range,
                true_pareto_info=true_info,
                normalized_flops=False,
                return_split=False,
                y_clip_quantile=y_clip_quantile,
                abs_param_diff=abs_param_diff,
                cost_estimator=cost_estimator,
                search_space=search_space,
                params_estimator=params_estimator,
            ))

            letter = string.ascii_lowercase[idx + 1]
            _replot_lines(ax_bench, ax)

            ax.set_title(f"({letter}) {BENCHMARK_TITLES.get(benchmark_name, benchmark_name)}", loc='left', fontsize=20)
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlim(ax_bench.get_xlim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            ax.set_xscale(ax_bench.get_xscale())
            ax.set_yscale(ax_bench.get_yscale())
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if idx == 0:
                shared_handles, shared_labels = ax_bench.get_legend_handles_labels()

            benchmark_error_curves: dict[str, np.ndarray] = {}
            true_flops = np.asarray(true_info.get("flops", []), dtype=float)
            true_n_params = np.asarray(true_info.get("n_params", []), dtype=float)
            valid_true = np.isfinite(true_flops) & np.isfinite(true_n_params) & (true_flops > 0) & (true_n_params > 0)
            if np.any(valid_true):
                valid_flops = true_flops[valid_true]
                valid_n_params = true_n_params[valid_true]
                closest_idx = int(np.argmin(np.abs(valid_flops - target_flop)))
                true_val = float(valid_n_params[closest_idx])

                for opt_name in optimizers.keys():
                    label = get_optimizer_label(opt_name)
                    pred_xs = np.array([])
                    pred_ys = np.array([])
                    for line in ax_bench.get_lines():
                        if line.get_label() == label:
                            pred_xs = np.asarray(line.get_xdata(), dtype=float)
                            pred_ys = np.asarray(line.get_ydata(), dtype=float)
                            break

                    if len(pred_xs) == 0 or len(pred_ys) == 0:
                        continue

                    finite_mask = np.isfinite(pred_xs) & np.isfinite(pred_ys) & (pred_xs > 0) & (pred_ys > 0)
                    if not np.any(finite_mask):
                        continue

                    pred_xs = pred_xs[finite_mask]
                    pred_ys = pred_ys[finite_mask]
                    error_curve = pred_ys if abs_param_diff else np.abs(pred_ys - true_val)
                    norm_bounds = _get_global_flop_bounds([pred_xs])
                    if norm_bounds is None:
                        continue

                    normalized_xs = _normalize_flop_counts(pred_xs, norm_bounds[0], norm_bounds[1])
                    order = np.argsort(normalized_xs)
                    normalized_xs = normalized_xs[order]
                    error_curve = error_curve[order]
                    normalized_xs, unique_idx = np.unique(normalized_xs, return_index=True)
                    error_curve = error_curve[unique_idx]
                    if len(normalized_xs) < 2:
                        continue

                    benchmark_error_curves[opt_name] = np.interp(
                        normalized_rank_grid,
                        normalized_xs,
                        error_curve,
                        left=error_curve[0],
                        right=error_curve[-1],
                    )

            benchmark_rank_curves = {
                opt_name: np.full_like(normalized_rank_grid, np.nan, dtype=float)
                for opt_name in benchmark_error_curves.keys()
            }
            for grid_idx in range(len(normalized_rank_grid)):
                valid_errors = {
                    opt_name: errors[grid_idx]
                    for opt_name, errors in benchmark_error_curves.items()
                    if np.isfinite(errors[grid_idx])
                }
                if not valid_errors:
                    continue

                for opt_name in benchmark_error_curves.keys():
                    if opt_name in valid_errors:
                        current_error = valid_errors[opt_name]
                        benchmark_rank_curves[opt_name][grid_idx] = sum(
                            1 for other_error in valid_errors.values() if other_error < current_error
                        ) + 1
                    else:
                        benchmark_rank_curves[opt_name][grid_idx] = len(valid_errors) + 1

            for opt_name, rank_curve in benchmark_rank_curves.items():
                if np.any(np.isfinite(rank_curve)):
                    per_optimizer_rank_curves[opt_name].append(rank_curve)

            plt.close(fig_bench)

        except Exception as e:
            print(f"Warning: Could not plot predicted n_param for {benchmark_name}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {benchmark_name}", ha="center", va="center", transform=ax.transAxes)

    agg_ax = axes[0]
    try:
        for opt_name in optimizers.keys():
            rank_curves = per_optimizer_rank_curves[opt_name]
            if not rank_curves:
                continue

            rank_array = np.array(rank_curves, dtype=float)
            mean_ranks = np.nanmean(rank_array, axis=0)
            std_ranks = np.nanstd(rank_array, axis=0)
            valid_counts = np.maximum(np.sum(np.isfinite(rank_array), axis=0), 1)
            se_ranks = std_ranks / np.sqrt(valid_counts)

            color = get_optimizer_color(opt_name)
            linestyle = get_optimizer_linestyle(opt_name)

            agg_ax.plot(
                normalized_rank_grid,
                mean_ranks,
                label=get_optimizer_label(opt_name),
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )
            agg_ax.fill_between(
                normalized_rank_grid,
                mean_ranks - se_ranks,
                mean_ranks + se_ranks,
                alpha=0.2,
                color=color,
            )

        agg_ax.set_xlabel("Normalized FLOPs", fontsize=20)
        agg_ax.set_ylabel("Model size Prediction Error Rank", fontsize=20)
        agg_ax.set_title("(a) Across Benchmarks", loc='left', fontsize=20)
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        agg_ax.set_xlim(0.0, 1.0)
        agg_ax.invert_yaxis()
        num_optimizers = len(optimizers)
        agg_ax.set_ylim(num_optimizers + 0.5, 0.5)
        agg_ax.set_yticks(range(1, num_optimizers + 1))
        agg_ax.set_yticklabels(range(1, num_optimizers + 1))

    except Exception as e:
        print(f"Warning: Could not compute aggregate predicted-n_param ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", ha="center", va="center", transform=agg_ax.transAxes)

    for ax in axes[1:]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=min(4, len(optimizers)), fontsize=20, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for j in range(num_subplots, len(axes)):
        axes[j].axis('off')

    return fig, axes



def _warm_single_cache(args: tuple) -> str:
    """Top-level worker: fit (and cache) one predictor's trajectory for one
    (benchmark, optimizer, seed) combo.

    Must be a module-level function so ProcessPoolExecutor can pickle it.
    All heavy imports are deferred to the worker process.

    Fitting is target_flop-independent (see get_loss_fit_trajectory /
    get_scale_param_fit_trajectory / get_hp_fit_trajectory), so warming no
    longer varies by target_flop/ratio -- each (predictor_type, n_trials) is
    fit once here and can serve any target_flop cheaply afterward.
    """
    (base_path_str, benchmark_name, opt_name, seed, predictor_type, kind,
     key_to_extract, loss_key, best_loss_guess, exponent_bounds, override) = args

    from pathlib import Path as _Path
    from src.plotting.standalone_plots import process_seed as _process_seed

    bench_path = _Path(base_path_str) / benchmark_name
    predictor_slug = str(
        getattr(predictor_type, "value", None)
        or getattr(predictor_type, "name", None)
        or str(predictor_type)
    )
    tag = f"[{kind}/{predictor_slug}] {benchmark_name}/{opt_name}/seed_{seed}"
    cache_dir = prediction_cache_dir(bench_path, opt_name, seed, loss_key)

    try:
        _, _, _, rows, learning_curve_rows, configs = _process_seed(
            path=bench_path / opt_name,
            seed=seed,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            return_incumbent=False,
        )
    except Exception as e:
        return f"SKIP {tag}: {e}"

    if not rows:
        return f"SKIP {tag}: no rows"

    from src.plotting.consts import (
        search_space_per_benchmark as _spaces,
        cost_estimator_per_benchmark as _cost_ests,
        params_estimator_per_benchmark as _params_ests,
    )
    _space = _spaces[benchmark_name]
    _cost_est = _cost_ests[benchmark_name]
    _params_est = _params_ests[benchmark_name]

    try:
        if kind == "loss":
            from src.loss_predictor import get_loss_fit_trajectory
            get_loss_fit_trajectory(
                predictor_type,
                rows=rows,
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
                space=_space,
                configs=configs,
                cost_estimator=_cost_est,
                cache_dir=cache_dir,
                override=override,
            )

        elif kind == "nparam":
            from src.scale_param_predictor import get_scale_param_fit_trajectory, ScaleParamPredictorTypes as _SPT
            use_lc = predictor_type == _SPT.POWER_LAW_FIT and len(learning_curve_rows) > 0
            source_rows = learning_curve_rows if use_lc else rows
            if len(source_rows) == 0:
                return f"SKIP {tag}: no trials"
            get_scale_param_fit_trajectory(
                predictor_type,
                rows=source_rows,
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
                space=_space,
                configs=configs,
                cost_estimator=_cost_est,
                params_estimator=_params_est,
                cache_dir=cache_dir,
                override=override,
            )

        elif kind == "hp":
            from src.hp_predictor import get_hp_fit_trajectory
            flops_list = [float(r[3]) for r in rows]
            losses_list = [float(r[2]) for r in rows]
            get_hp_fit_trajectory(
                _space,
                predictor_type,
                configs=configs,
                losses=losses_list,
                flops=flops_list,
                cost_estimator=_cost_est,
                cache_dir=cache_dir,
                override=override,
            )

        else:
            return f"SKIP {tag}: unknown kind {kind!r}"

    except Exception as e:
        return f"ERROR {tag}: {e}"

    return f"DONE {tag}"


def warm_prediction_cache(
    *,
    base_path: str | Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int | str]],
    loss_predictor_types: list[Any] | None = None,
    nparam_predictor_types: list[Any] | None = None,
    hp_predictor_types: list[Any] | None = None,
    key_to_extract: str | None = "cost",
    loss_key: str = "objective_to_minimize",
    override: bool = False,
) -> None:
    """Pre-warm the prediction fit cache for all (benchmark, optimizer, seed, predictor) combos.

    Fitting is target_flop-independent (see get_loss_fit_trajectory /
    get_scale_param_fit_trajectory / get_hp_fit_trajectory), so warming no
    longer sweeps a target_flop/ratio axis -- each (predictor_type, n_trials)
    is fit once here and can serve any target_flop cheaply afterward.

    Runs sequentially — parallelism is achieved by launching multiple jobs via
    pred_cache_gen.py with different --optimizer / --benchmark arguments.
    Safe to re-run: already-cached fits are loaded, not recomputed, unless
    `override=True`, which forces every fit to be recomputed and the cache
    file overwritten with the new result.

    For GP_POSTERIOR predictors the trained model is shared across loss,
    nparam, and hp kinds (same cache_dir, same n), so each (optimizer, seed)
    trains the GP only once per trial prefix regardless of how many kinds are
    requested.
    """
    from src.loss_predictor import LossPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    from src.hp_predictor import HPPredictorTypes

    if loss_predictor_types is None:
        loss_predictor_types = list(LossPredictorTypes)
    if nparam_predictor_types is None:
        nparam_predictor_types = list(ScaleParamPredictorTypes)
    if hp_predictor_types is None:
        hp_predictor_types = list(HPPredictorTypes)

    base_path = Path(base_path)
    tasks: list[tuple] = []
    for benchmark_name in benchmarks:
        best_loss_guess = _get_best_loss_guess(benchmark_name, loss_key)
        exponent_bounds = _get_exponent_bounds(loss_key)
        for opt_name, seeds in optimizers.items():
            for seed in seeds:
                for p_type in loss_predictor_types:
                    tasks.append((
                        str(base_path), benchmark_name, opt_name, seed,
                        p_type, "loss",
                        key_to_extract, loss_key, best_loss_guess, exponent_bounds, override,
                    ))
                for p_type in nparam_predictor_types:
                    tasks.append((
                        str(base_path), benchmark_name, opt_name, seed,
                        p_type, "nparam",
                        key_to_extract, loss_key, best_loss_guess, exponent_bounds, override,
                    ))
                for p_type in hp_predictor_types:
                    tasks.append((
                        str(base_path), benchmark_name, opt_name, seed,
                        p_type, "hp",
                        key_to_extract, loss_key, best_loss_guess, exponent_bounds, override,
                    ))

    total = len(tasks)
    print(f"Warming prediction cache: {total} tasks...")
    for done, task in enumerate(tasks, 1):
        msg = _warm_single_cache(task)
        print(f"  [{done}/{total}] {msg}")
    print("Cache warming complete.")


if __name__ == "__main__":
    """
    Plot loss, hypervolume, and rankings for all benchmarks in configs.
    
    Per-seed data is kept throughout calculations, then aggregated with mean/std.
    """
    from pathlib import Path
    
    
    # Try to import benchmark configs, fallback to hardcoded
    from src.benchmarks.configs import BenchmarkEnum
    from src.benchmarks.llm_space import LLM_TARGETS
    from src.benchmarks.openclip_space import OPENCLIP_TARGETS
    from src.benchmarks.tabpfn_space import TABPFN_TARGETS
    from src.plotting.plot_styles import BENCHMARK_TITLES
    from src.extrapolate import load_or_generate_pareto_info
    from src.loss_predictor import LossPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    from src.benchmarks.configs import get_benchmark
    from src.plotting.consts import target_flop_list_per_benchmark, target_flop_multipliers, \
    bench_confs, true_pareto_info_per_benchmark, acc_pred_flop_ranges_per_benchmark_by_ratio,\
        hp_target_flop_ratios, flop_range_to_show_in_pareto_per_benchmark, \
            selected_best_seed_per_benchmark, acc_pred_flop_ranges_per_benchmark, acc_flop_ranges_per_benchmark, \
            all_seeds, results_dir, optimizers_dict, loss_key, target, search_space_per_benchmark, cost_estimator_per_benchmark, params_estimator_per_benchmark
    
    benchmarks = [
        # BenchmarkEnum.OPENCLIP.value,
        # BenchmarkEnum.LLM.value,
        BenchmarkEnum.TABPFN.value,
    ]


    from src.plotting.utils import merge_pareto_plots, merge_split_composite_plots

    # Every benchmark's target_flop_list has the same number of tiers (see
    # consts.py); each tier index picks out one target_flop per benchmark.
    num_tiers = len(target_flop_list_per_benchmark[benchmarks[0]])

    for tier_idx in range(num_tiers):
        predicted_loss_merged = defaultdict(dict)
        avg_predicted_loss_merged = defaultdict(dict)
        predicted_params_merged = defaultdict(dict)
        avg_predicted_params_merged = defaultdict(dict)
        loss_predictor_comparison_merged = defaultdict(dict)
        n_param_predictor_comparison_merged = defaultdict(dict)
        # Per-benchmark (optimizer, predictor) error/rank curves, kept so the
        # 3-panel composites can be built once after the benchmark loop.
        nparam_rankings_per_benchmark: dict[str, Any] = {}
        loss_rankings_per_benchmark: dict[str, Any] = {}
        nparam_ranking_predictor_types: list[Any] = []
        loss_ranking_predictor_types: list[Any] = []

        for benchmark_name in benchmarks:
            target_flop = target_flop_list_per_benchmark[benchmark_name][tier_idx]
            target_subdir = f"{target_flop_multipliers[tier_idx]}x"

            print(f"\n{'='*60}")
            print(f"Generating plots for: {benchmark_name} (target FLOP: {target_flop:.2e}, {target_subdir})")
            print(f"{'='*60}")
            # Downstream figures get their own subtree; they are a different metric
            # than the upstream ones and must not overwrite them.
            plot_output_dir = results_dir / benchmark_name / "plots" / target_subdir
            if target:
                plot_output_dir = plot_output_dir / target
            plot_output_dir.mkdir(parents=True, exist_ok=True)

            benchmark_cfg = bench_confs[benchmark_name]

            # The empirical reference front is measured straight from the benchmark
            # csvs; for a downstream target it only exists if that target is a column
            # of the benchmark's csv(s). Without it, the error-based views
            # (|L - L*|, |log N - log N*| and the rankings built on them) have nothing
            # to measure against, so plot the raw trajectories instead.
            true_pareto_info = true_pareto_info_per_benchmark[benchmark_name]
            has_reference = true_pareto_info is not None
            if not has_reference:
                print(f"  ! No empirical reference front for {benchmark_name}"
                      f"{f' / {target}' if target else ''}: plotting raw predictions and skipping rankings.")

            for predictor_type in [
                LossPredictorTypes.KAPLAN,
                LossPredictorTypes.CHINCHILLA3,
                # LossPredictorTypes.GP_POSTERIOR,
                ]:
                predictor_slug = predictor_type.value.lower().replace(" ", "_")
                print(f"  Plotting {get_predictor_label(predictor_type)} predicted loss at FLOP: {target_flop:.2e} over cumulative FLOPs...")
                selcted_seed = selected_best_seed_per_benchmark[benchmark_name]
                # for across_seeds in [True, False]:
                for across_seeds in [True]:
                    if across_seeds:
                        (fig_pred_all, _)= plot_predicted_loss_over_flops(
                            base_path=results_dir / benchmark_name,
                            optimizers=optimizers_dict,
                            target_flop=target_flop,
                            predictor_type=predictor_type,
                            key_to_extract="cost",
                            loss_key=loss_key,
                            title=f"{BENCHMARK_TITLES[benchmark_name]} ({get_predictor_label(predictor_type)})",
                            flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                            true_pareto_info=true_pareto_info,
                            return_split=False,
                            normalized_flops=False,
                            abs_loss_diff=has_reference,
                            cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                            search_space=search_space_per_benchmark[benchmark_name],
                        )
                        avg_predicted_loss_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_pred_all, _)
                        fig_pred_all.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_all_{target_flop:.2e}.pdf", dpi=300, bbox_inches="tight")
                        plt.close(fig_pred_all)
                        print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_all_{target_flop:.2e}.pdf'}")
                    else:
                        # if predictor_type == LossPredictorTypes.KAPLAN:
                        #     continue
                        for seed in all_seeds:
                            (fig_pred_all, _)= plot_predicted_loss_over_flops(
                                base_path=results_dir / benchmark_name,
                                optimizers={k: [seed] for k in optimizers_dict.keys()},
                                target_flop=target_flop,
                                predictor_type=predictor_type,
                                key_to_extract="cost",
                                loss_key=loss_key,
                                title=f"{BENCHMARK_TITLES[benchmark_name]} ({get_predictor_label(predictor_type)})",
                                flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                                true_pareto_info=true_pareto_info,
                                return_split=False,
                                normalized_flops=False,
                                cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                            search_space=search_space_per_benchmark[benchmark_name],
                            )
                            if seed == selcted_seed:
                                predicted_loss_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_pred_all, _)
                            fig_pred_all.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_all_{seed}_{target_flop:.2e}.pdf", dpi=300, bbox_inches="tight")
                            plt.close(fig_pred_all)
                            print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_all_{seed}_{target_flop:.2e}.pdf'}")
            for predictor_type in [
                ScaleParamPredictorTypes.CHINCHILLA3,
                ScaleParamPredictorTypes.POWER_LAW_FIT,
                # ScaleParamPredictorTypes.GP_POSTERIOR
                ]:
                selcted_seed = selected_best_seed_per_benchmark[benchmark_name]
                predictor_slug = predictor_type.value.lower().replace(" ", "_")
                print(f"  Plotting {get_predictor_label(predictor_type)} predicted n_param at FLOP: {target_flop:.2e} over cumulative FLOPs...")
                # for across_seeds in [True, False]:
                for across_seeds in [True]:
                    if across_seeds:
                        (fig_nparam_all, _) = plot_predicted_n_param_over_flops(
                            base_path=results_dir / benchmark_name,
                            optimizers=optimizers_dict if across_seeds else {k: [seed] for k in optimizers_dict.keys()},
                            target_flop=target_flop,
                            predictor_type=predictor_type,
                            key_to_extract="cost",
                            loss_key=loss_key,
                            title=f"{BENCHMARK_TITLES[benchmark_name]} ({get_predictor_label(predictor_type)})",
                            flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                            true_pareto_info=true_pareto_info,
                            return_split=False,
                            normalized_flops=False,
                            abs_param_diff=has_reference,
                            cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                            search_space=search_space_per_benchmark[benchmark_name],
                            params_estimator=params_estimator_per_benchmark[benchmark_name]
                        )
                        avg_predicted_params_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_nparam_all, _)
                        output_path = plot_output_dir / f"{predictor_slug}_predicted_n_param_all_{target_flop:.2e}.pdf"
                        fig_nparam_all.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close(fig_nparam_all)
                    else:
                        if predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
                            continue
                        for seed in all_seeds:
                            (fig_nparam_all, _) = plot_predicted_n_param_over_flops(
                                base_path=results_dir / benchmark_name,
                                optimizers={k: [seed] for k in optimizers_dict.keys()},
                                target_flop=target_flop,
                                predictor_type=predictor_type,
                                key_to_extract="cost",
                                loss_key=loss_key,
                                title=f"{BENCHMARK_TITLES[benchmark_name]} ({get_predictor_label(predictor_type)})",
                                flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                                true_pareto_info=true_pareto_info,
                                return_split=False,
                                normalized_flops=False,
                                cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                                search_space=search_space_per_benchmark[benchmark_name],
                                params_estimator=params_estimator_per_benchmark[benchmark_name]
                            )
                            if seed == selcted_seed:
                                predicted_params_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_nparam_all, _)
                            output_path = plot_output_dir / f"{predictor_slug}_predicted_n_param_all_{seed}_{target_flop:.2e}.pdf"
                            fig_nparam_all.savefig(output_path, dpi=300, bbox_inches="tight")
                            plt.close(fig_nparam_all)

            # Both ranking families measure |prediction - reference|, so they are
            # skipped entirely when there is no reference front for this target.
            if has_reference:
                nparam_ranking_predictor_types = [
                    ScaleParamPredictorTypes.CHINCHILLA3,
                    ScaleParamPredictorTypes.POWER_LAW_FIT,
                ]
                print(f"  Ranking all optimizers x n_param predictors at FLOP: {target_flop:.2e}...")
                nparam_rankings = calculate_predicted_n_param_error_rankings(
                    base_path=results_dir / benchmark_name,
                    optimizers=optimizers_dict,
                    target_flop=target_flop,
                    predictor_types=nparam_ranking_predictor_types,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                    true_pareto_info=true_pareto_info,
                    cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                    search_space=search_space_per_benchmark[benchmark_name],
                    params_estimator=params_estimator_per_benchmark[benchmark_name],
                )
                print_predictor_error_rankings(
                    nparam_rankings,
                    header=f"  Model-size prediction — {BENCHMARK_TITLES[benchmark_name]} (target FLOP {target_flop:.2e}):",
                )
                nparam_rankings_per_benchmark[benchmark_name] = nparam_rankings

                loss_ranking_predictor_types = [
                    LossPredictorTypes.CHINCHILLA3,
                    LossPredictorTypes.KAPLAN,
                ]
                print(f"  Ranking all optimizers x loss predictors at FLOP: {target_flop:.2e}...")
                loss_rankings = calculate_predicted_loss_error_rankings(
                    base_path=results_dir / benchmark_name,
                    optimizers=optimizers_dict,
                    target_flop=target_flop,
                    predictor_types=loss_ranking_predictor_types,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                    true_pareto_info=true_pareto_info,
                    cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                    search_space=search_space_per_benchmark[benchmark_name],
                )
                print_predictor_error_rankings(
                    loss_rankings,
                    header=f"  Loss prediction — {BENCHMARK_TITLES[benchmark_name]} (target FLOP {target_flop:.2e}):",
                )
                loss_rankings_per_benchmark[benchmark_name] = loss_rankings

            # continue
            model_size_predictors_to_compare =[p for p in ScaleParamPredictorTypes if p != ScaleParamPredictorTypes.GP_POSTERIOR]

            loss_predictors_to_compare = [p for p in LossPredictorTypes if p != LossPredictorTypes.GP_POSTERIOR]

            for optimizer in ["chinchilla_approach1", "carbs", "bo_left_budget_modified_jes"]:
                fig_loss_comp, ax_loss_comp = plot_loss_predictor_comparison_for_optimizer(
                    base_path=results_dir / benchmark_name,
                    optimizer_name=optimizer,
                    seed=all_seeds,
                    target_flop=target_flop,
                    predictor_types=loss_predictors_to_compare,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                    true_pareto_info=true_pareto_info,
                    # title=f"Loss Predictor Comparison (Chinchilla A1) - {BENCHMARK_TITLES[benchmark_name]}",
                )
                loss_predictor_comparison_merged[optimizer][BENCHMARK_TITLES[benchmark_name]] = (fig_loss_comp, ax_loss_comp)

                loss_comp_filename = f"loss_predictor_comparison_{optimizer}_{target_flop:.2e}.pdf"
                fig_loss_comp.savefig(plot_output_dir / loss_comp_filename, dpi=300, bbox_inches="tight")
                plt.close(fig_loss_comp)
                print(f"    ✓ Saved: {plot_output_dir / loss_comp_filename}")

            print(f"  Plotting n_param predictor comparison for Chinchilla A1...")
            for optimizer in ["chinchilla_approach1", "carbs", "bo_left_budget_modified_jes"]:
                fig_comp, ax_comp = plot_n_param_predictor_comparison_for_optimizer(
                    base_path=results_dir / benchmark_name,
                    optimizer_name=optimizer,
                    seed=all_seeds,
                    target_flop=target_flop,
                    predictor_types=model_size_predictors_to_compare,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                    true_pareto_info=true_pareto_info,
                    search_space=search_space_per_benchmark[benchmark_name],
                    cost_estimator=cost_estimator_per_benchmark[benchmark_name],
                    params_estimator=params_estimator_per_benchmark[benchmark_name],
                    # title=f"Predictor Comparison (Chinchilla A1) - {BENCHMARK_TITLES[benchmark_name]}"
                )
                n_param_predictor_comparison_merged[optimizer][BENCHMARK_TITLES[benchmark_name]] = (fig_comp, ax_comp)

                comp_filename = f"n_param_predictor_comparison_{optimizer}_{target_flop:.2e}.pdf"
                fig_comp.savefig(plot_output_dir / comp_filename, dpi=300, bbox_inches="tight")
                plt.close(fig_comp)
                print(f"    ✓ Saved: {plot_output_dir / comp_filename}")

            report_budget = acc_pred_flop_ranges_per_benchmark[benchmark_name][1]

            print(f"  Generating summary table for all predictors/optimizers at budget {report_budget:.2e}...")

        # Composite plots aggregate across benchmarks for this tier; each
        # benchmark contributes its own target_flop (== max_flops * multiplier),
        # so the composite dir is keyed by the multiplier rather than a flop value.
        plots_dir = results_dir / "plots" / f"{target_flop_multipliers[tier_idx]}x"
        if target:
            plots_dir = plots_dir / target
        plots_dir.mkdir(parents=True, exist_ok=True)

        if loss_predictor_comparison_merged:
            for optimizer in loss_predictor_comparison_merged:
                opt_comparision = loss_predictor_comparison_merged[optimizer]
                loss_predictor_comp_master_fig, _ = merge_pareto_plots(opt_comparision)
                loss_predictor_comp_master_fig.savefig(plots_dir / f"loss_predictor_comparison_{optimizer}_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(loss_predictor_comp_master_fig)
                print(f"✓ Saved: {plots_dir / f'loss_predictor_comparison_{optimizer}_composite_all.pdf'}")
        if n_param_predictor_comparison_merged:
            for optimizer in n_param_predictor_comparison_merged:
                opt_comparision = n_param_predictor_comparison_merged[optimizer]
                n_param_predictor_comp_master_fig, _ = merge_pareto_plots(opt_comparision)
                n_param_predictor_comp_master_fig.savefig(plots_dir / f"n_param_predictor_comparison_{optimizer}_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(n_param_predictor_comp_master_fig)
                print(f"✓ Saved: {plots_dir / f'n_param_predictor_comparison_{optimizer}_composite_all.pdf'}")

        # One figure per prediction target: |error| per benchmark, then the
        # across-benchmark relative ranking.
        if nparam_rankings_per_benchmark:
            fig_nparam_comp, _ = plot_prediction_error_composite_across_benchmarks(
                metric="n_param",
                benchmarks=benchmarks,
                optimizers=optimizers_dict,
                predictor_types=nparam_ranking_predictor_types,
                rankings_per_benchmark=nparam_rankings_per_benchmark,
            )
            fig_nparam_comp.savefig(plots_dir / "predicted_n_param_abs_error_composite_all.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig_nparam_comp)
            print(f"✓ Saved: {plots_dir / 'predicted_n_param_abs_error_composite_all.pdf'}")

        if loss_rankings_per_benchmark:
            fig_loss_comp_abs, _ = plot_prediction_error_composite_across_benchmarks(
                metric="loss",
                benchmarks=benchmarks,
                optimizers=optimizers_dict,
                predictor_types=loss_ranking_predictor_types,
                rankings_per_benchmark=loss_rankings_per_benchmark,
                log_y=True,
            )
            fig_loss_comp_abs.savefig(plots_dir / "predicted_loss_abs_error_composite_all.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig_loss_comp_abs)
            print(f"✓ Saved: {plots_dir / 'predicted_loss_abs_error_composite_all.pdf'}")

        if avg_predicted_loss_merged:
            for pred_type in avg_predicted_loss_merged:
                res = avg_predicted_loss_merged[pred_type]
                avg_pred_loss_master_fig, _ = merge_pareto_plots(res)
                avg_pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_avg_predicted_loss_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(avg_pred_loss_master_fig)
                print(f"✓ Saved: {plots_dir / f'{pred_type}_avg_predicted_loss_composite_all.pdf'}")

        if predicted_loss_merged:
            for pred_type in predicted_loss_merged:
                res = predicted_loss_merged[pred_type]
                pred_loss_master_fig, _ = merge_pareto_plots(res)
                pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_predicted_loss_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(pred_loss_master_fig)
                print(f"✓ Saved: {plots_dir / f'{pred_type}_predicted_loss_composite_all.pdf'}")

        if avg_predicted_params_merged:
            for pred_type in avg_predicted_params_merged:
                res = avg_predicted_params_merged[pred_type]
                avg_pred_loss_master_fig, _ = merge_pareto_plots(res)
                avg_pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_avg_predicted_params_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(avg_pred_loss_master_fig)
                print(f"✓ Saved: {plots_dir / f'{pred_type}_avg_predicted_params_composite_all.pdf'}")

        if predicted_params_merged:
            for pred_type in predicted_params_merged:
                res = predicted_params_merged[pred_type]
                pred_loss_master_fig, _ = merge_pareto_plots(res)
                pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_predicted_params_composite_all.pdf", dpi=300, bbox_inches="tight")
                plt.close(pred_loss_master_fig)
            print(f"✓ Saved: {plots_dir / f'{pred_type}_predicted_params_composite_all.pdf'}")

    raise Exception("")

    benchmarks = [
        # BenchmarkEnum.OPENCLIP.value, 
        # BenchmarkEnum.LLM.value,
        BenchmarkEnum.TABPFN.value,
    ]


    print(f"\n{'='*60}")
    print("Generating the predicted loss across all benchmarks...")
    (pred_loss_comp_all, _)= plot_predicted_loss_composite_across_benchmarks(
            base_path=results_dir,
            benchmarks=benchmarks,
            optimizers=optimizers_dict,
            predictor_type=LossPredictorTypes.CHINCHILLA3,
            cost_estimator_per_benchmark=cost_estimator_per_benchmark,
            search_space_per_benchmark=search_space_per_benchmark,
            key_to_extract="cost",
            loss_key="objective_to_minimize",
            target_flop_per_benchmark=target_flop_per_benchmark,
            flop_ranges_per_benchmark=acc_pred_flop_ranges_per_benchmark,
            true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
            abs_loss_diff=True,
        )
    pred_loss_comp_all.savefig(plots_dir / "pred_loss_comp_all.pdf", dpi=300, bbox_inches="tight")
    plt.close(pred_loss_comp_all)
    print(f"✓ Saved: {plots_dir / 'pred_loss_comp_all.pdf'}")


    (pred_n_param_comp_all, _)= plot_predicted_n_param_composite_across_benchmarks(
            base_path=results_dir,
            benchmarks=benchmarks,
            optimizers=optimizers_dict,
            # target_flop=target_flop_per_benchmark,
            predictor_type=ScaleParamPredictorTypes.CHINCHILLA3,
            key_to_extract="cost",
            loss_key="objective_to_minimize",
            # figsize=figsize,
            # title=title,
            target_flop_per_benchmark=target_flop_per_benchmark,
            search_space_per_benchmark=search_space_per_benchmark,
            cost_estimator_per_benchmark=cost_estimator_per_benchmark,
            params_estimator_per_benchmark=params_estimator_per_benchmark,
            flop_ranges_per_benchmark=acc_pred_flop_ranges_per_benchmark,
            true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
            abs_param_diff=True,
        )
    pred_n_param_comp_all.savefig(plots_dir / "pred_n_param_comp_all.pdf", dpi=300, bbox_inches="tight")
    plt.close(pred_n_param_comp_all)
    print(f"✓ Saved: {plots_dir / 'pred_n_param_comp_all.pdf'}")
    # raise Exception("Not ready")


   
