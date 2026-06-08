from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Callable
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import string
from dataclasses import asdict, is_dataclass
import neps
from neps.state.trial import State

from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit
from src.benchmarks.configs import get_benchmark
from src.plotting.utils import OPTIMIZER_LABELS, get_optimizer_color,\
    get_optimizer_linestyle, get_optimizer_label, _split_optimizers_by_priors, \
    _placeholder_plot, _composite_empty_placeholder, _get_global_flop_bounds, \
    _normalize_flop_counts, _actual_costs_from_cumulative, _get_hypervolume_bounds, \
    _get_common_xlim, _replot_lines, split_by_priors


from src.plotting.standalone_plots import (
    _get_best_loss_guess,
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



































def _get_prediction_cache_path(
    base_path: Path,
    opt_name: str,
    seed: int | str,
    predictor_slug: str,
    target_flop: float,
    kind: str,
) -> Path:
    safe_slug = predictor_slug.replace(" ", "_").replace("/", "_")
    return base_path / opt_name / f"seed_{seed}" / "prediction_cache" / f"{kind}_{safe_slug}_{target_flop:.2e}.npz"


def _load_or_compute_trajectory(
    cache_path: Path,
    n_trials: int,
    compute_fn,
) -> np.ndarray:
    """Return cached per-trial predictions or run compute_fn() and cache the result."""
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if int(data["n_trials"]) == n_trials:
                return data["pred_values"]
        except Exception:
            pass

    pred_values = compute_fn()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, pred_values=pred_values, n_trials=np.array(n_trials))
    except Exception as e:
        print(f"Warning: could not save prediction cache to {cache_path}: {e}")
    return pred_values


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
) -> Any:
    """Plot predicted loss over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    from src.loss_predictor import LossPredictorTypes, predict_loss

    if predictor_type is None:
        predictor_type = LossPredictorTypes.KAPLAN

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name)
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
            _loss_cache_path = _get_prediction_cache_path(base_path, opt_name, seed, predictor_slug, target_flop, "loss")
            _rows_snap = rows

            def _compute_loss_preds():
                _preds: list[float] = []
                for _n in range(1, len(_rows_snap) + 1):
                    try:
                        _loss, _info = predict_loss(
                            predictor_type=predictor_type,
                            target_flop=target_flop,
                            rows=_rows_snap[:_n],
                            best_loss_guess=best_loss_guess,
                        )
                        if _n == len(_rows_snap):
                            _fpath = base_path / "plots" / predictor_slug / opt_name / f"seed_{seed}" / "final_prediction_info.json"
                            try:
                                _fpath.parent.mkdir(parents=True, exist_ok=True)
                                with open(_fpath, "w+") as _f:
                                    json.dump(asdict(_info), _f, indent=2)
                                print(f"Saved final prediction info for {opt_name} seed {seed} to {_fpath}")
                            except Exception as _e:
                                print(f"Warning: could not save final prediction info for {opt_name} seed {seed}: {_e}")
                        _preds.append(float(_loss))
                    except Exception:
                        _preds.append(np.nan)
                return np.asarray(_preds, dtype=float)

            x = np.asarray(cumulative_flops, dtype=float)
            y = _load_or_compute_trajectory(_loss_cache_path, len(rows), _compute_loss_preds)
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
    from src.loss_predictor import LossPredictorTypes, predict_loss

    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    if predictor_types is None:
        predictor_types = list(LossPredictorTypes)

    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name)
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
            pred_values: list[float] = []

            for n in range(1, len(rows) + 1):
                try:
                    pred_loss, _ = predict_loss(
                        predictor_type=p_type,
                        target_flop=target_flop,
                        rows=rows[:n],
                        best_loss_guess=best_loss_guess,
                    )
                    pred_values.append(float(pred_loss))
                except Exception:
                    pred_values.append(np.nan)

            x = np.asarray(cumulative_flops, dtype=float)
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

        label = p_type.value if hasattr(p_type, "value") else str(p_type)
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
) -> tuple[Any, Any]:
    """
    Plots multiple scaling law predictors for one optimizer's trajectory.
    Useful for comparing how different predictors (Kaplan vs Chinchilla vs Power Law) 
    converge toward the truth using the same data. Multiple seeds are aggregated
    as mean with standard-error bands.
    """
    from src.scale_param_predictor import predict_scale_parameters
    
    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name)
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
            pred_values = []
            
            # We simulate the optimizer's progress trial-by-trial
            for n in range(1, len(rows) + 1):
                try:
                    # Use learning curves if predictor supports them/needs them
                    # Some predictors (like Chinchilla Approach 1) prefer LC rows
                    source = learning_curve_rows[:n] if len(learning_curve_rows) > 0 else rows[:n]
                    
                    (pred_n, _), _ = predict_scale_parameters(
                        predictor_type=p_type,
                        target_flop=target_flop,
                        rows=source,
                        best_loss_guess=best_loss_guess,
                    )
                    pred_values.append(float(pred_n))
                except Exception:
                    pred_values.append(np.nan)

            x = np.asarray(cumulative_flops, dtype=float)
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

        label = p_type.value if hasattr(p_type, 'value') else str(p_type)
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
    ax.set_ylabel(r"$N$", fontsize=20)
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
) -> Any:
    """Plot predicted n_param over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    from src.scale_param_predictor import ScaleParamPredictorTypes, predict_scale_parameters

    if predictor_type is None:
        predictor_type = ScaleParamPredictorTypes.POWER_LAW_FIT

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)
    best_loss_guess = _get_best_loss_guess(base_path.name)

    all_flop_arrays: list[np.ndarray] = []
    all_data_per_opt: dict[str, list[dict[str, np.ndarray]]] = {}
    final_prediction_rows_by_opt: dict[str, list[dict[str, float]]] = {}
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
            _nparam_cache_path = _get_prediction_cache_path(base_path, opt_name, seed, _nparam_predictor_slug, target_flop, "nparam")
            _source_rows_snap = source_rows

            def _compute_nparam_preds():
                _preds: list[float] = []
                for _n in range(1, total_trials + 1):
                    try:
                        (_n_param, _), _ = predict_scale_parameters(
                            predictor_type=predictor_type,
                            target_flop=target_flop,
                            rows=_source_rows_snap[:_n],
                            best_loss_guess=best_loss_guess,
                        )
                        _preds.append(float(_n_param))
                    except Exception:
                        _preds.append(np.nan)
                return np.asarray(_preds, dtype=float)

            pred_n_params_arr = _load_or_compute_trajectory(_nparam_cache_path, total_trials, _compute_nparam_preds)

            try:
                (final_pred_n_param, final_pred_n_data), final_fit_info = predict_scale_parameters(
                    predictor_type=predictor_type,
                    target_flop=target_flop,
                    rows=source_rows[:total_trials],
                    best_loss_guess=best_loss_guess,
                )
                final_info_values: dict[str, float] = {}
                if is_dataclass(final_fit_info):
                    final_info_values = {
                        str(k): float(v)
                        for k, v in asdict(final_fit_info).items()
                        if isinstance(v, (int, float, np.integer, np.floating))
                    }
                final_prediction_rows_by_opt.setdefault(opt_name, []).append({
                    "spent_over_target": float(cumulative_flops[total_trials - 1]) / float(target_flop),
                    "pred_n_param": float(final_pred_n_param),
                    "pred_n_data": float(final_pred_n_data),
                    **final_info_values,
                })
            except Exception:
                pass

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
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted Loss Composite Across Benchmarks",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
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
                error_curve = np.abs(pred_ys - true_val)
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
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (5.5, 3.5),
    title: str = "Predicted n_param Composite Across Benchmarks",
    flop_ranges_per_benchmark: dict[str, tuple[float, float]] | None = None,
    y_clip_quantile: float | None = 0.99,
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
                    error_curve = np.abs(pred_ys - true_val)
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
    from src.plotting.consts import target_flop_per_benchmark, \
    bench_confs, true_pareto_info_per_benchmark, acc_pred_flop_ranges_per_benchmark_by_ratio,\
        hp_target_flop_ratios, target_flop_per_benchmark, flop_range_to_show_in_pareto_per_benchmark, \
            selected_best_seed_per_benchmark, acc_pred_flop_ranges_per_benchmark, acc_flop_ranges_per_benchmark, \
            all_seeds, results_dir, optimizers_dict, loss_key
    
    benchmarks = [
        BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value,
        # BenchmarkEnum.TABPFN.value,
    ]
    predicted_loss_merged = defaultdict(dict)
    avg_predicted_loss_merged = defaultdict(dict)
    predicted_params_merged = defaultdict(dict)
    avg_predicted_params_merged = defaultdict(dict)
    loss_predictor_comparison_merged = defaultdict(dict)
    n_param_predictor_comparison_merged = defaultdict(dict)


    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {benchmark_name}")
        print(f"{'='*60}")
        # true_pareto_info = load_or_generate_pareto_info(benchmark_name)
        targets = {
            BenchmarkEnum.LLM.value: LLM_TARGETS, 
            BenchmarkEnum.OPENCLIP.value: OPENCLIP_TARGETS,
            BenchmarkEnum.TABPFN.value: TABPFN_TARGETS,
        }
        # for target in targets[benchmark_name]:
        #     try:
        #         target = target.value
        #     except AttributeError:
        #         pass
        #     loss_key = f"extra.predictions.{target}.mean"
            # Create output directory
        plot_output_dir = results_dir / benchmark_name / "plots"
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        
        benchmark_cfg = bench_confs[benchmark_name]
        
        

        target_flop = target_flop_per_benchmark[benchmark_name]

        for predictor_type in [
            LossPredictorTypes.KAPLAN, 
            LossPredictorTypes.CHINCHILLA3,
            ]:
            predictor_slug = predictor_type.value.lower().replace(" ", "_")
            print(f"  Plotting {predictor_type.value} predicted loss at FLOP: {target_flop_per_benchmark[benchmark_name]:.2e} over cumulative FLOPs...")
            selcted_seed = selected_best_seed_per_benchmark[benchmark_name]
            for across_seeds in [True, False]:
                if across_seeds:
                    (fig_pred_all, _)= plot_predicted_loss_over_flops(
                        base_path=results_dir / benchmark_name,
                        optimizers=optimizers_dict,
                        target_flop=target_flop_per_benchmark[benchmark_name],
                        predictor_type=predictor_type,
                        key_to_extract="cost",
                        loss_key=loss_key,
                        title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
                        flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                        true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                        return_split=False,
                        normalized_flops=False,
                        abs_loss_diff=True,
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
                            target_flop=target_flop_per_benchmark[benchmark_name],
                            predictor_type=predictor_type,
                            key_to_extract="cost",
                            loss_key=loss_key,
                            title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
                            flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                            true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                            return_split=False,
                            normalized_flops=False,
                        )
                        if seed == selcted_seed:
                            predicted_loss_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_pred_all, _)
                        fig_pred_all.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_all_{seed}_{target_flop:.2e}.pdf", dpi=300, bbox_inches="tight")
                        plt.close(fig_pred_all)
                        print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_all_{seed}_{target_flop:.2e}.pdf'}")
        for predictor_type in [
            ScaleParamPredictorTypes.CHINCHILLA3, 
            ScaleParamPredictorTypes.POWER_LAW_FIT
            ]:
            selcted_seed = selected_best_seed_per_benchmark[benchmark_name]
            predictor_slug = predictor_type.value.lower().replace(" ", "_")
            print(f"  Plotting {predictor_type.value} predicted n_param at FLOP: {target_flop_per_benchmark[benchmark_name]:.2e} over cumulative FLOPs...")
            for across_seeds in [True, False]:
                if across_seeds:
                    (fig_nparam_all, _) = plot_predicted_n_param_over_flops(
                        base_path=results_dir / benchmark_name,
                        optimizers=optimizers_dict if across_seeds else {k: [seed] for k in optimizers_dict.keys()},
                        target_flop=target_flop,
                        predictor_type=predictor_type,
                        key_to_extract="cost",
                        loss_key=loss_key,
                        title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
                        flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                        true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                        return_split=False,
                        normalized_flops=False,
                        abs_param_diff=True,
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
                            title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
                            flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                            true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                            return_split=False,
                            normalized_flops=False,
                        )
                        if seed == selcted_seed:
                            predicted_params_merged[predictor_type.value][BENCHMARK_TITLES[benchmark_name]] = (fig_nparam_all, _)
                        output_path = plot_output_dir / f"{predictor_slug}_predicted_n_param_all_{seed}_{target_flop:.2e}.pdf"
                        fig_nparam_all.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close(fig_nparam_all)


        model_size_predictors_to_compare = list(ScaleParamPredictorTypes)
        target_flop = target_flop_per_benchmark[benchmark_name]

        loss_predictors_to_compare = list(LossPredictorTypes)

        # for optimizer in ["chinchilla_approach1", "carbs"]:
        #     fig_loss_comp, ax_loss_comp = plot_loss_predictor_comparison_for_optimizer(
        #         base_path=results_dir / benchmark_name,
        #         optimizer_name=optimizer,
        #         seed=all_seeds,
        #         target_flop=target_flop,
        #         predictor_types=loss_predictors_to_compare,
        #         key_to_extract="cost",
        #         loss_key=loss_key,
        #         flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
        #         true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
        #         # title=f"Loss Predictor Comparison (Chinchilla A1) - {BENCHMARK_TITLES[benchmark_name]}",
        #     )
        #     loss_predictor_comparison_merged[optimizer][BENCHMARK_TITLES[benchmark_name]] = (fig_loss_comp, ax_loss_comp)

        #     loss_comp_filename = f"loss_predictor_comparison_{optimizer}_{target_flop:.2e}.pdf"
        #     fig_loss_comp.savefig(plot_output_dir / loss_comp_filename, dpi=300, bbox_inches="tight")
        #     plt.close(fig_loss_comp)
        #     print(f"    ✓ Saved: {plot_output_dir / loss_comp_filename}")

        print(f"  Plotting n_param predictor comparison for Chinchilla A1...")
        for optimizer in ["chinchilla_approach1", "carbs"]:
            fig_comp, ax_comp = plot_n_param_predictor_comparison_for_optimizer(
                base_path=results_dir / benchmark_name,
                optimizer_name=optimizer,
                seed=all_seeds,
                target_flop=target_flop,
                predictor_types=model_size_predictors_to_compare,
                key_to_extract="cost",
                loss_key=loss_key,
                flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
                true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
                # title=f"Predictor Comparison (Chinchilla A1) - {BENCHMARK_TITLES[benchmark_name]}"
            )
            n_param_predictor_comparison_merged[optimizer][BENCHMARK_TITLES[benchmark_name]] = (fig_comp, ax_comp)

            comp_filename = f"n_param_predictor_comparison_{optimizer}_{target_flop:.2e}.pdf"
            fig_comp.savefig(plot_output_dir / comp_filename, dpi=300, bbox_inches="tight")
            plt.close(fig_comp)
            print(f"    ✓ Saved: {plot_output_dir / comp_filename}")

        report_budget = acc_pred_flop_ranges_per_benchmark[benchmark_name][1] 

        print(f"  Generating summary table for all predictors/optimizers at budget {report_budget:.2e}...")

      
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)


    from src.plotting.utils import merge_pareto_plots, merge_split_composite_plots

    

    # for optimizer in loss_predictor_comparison_merged:
    #     opt_comparision = loss_predictor_comparison_merged[optimizer]
    #     loss_predictor_comp_master_fig, _ = merge_pareto_plots(opt_comparision)
    #     loss_predictor_comp_master_fig.savefig(plots_dir / f"loss_predictor_comparison_{optimizer}_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(loss_predictor_comp_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'loss_predictor_comparison_{optimizer}_composite_all.pdf'}")

    # for optimizer in n_param_predictor_comparison_merged:
    #     opt_comparision = n_param_predictor_comparison_merged[optimizer]
    #     n_param_predictor_comp_master_fig, _ = merge_pareto_plots(opt_comparision)
    #     n_param_predictor_comp_master_fig.savefig(plots_dir / f"n_param_predictor_comparison_{optimizer}_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(n_param_predictor_comp_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'n_param_predictor_comparison_{optimizer}_composite_all.pdf'}")

    # for pred_type in avg_predicted_loss_merged:
    #     res = avg_predicted_loss_merged[pred_type]
    #     avg_pred_loss_master_fig, _ = merge_pareto_plots(res)
    #     avg_pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_avg_predicted_loss_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(avg_pred_loss_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'{pred_type}_avg_predicted_loss_composite_all.pdf'}")

    # for pred_type in predicted_loss_merged:
    #     res = predicted_loss_merged[pred_type]
    #     pred_loss_master_fig, _ = merge_pareto_plots(res)
    #     pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_predicted_loss_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(pred_loss_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'{pred_type}_predicted_loss_composite_all.pdf'}")

    # for pred_type in avg_predicted_params_merged:
    #     res = avg_predicted_params_merged[pred_type]
    #     avg_pred_loss_master_fig, _ = merge_pareto_plots(res)
    #     avg_pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_avg_predicted_params_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(avg_pred_loss_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'{pred_type}_avg_predicted_params_composite_all.pdf'}")

    # for pred_type in predicted_params_merged:
    #     res = predicted_params_merged[pred_type]
    #     pred_loss_master_fig, _ = merge_pareto_plots(res)
    #     pred_loss_master_fig.savefig(plots_dir / f"{pred_type}_predicted_params_composite_all.pdf", dpi=300, bbox_inches="tight")
    #     plt.close(pred_loss_master_fig)
    #     print(f"✓ Saved: {plots_dir / f'{pred_type}_predicted_params_composite_all.pdf'}")

    # raise Exception("")
    

    benchmarks = [
        BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value,
        # BenchmarkEnum.TABPFN.value,
    ]
    optimizers_dict = {
        "random_search": all_seeds,
        "random_search_with_priors": all_seeds,
        "chinchilla_approach1": all_seeds,
        "chinchilla_approach1_with_priors": all_seeds,
        "chinchilla_approach2": all_seeds,
        "chinchilla_approach2_with_priors": all_seeds,
        "carbs": all_seeds,
        "carbs_with_priors": all_seeds,
    }

    # print(f"\n{'='*60}")
    # print("Generating the predicted loss across all benchmarks...")
    # (pred_loss_comp_all, _)= plot_predicted_loss_composite_across_benchmarks(
    #         base_path=results_dir,
    #         benchmarks=benchmarks,
    #         optimizers=optimizers_dict,
    #         # target_flop=target_flop_per_benchmark,
    #         predictor_type=LossPredictorTypes.CHINCHILLA3,
    #         key_to_extract="cost",
    #         loss_key="objective_to_minimize",
    #         # figsize=figsize,
    #         # title=title,
    #         target_flop_per_benchmark=target_flop_per_benchmark,
    #         flop_ranges_per_benchmark=acc_pred_flop_ranges_per_benchmark,
    #         true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
    #         return_split=False,
    #     )
    # pred_loss_comp_all.savefig(plots_dir / "pred_loss_comp_all.pdf", dpi=300, bbox_inches="tight")
    # plt.close(pred_loss_comp_all)
    # print(f"✓ Saved: {plots_dir / 'pred_loss_comp_all.pdf'}")


    # (pred_n_param_comp_all, _)= plot_predicted_n_param_composite_across_benchmarks(
    #         base_path=results_dir,
    #         benchmarks=benchmarks,
    #         optimizers=optimizers_dict,
    #         # target_flop=target_flop_per_benchmark,
    #         predictor_type=ScaleParamPredictorTypes.CHINCHILLA3,
    #         key_to_extract="cost",
    #         loss_key="objective_to_minimize",
    #         # figsize=figsize,
    #         # title=title,
    #         target_flop_per_benchmark=target_flop_per_benchmark,
    #         flop_ranges_per_benchmark=acc_pred_flop_ranges_per_benchmark,
    #         true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
    #         return_split=False,
    #     )
    # pred_n_param_comp_all.savefig(plots_dir / "pred_n_param_comp_all.pdf", dpi=300, bbox_inches="tight")
    # plt.close(pred_n_param_comp_all)
    # print(f"✓ Saved: {plots_dir / 'pred_n_param_comp_all.pdf'}")
    # raise Exception("Not ready")


   
