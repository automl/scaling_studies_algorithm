from pathlib import Path
from typing import Any, cast, Callable

import numpy as np
import matplotlib.pyplot as plt
import string
from src.plotting.plot_styles import OPTIMIZER_COLORS



OPTIMIZER_LABELS: dict[str, str] = {
    "random_search": "Random Search",
    "random_search_with_priors": "Random Search (with priors)",
    "chinchilla_approach1": "Chinchilla A1",
    "chinchilla_approach1_with_priors": "Chinchilla A1 (with priors)",
    "chinchilla_approach2": "Chinchilla A2",
    "chinchilla_approach2_with_priors": "Chinchilla A2 (with priors)",
    "carbs": "CARBS",
    "carbs_with_priors": "CARBS (with priors)",
}


def get_optimizer_color(opt_name: str) -> str:
    """Get color for optimizer, with fallback to default."""
    return OPTIMIZER_COLORS.get(opt_name, OPTIMIZER_COLORS["default"])


def get_optimizer_linestyle(opt_name: str) -> str:
    """Get linestyle for optimizer: dashed for _with_priors variants, solid otherwise."""
    return "--" if "_with_priors" in opt_name else "-"


def get_optimizer_label(opt_name: str) -> str:
    """Get display label for optimizer used in plot legends."""
    if opt_name in OPTIMIZER_LABELS:
        return OPTIMIZER_LABELS[opt_name]

    label = opt_name.replace("_with_priors", " (with priors)").replace("_", " ").strip()
    return " ".join(token.upper() if token.lower() == "carbs" else token.capitalize() for token in label.split())


def _split_optimizers_by_priors(
    optimizers: dict[str, list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]]]:
    with_priors = {k: v for k, v in optimizers.items() if "_with_priors" in k}
    without_priors = {k: v for k, v in optimizers.items() if "_with_priors" not in k}
    return dict(optimizers), with_priors, without_priors


def _placeholder_plot(
    *,
    figsize: tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.text(0.5, 0.5, "No optimizers in this group", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    return fig, ax


def _get_global_flop_bounds(flop_grids: list[np.ndarray]) -> tuple[float, float] | None:
    """Compute shared positive FLOP bounds used for log-based normalization."""
    positive = [grid[grid > 1e-8] for grid in flop_grids if len(grid[grid > 1e-8]) > 0]
    if not positive:
        return None
    min_flop = min(float(np.min(vals)) for vals in positive)
    max_flop = max(float(np.max(vals)) for vals in positive)
    return min_flop, max_flop


def _normalize_flop_counts(
    flop_counts: np.ndarray,
    min_flop: float,
    max_flop: float,
) -> np.ndarray:
    """Normalize FLOPs to [0, 1] in log-space for cross-benchmark comparability."""
    positive_mask = flop_counts > 1e-8
    if np.isclose(min_flop, max_flop):
        normalized = np.ones_like(flop_counts, dtype=float) * 0.5
        normalized[~positive_mask] = 0.0
        return normalized

    normalized = np.zeros_like(flop_counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized[positive_mask] = (
            (np.log10(flop_counts[positive_mask]) - np.log10(min_flop))
            / (np.log10(max_flop) - np.log10(min_flop))
        )
    normalized[~positive_mask] = 0.0
    return np.clip(normalized, 0.0, 1.0)


def _actual_costs_from_cumulative(cumulative_costs: np.ndarray) -> np.ndarray:
    """Recover per-evaluation costs from a cumulative FLOP trajectory."""
    if len(cumulative_costs) == 0:
        return np.array([], dtype=float)
    return np.diff(np.concatenate(([0.0], cumulative_costs.astype(float))))


def _get_hypervolume_bounds(
    raw_seed_data: dict[str, dict[str, Any]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Compute shared loss and per-trial log-FLOP bounds for hypervolume."""
    all_losses: list[float] = []
    all_log_costs: list[float] = []

    for data in raw_seed_data.values():
        for losses in data["per_seed_losses"].values():
            losses_arr = np.asarray(losses, dtype=float)
            all_losses.extend(losses_arr[np.isfinite(losses_arr)].tolist())

        for costs in data["per_seed_costs"].values():
            actual_costs = _actual_costs_from_cumulative(np.asarray(costs, dtype=float))
            valid_costs = actual_costs[np.isfinite(actual_costs) & (actual_costs > 0)]
            all_log_costs.extend(np.log10(np.maximum(1.0, valid_costs)).tolist())

    loss_bounds = (min(all_losses), max(all_losses)) if all_losses else None
    flops_bounds = (min(all_log_costs), max(all_log_costs)) if all_log_costs else None
    return loss_bounds, flops_bounds


def _get_common_xlim(
    x_series: list[np.ndarray],
    default: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return shared x-range intersection across non-empty curves."""
    if not x_series:
        return default

    starts: list[float] = []
    ends: list[float] = []
    for xs in x_series:
        if len(xs) == 0:
            continue
        starts.append(float(xs[0]))
        ends.append(float(xs[-1]))

    if not starts or not ends:
        return default

    left = max(starts)
    right = min(ends)
    if left >= right:
        return default
    return (left, right)


def _replot_lines(source_ax: Any, target_ax: Any) -> None:
    """Re-plot lines and fill_between bands from source axis to target axis."""
    for line in source_ax.get_lines():
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        is_axes_width_hline = (
            len(x_data) == 2
            and len(y_data) == 2
            and np.allclose(x_data, [0.0, 1.0])
            and np.isclose(y_data[0], y_data[1])
        )

        if is_axes_width_hline:
            target_ax.axhline(
                y=float(y_data[0]),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                alpha=line.get_alpha(),
            )
            continue

        target_ax.plot(
            x_data,
            y_data,
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            alpha=line.get_alpha(),
        )

    # Recreate confidence bands from PolyCollection paths generated by fill_between.
    for collection in source_ax.collections:
        paths = collection.get_paths()
        facecolors = collection.get_facecolors()
        if len(paths) == 0 or len(facecolors) == 0:
            continue

        band_color = tuple(facecolors[0])
        for path in paths:
            verts = np.asarray(path.vertices, dtype=float)
            if verts.shape[0] < 3:
                continue
            target_ax.fill(
                verts[:, 0],
                verts[:, 1],
                facecolor=band_color,
                edgecolor="none",
            )

