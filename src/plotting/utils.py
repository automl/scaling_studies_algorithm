from pathlib import Path
import functools
import inspect
import re
from typing import Any, cast, Callable

import numpy as np
import matplotlib.pyplot as plt
import string
from src.plotting.plot_styles import OPTIMIZER_COLORS



OPTIMIZER_LABELS: dict[str, str] = {
    "random_search": "RS",
    "chinchilla_approach1": "Chinchilla A1",
    "chinchilla_approach2": "Chinchilla A2",
    "chinchilla_gemstones": "Chinchilla Gemstones",
    "carbs": "CARBS",
    "bo_space_expansion_mes": "BO Space Expansion (MES)",
    "bo_space_expansion_ei": "BO Space Expansion (EI)",
    "bo_space_expansion_ei_cost_aware": "BO Space Expansion (EI, Cost-Aware)",
    "bo_space_expansion_jes": "BO Space Expansion (JES)",
    "bo_left_budget_zero_mean_ei": "BO Left Budget (zero-mean EI)",
    "bo_left_budget": "BO Left Budget",
    "bo_left_budget_mes": "BO Left Budget (MES)",
    "bo_left_budget_ei": "BO Left Budget (EI)",
    "bo_left_budget_jes": "BO Left Budget (JES)",
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
    # ax.set_title(title, fontsize=14)
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



import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.patches import Polygon


def _flatten_axes(axes: Any) -> list[Any]:
    axes_array = np.asarray(axes, dtype=object)
    return axes_array.flatten().tolist()


def _strip_panel_prefix(title: str) -> str:
    return re.sub(r"^\([a-z]\)\s*", "", title).strip()


def _copy_axis_artists(source_ax: Any, target_ax: Any) -> None:
    """Copy the visible plotted artists from one Matplotlib axis to another."""
    for line in source_ax.get_lines():
        transform = line.get_transform()

        if transform == source_ax.get_yaxis_transform():
            target_ax.axhline(
                y=line.get_ydata()[0],
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                alpha=line.get_alpha(),
                zorder=line.get_zorder(),
            )
        elif transform == source_ax.get_xaxis_transform():
            target_ax.axvline(
                x=line.get_xdata()[0],
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                alpha=line.get_alpha(),
                zorder=line.get_zorder(),
            )
        else:
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                alpha=line.get_alpha(),
                zorder=line.get_zorder(),
            )

    for collection in source_ax.collections:
        if isinstance(collection, PathCollection):
            offsets = collection.get_offsets()
            if len(offsets) == 0:
                continue

            facecolors = collection.get_facecolors()
            edgecolors = collection.get_edgecolors()
            sizes = collection.get_sizes()
            linewidths = collection.get_linewidths()
            paths = collection.get_paths()
            target_ax.scatter(
                offsets[:, 0],
                offsets[:, 1],
                label=collection.get_label(),
                c=facecolors if len(facecolors) else None,
                edgecolors=edgecolors if len(edgecolors) else None,
                s=sizes if len(sizes) else None,
                marker=paths[0] if paths else None,
                linewidths=linewidths if len(linewidths) else None,
                alpha=collection.get_alpha(),
                zorder=collection.get_zorder(),
            )
        elif isinstance(collection, PolyCollection):
            verts = [path.vertices for path in collection.get_paths()]
            if not verts:
                continue

            target_ax.add_collection(PolyCollection(
                verts,
                facecolors=collection.get_facecolor(),
                edgecolors=collection.get_edgecolor(),
                alpha=collection.get_alpha(),
                zorder=collection.get_zorder(),
                label=collection.get_label(),
            ))

    for patch in source_ax.patches:
        if not isinstance(patch, Polygon):
            continue

        target_ax.add_patch(Polygon(
            patch.get_xy(),
            closed=patch.get_closed(),
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            linewidth=patch.get_linewidth(),
            linestyle=patch.get_linestyle(),
            alpha=patch.get_alpha(),
            zorder=patch.get_zorder(),
            label=patch.get_label(),
        ))


def _copy_axis_format(source_ax: Any, target_ax: Any, *, show_title: bool = True) -> None:
    target_ax.set_xscale(source_ax.get_xscale())
    target_ax.set_yscale(source_ax.get_yscale())
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    title = _strip_panel_prefix(source_ax.get_title(loc="left")) if show_title else ""
    target_ax.set_title(title, loc="left", fontsize=20)
    target_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def _dedupe_legend(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    clean_handles: list[Any] = []
    clean_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in clean_labels:
            continue
        clean_handles.append(handle)
        clean_labels.append(label)
    return clean_handles, clean_labels


def merge_split_composite_plots(
    split_results: list[tuple[Any, Any]] | tuple[tuple[Any, Any], ...],
    *,
    group_titles: list[str] | tuple[str, ...] = ("All", "With priors", "Without priors"),
    figsize: tuple[float, float] | None = None,
    close_sources: bool = False,
    n_per_row: int = 3,
    legend_groups: list[int] | None = None,
    panel_labels: bool = False,
) -> tuple[Any, np.ndarray]:
    """
    Combine split composite plots into one figure with a shared legend.

    Each input result is expected to be a ``(fig, axes)`` pair, such as the
    all/with-priors/without-priors outputs returned by the composite plotting
    functions. Axes are arranged in horizontal rows with n_per_row axes per row.

    By default the shared legend is built from every group's axes. Pass
    ``legend_groups`` (indices into ``split_results``) to pull the legend from
    only a subset of groups, e.g. ``legend_groups=[0]`` for the first group only.

    By default each first-row axis keeps its source benchmark name as its title.
    Pass ``panel_labels=True`` to replace every axis's title with a sequential
    "(a)", "(b)", ... panel label instead.
    """
    if not split_results:
        raise ValueError("split_results must contain at least one (fig, axes) pair")

    axes_per_group = [_flatten_axes(axes) for _fig, axes in split_results]

    # Flatten all axes from all groups into a single list, tracking each
    # axis's source group so the legend can optionally be restricted.
    all_axes_flat = []
    axis_group_idx = []
    for group_idx, source_axes in enumerate(axes_per_group):
        all_axes_flat.extend(source_axes)
        axis_group_idx.extend([group_idx] * len(source_axes))

    total_axes = len(all_axes_flat)
    num_cols = n_per_row
    num_rows = (total_axes + num_cols - 1) // num_cols  # Ceiling division
    
    if figsize is None:
        figsize = (6 * num_cols, 3.8 * num_rows)

    master_fig, master_axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    shared_handles: list[Any] = []
    shared_labels: list[str] = []

    # Fill grid left-to-right, top-to-bottom with 3 per row
    axis_idx = 0
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            target_ax = master_axes[row_idx, col_idx]
            
            if axis_idx >= len(all_axes_flat):
                target_ax.axis("off")
                continue
            
            source_ax = all_axes_flat[axis_idx]
            _copy_axis_artists(source_ax, target_ax)
            
            # Copy format without title, we'll handle titles manually
            _copy_axis_format(source_ax, target_ax, show_title=False)
            
            if panel_labels:
                letter = string.ascii_lowercase[axis_idx % len(string.ascii_lowercase)]
                target_ax.set_title(f"({letter})", loc="left", fontsize=20)
            elif row_idx == 0:
                # Set title only in the first row, centered
                title = _strip_panel_prefix(source_ax.get_title(loc="left"))
                target_ax.set_title(title, loc="center", fontsize=20)
            
            # Keep x-axis labels only in the last row
            if row_idx != num_rows - 1:
                target_ax.set_xlabel("")
            
            handles, labels = target_ax.get_legend_handles_labels()
            if legend_groups is None or axis_group_idx[axis_idx] in legend_groups:
                shared_handles.extend(handles)
                shared_labels.extend(labels)

            if target_ax.get_legend() is not None:
                target_ax.get_legend().remove()
            
            axis_idx += 1

    # Close source figures if requested
    for source_fig, _source_axes in split_results:
        if close_sources:
            plt.close(source_fig)

    clean_handles, clean_labels = _dedupe_legend(shared_handles, shared_labels)
    if clean_handles:
        master_fig.legend(
            clean_handles,
            clean_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(4, len(clean_labels)),
            fontsize=20,
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return master_fig, master_axes


def _composite_empty_placeholder(kw: dict[str, Any]) -> tuple[Any, Any]:
    """Placeholder figure for composite multi-panel plots when a group is empty."""
    benchmarks = kw.get("benchmarks", [])
    cols = max(len(benchmarks) + 1, 1)
    fig, ax = plt.subplots(1, 1, figsize=(6 * cols, 4))
    ax.text(0.5, 0.5, "No optimizers in this group",
            ha="center", va="center", transform=ax.transAxes)
    return fig, ax


def split_by_priors(
    get_placeholder: Callable[[dict[str, Any]], tuple[Any, Any]],
) -> Callable:
    """
    Decorator for plot functions that produce separate figures for
    'all', 'with priors', and 'without priors' optimizer groups.

    The decorated function must accept keyword-only arguments and implement
    only the single-group (return_split=False) case. This decorator intercepts
    return_split=True, calls the function three times with the appropriate
    optimizer subsets, and uses get_placeholder when a group is empty.

    Args:
        get_placeholder: callable(kwargs) -> (fig, axes) used when a group
            has no optimizers. Receives the full kwargs dict with 'title'
            already overridden to the group-specific title.
    """
    def decorator(func: Callable) -> Callable:
        _has_title = "title" in inspect.signature(func).parameters

        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            if not kwargs.pop("return_split", True):
                return func(**kwargs)

            optimizers = kwargs["optimizers"]
            title = kwargs.get("title", "")
            all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

            base = {**kwargs}  # return_split already popped above
            result_all = func(**{**base, "optimizers": all_opts})

            def _run_or_placeholder(group: dict, suffix: str) -> Any:
                group_title = f"{title} {suffix}".strip()
                if group:
                    call_kw = {**base, "optimizers": group}
                    if _has_title:
                        call_kw["title"] = group_title
                    return func(**call_kw)
                return get_placeholder({**kwargs, "title": group_title})

            result_with = _run_or_placeholder(with_priors, "(with priors)")
            result_without = _run_or_placeholder(without_priors, "(without priors)")
            return result_all, result_with, result_without

        return wrapper
    return decorator


def merge_pareto_plots(benchmark_dict):
    """
    Takes a dictionary of {"Benchmark Name": (fig, ax)} and merges them 
    into a single side-by-side NeurIPS-formatted figure.
    """
    num_plots = len(benchmark_dict)
    
    master_fig, master_axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    
    # Ensure master_axes is iterable even if there's only one plot
    if num_plots == 1:
        master_axes = [master_axes]

    for i, (benchmark_name, (old_fig, old_ax)) in enumerate(benchmark_dict.items()):
        new_ax = master_axes[i]

        # 1. Extract and copy standard lines AND axhline/axvline
        for line in old_ax.get_lines():
            transform = line.get_transform()
            
            # Horizontal lines
            if transform == old_ax.get_yaxis_transform():
                new_ax.axhline(y=line.get_ydata()[0],
                               label=line.get_label(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               linewidth=line.get_linewidth(),
                               zorder=line.get_zorder())
                
            # Vertical lines
            elif transform == old_ax.get_xaxis_transform():
                new_ax.axvline(x=line.get_xdata()[0],
                               label=line.get_label(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               linewidth=line.get_linewidth(),
                               zorder=line.get_zorder())
                
            # Standard plotted lines
            else:
                new_ax.plot(line.get_xdata(), line.get_ydata(),
                            label=line.get_label(),
                            color=line.get_color(),
                            linestyle=line.get_linestyle(),
                            linewidth=line.get_linewidth(),
                            marker=line.get_marker(),
                            zorder=line.get_zorder())

        # 2. Extract Collections (Scatter points AND fill_between shading)
        for collection in old_ax.collections:
            
            # Handle Scatter Points
            if isinstance(collection, PathCollection):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    paths = collection.get_paths()
                    linewidths = collection.get_linewidths()
                    edgecolors = collection.get_edgecolors()
                    new_ax.scatter(offsets[:, 0], offsets[:, 1],
                                   label=collection.get_label(),
                                   color=collection.get_facecolors()[0] if len(collection.get_facecolors()) else None,
                                   edgecolors=edgecolors if len(edgecolors) else None,
                                   alpha=collection.get_alpha(),
                                   s=collection.get_sizes()[0] if len(collection.get_sizes()) else None,
                                   marker=paths[0] if paths else None,
                                   linewidths=linewidths[0] if len(linewidths) else None,
                                   zorder=collection.get_zorder())
            
            # Handle fill_between (Shaded confidence intervals)
            # Handle fill_between (Shaded confidence intervals)
            elif isinstance(collection, PolyCollection):
                
                # EXTRACT THE VERTICES HERE:
                # Get the raw (N, 2) coordinate arrays from the path objects
                verts = [path.vertices for path in collection.get_paths()]
                
                new_poly = PolyCollection(
                    verts,  # <--- Pass the raw vertices, not the Path objects
                    facecolors=collection.get_facecolor(),
                    edgecolors=collection.get_edgecolor(),
                    alpha=collection.get_alpha(),
                    zorder=collection.get_zorder(), 
                    label=collection.get_label()
                )
                new_ax.add_collection(new_poly)

        # 3. Copy formatting (Scales, Limits, Grid)
        new_ax.set_xscale(old_ax.get_xscale())
        new_ax.set_yscale(old_ax.get_yscale())
        new_ax.set_xlim(old_ax.get_xlim())
        new_ax.set_ylim(old_ax.get_ylim())
        
        new_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 4. Dynamic NeurIPS Title and Labels
        panel_letter = chr(97 + i) 
        new_ax.set_title(f"({panel_letter}) {benchmark_name}", fontsize=15, loc='left')
        
        new_ax.set_xlabel(old_ax.get_xlabel() or "Cumulative FLOPs", fontsize=15)
        
        # Only put the Y-axis label on the far left plot to save space
        if i == 0:
            new_ax.set_ylabel(old_ax.get_ylabel() or "Loss", fontsize=15)

        # 5. Close the old figure to free up memory
        plt.close(old_fig)

    # --- THE UNIFIED LEGEND ---
    # Grab the handles and labels ONLY from the first subplot to prevent duplicates
    handles, labels = master_axes[0].get_legend_handles_labels()
    
    clean_handles, clean_labels = [], []
    for h, l in zip(handles, labels):
        # Ignore Matplotlib's hidden labels (starting with '_') and prevent accidental duplicates
        if not l.startswith('_') and l not in clean_labels:
            clean_handles.append(h)
            clean_labels.append(l)

    # Apply the single legend to the bottom of the master figure
    if clean_handles:
        master_fig.legend(clean_handles, clean_labels, loc='lower center',
                          bbox_to_anchor=(0.5, -0.2),
                          ncol=min(4, len(clean_labels)), fontsize=15, frameon=True)

    # --- Adjust Layout ---
    plt.tight_layout()
    # Add padding to the bottom so the axes don't sit on top of the legend
    master_fig.subplots_adjust(bottom=0.2)

    return master_fig, master_axes
