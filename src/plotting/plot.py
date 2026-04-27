from pathlib import Path
from typing import Any, cast, Callable

import numpy as np
import matplotlib.pyplot as plt
import string

import neps
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
from neps.state.trial import State

from src.plotting.plot_styles import OPTIMIZER_COLORS
from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit

plt.rcParams.update({
    "font.family": "serif",        # Forces Times-style font
    "font.size": 10,               # Base font size
    "axes.labelsize": 10,          # X and Y labels
    "xtick.labelsize": 8,          # X-axis numbers
    "ytick.labelsize": 8,          # Y-axis numbers
    "legend.fontsize": 8,          # Legend labels
    "figure.figsize": (5.5, 3.5),  # Fits the exact width of a NeurIPS page
    "pdf.fonttype": 42,            # Ensures fonts render correctly in the final PDF
    "ps.fonttype": 42
})

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
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
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
        target_ax.plot(
            np.asarray(line.get_xdata(), dtype=float),
            np.asarray(line.get_ydata(), dtype=float),
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


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    return_incumbent: bool = False,
) -> tuple[
    list[float],
    list[float],
    float,
    list[tuple[float, float, float, float]],
    list[list[tuple[float, float, float, float]]],
    list[dict[str, Any]],
]:
    path = Path(path)

    flop_scale = 1e9 if "openclip" in str(path).lower() else 1.0

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
    
    # Replace NaN values with the first next non-NaN element
    for i in range(len(losses)):
        if np.isnan(losses[i]):
            # Find next non-NaN element
            for j in range(i + 1, len(losses)):
                if not np.isnan(losses[j]):
                    losses[i] = losses[j]
                    break
    
    cumsum_costs = list(np.cumsum(costs))
    max_cost = cumsum_costs[-1] if cumsum_costs else -1.0
    
    print(f"Processed seed {seed} for {path}, accumulated cost: {max_cost:.2e} FLOPs")
    if return_incumbent:
        return list(np.minimum.accumulate(losses)), cumsum_costs, max_cost, rows, learning_curve_rows, configs
    return losses, cumsum_costs, max_cost, rows, learning_curve_rows, configs


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
            # Interpolate to common grid
            interp_losses = np.interp(flop_grid, costs, losses, left=losses[0], right=losses[-1])
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


def plot_optimizer_loss(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
    return_split: bool = True,
) -> Any:
    """
    Plot loss over FLOPs for multiple optimizers with confidence bands.
    
    Keeps per-seed data and displays mean loss with std deviation as confidence band.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost (typically 'flopcount')
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        title: Plot title
        flop_range: Tuple (min_flops, max_flops) to display on x-axis (trims data outside this range)
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_optimizer_loss(
            base_path=base_path,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            normalize_flops=normalize_flops,
            return_split=False,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_optimizer_loss(
                base_path=base_path,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Average Loss",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_optimizer_loss(
                base_path=base_path,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Average Loss",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

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
    common_start = max(all_first_flops) if all_first_flops else 0
    
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
    
    ax.set_xlabel("Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)", fontsize=13)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        ax.set_xlim(max(x_lim_min), min(x_lim_max))
    ax.set_ylabel("Average Loss", fontsize=13)
    # ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if not normalize_flops:
        ax.set_xscale("log")
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
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
) -> dict[str, dict]:
    """
    Calculate relative rankings with normalized flop counts for cross-benchmark comparison.
    """
    ranking_metric = ranking_metric.lower()
    if ranking_metric not in {"incumbent", "hypervolume"}:
        raise ValueError("ranking_metric must be either 'incumbent' or 'hypervolume'")

    # Get per-seed data for all optimizers
    all_per_seed_data = {}
    all_flop_grids = []
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
        all_flop_grids.append(flop_grid)

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

    rankings = {}
    for opt_name, data in all_per_seed_data.items():
        original_flop_grid = data["flop_grid"]
        per_seed_values = data["per_seed_values"]

        # Filter original flop grid to only positive values for log scaling
        positive_mask = original_flop_grid > 1e-8
        if not positive_mask.any():
            continue
            
        # Normalize original flop grid using log scale (only for positive values)
        if np.isclose(min_flop, max_flop):
            original_normalized_flops = np.ones_like(original_flop_grid) * 0.5
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                original_normalized_flops = np.zeros_like(original_flop_grid)
                original_normalized_flops[positive_mask] = (
                    (np.log10(original_flop_grid[positive_mask]) - np.log10(min_flop)) / 
                    (np.log10(max_flop) - np.log10(min_flop))
                )
                # Set non-positive values to -inf (will be filtered out)
                original_normalized_flops[~positive_mask] = -np.inf
        
        seed_rankings_list = []
        for seed, values in per_seed_values.items():
            seed_ranks = []
            for i in range(len(original_flop_grid)):
                values_at_idx = []
                for other_opt in all_per_seed_data.keys():
                    try:
                        other_values = all_per_seed_data[other_opt]["per_seed_values"][seed]
                        if i < len(other_values):
                            values_at_idx.append(other_values[i])
                    except KeyError:
                        continue
                
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
            
            # Interpolate this seed's ranks onto the common normalized grid
            interp_ranks = np.interp(normalized_flop_grid_scaled, original_normalized_flops, seed_ranks)
            seed_rankings_list.append(interp_ranks)
        
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
) -> dict[str, dict]:
    """
    Calculate hypervolume with normalized flop counts for cross-benchmark comparison.
    """
    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
    
    # First pass: collect per-seed data and all flop grids for normalization.
    all_flop_grids: list[np.ndarray] = []
    all_per_seed_data = {}
    
    for opt_name, seeds in optimizers.items():
        flop_grid, per_seed_losses, _ = process_optimizer_per_seed(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
        )
        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_losses": per_seed_losses,
        }
        if len(flop_grid) > 0:
            all_flop_grids.append(flop_grid)

    bounds = _get_global_flop_bounds(all_flop_grids)
    
    hvs_dict = {}

    # Second pass: calculate hypervolume with dynamic reference point
    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        
        seed_hvs_list = []
        
        for seed, losses in per_seed_losses.items():
            res = ScalingLawGuidedOptimizer.compute_normalized_search_hypervolume(
                losses.tolist(), 
                flop_grid.tolist(),
            )
            assert res is not None
            _, hvs = res
            seed_hvs_list.append(np.array(hvs))
        
        # Aggregate across seeds
        seed_hvs_array = np.array(seed_hvs_list)
        mean_hvs = np.nanmean(seed_hvs_array, axis=0)
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


def calculate_pareto_fronts(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
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
            )
            seed_rows[seed] = rows

        all_per_seed_rows[opt_name] = seed_rows

    pareto_dict = {}
    for opt_name, seed_rows in all_per_seed_rows.items():

        pooled_front_flops = []
        pooled_front_losses = []

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
                f_points, l_points = res
                pooled_front_flops.extend(f_points)
                pooled_front_losses.extend(l_points)

        if not pooled_front_flops:
            continue

        pooled_flops_arr = np.asarray(pooled_front_flops, dtype=float)
        pooled_losses_arr = np.asarray(pooled_front_losses, dtype=float)
        pareto_dict[opt_name] = {
            "flop_counts": pooled_flops_arr.astype(float),
            "front_losses": pooled_losses_arr.astype(float),
        }

    return pareto_dict


def plot_pareto_front_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Pareto Front Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    pareto_flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = False,
    return_split: bool = True,
    true_pareto_info: dict[str, list] | None = None,
    pareto_target_flop: float | None = None,
) -> Any:
    """Plot Pareto-front loss over FLOPs with mean and std bands."""
    if pareto_flop_range is not None and pareto_flop_range[0] >= pareto_flop_range[1]:
        raise ValueError("pareto_flop_range must satisfy min < max")

    if pareto_target_flop is not None and pareto_target_flop <= 0:
        raise ValueError("pareto_target_flop must be positive when provided")

    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_pareto_front_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            pareto_flop_range=pareto_flop_range,
            normalize_flops=normalize_flops,
            return_split=False,
            true_pareto_info=true_pareto_info,
            pareto_target_flop=pareto_target_flop,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_pareto_front_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                pareto_flop_range=pareto_flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
                true_pareto_info=true_pareto_info,
                pareto_target_flop=pareto_target_flop,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Loss",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_pareto_front_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                pareto_flop_range=pareto_flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
                true_pareto_info=true_pareto_info,
                pareto_target_flop=pareto_target_flop,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Loss",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    fig, ax = plt.subplots(figsize=figsize)
    base_path = Path(base_path)

    pareto_data = calculate_pareto_fronts(
        base_path=base_path,
        optimizers=optimizers,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
    )

    all_data = []
    all_first_flops = []

    for opt_name, data in pareto_data.items():
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        flop_counts = np.asarray(data["flop_counts"], dtype=float)
        front_losses = data["front_losses"]
        # std_front_losses = data["std_front_losses"]

        if len(flop_counts) > 0:
            all_data.append({
                "opt_name": opt_name,
                "color": color,
                "linestyle": linestyle,
                "flop_counts": flop_counts,
                "front_losses": front_losses,
                # "std_front_losses": std_front_losses,
            })
            non_zero_flops = flop_counts[flop_counts > 1e-8]
            if len(non_zero_flops) > 0:
                all_first_flops.append(float(non_zero_flops[0]))

    common_start = min(all_first_flops) if all_first_flops else 0.0

    scatter_x_series: list[np.ndarray] = []
    for data in all_data:
        flop_counts = data["flop_counts"]
        front_losses = data["front_losses"]
        # std_front_losses = data["std_front_losses"]

        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        front_losses = front_losses[mask]

        if pareto_flop_range is not None:
            pareto_mask = (flop_counts >= pareto_flop_range[0]) & (flop_counts <= pareto_flop_range[1])
            flop_counts = flop_counts[pareto_mask]
            front_losses = front_losses[pareto_mask]
        # std_front_losses = std_front_losses[mask]

        # if len(flop_counts) > 0 and flop_counts[0] > common_start + 1e-8:
        #     flop_counts = np.concatenate([[common_start], flop_counts])
        #     front_losses = np.concatenate([[front_losses[0]], front_losses])
            # std_front_losses = np.concatenate([[std_front_losses[0]], std_front_losses])

        if len(flop_counts) > 0:
            scatter_x_series.append(flop_counts)

            # Fit in log-log space with shared utility used by Kaplan predictor.
            fit_result = fit_power_law(
                np.asarray(flop_counts, dtype=float),
                np.asarray(front_losses, dtype=float),
            )
            slope_txt = ""
            if fit_result is not None:
                slope_txt = f" (m={fit_result.slope:.2f})"
                fit_x = np.asarray(flop_counts, dtype=float)
                fit_x = fit_x[np.isfinite(fit_x) & (fit_x > 0)]
                if len(fit_x) >= 2:
                    fit_grid = np.logspace(np.log10(float(np.min(fit_x))), np.log10(float(np.max(fit_x))), 100)
                    fit_linestyle = ":" if "_with_priors" in data["opt_name"] else "-"
                    ax.plot(
                        fit_grid,
                        np.power(10.0, fit_result.intercept) * np.power(fit_grid, fit_result.slope),
                        color=data["color"],
                        linestyle=fit_linestyle,
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
                    ax.scatter(
                        [pareto_target_flop],
                        [predicted_loss],
                        marker="*",
                        s=140,
                        color=data["color"],
                        edgecolors="black",
                        linewidths=0.6,
                        alpha=0.95,
                        label=f"{get_optimizer_label(data['opt_name'])} Pred loss {predicted_loss:.2e}",
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

    ax.set_xlabel("FLOPs", fontsize=13)

    benchmark_xlim: tuple[float, float] | None = None
    finite_benchmark_x = [xs[np.isfinite(xs)] for xs in scatter_x_series if len(xs) > 0]
    finite_benchmark_x = [xs for xs in finite_benchmark_x if len(xs) > 0]
    if finite_benchmark_x:
        left = min(float(np.min(xs)) for xs in finite_benchmark_x)
        right = max(float(np.max(xs)) for xs in finite_benchmark_x)
        if left < right:
            benchmark_xlim = (left, right)

    # Overlay true Pareto-front points in the same FLOP units as optimizer curves.
    if true_pareto_info is not None:
        true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
        true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)

        # Keep unit consistency with process_seed/process_optimizer_per_seed.


        valid_true = np.isfinite(true_flops) & np.isfinite(true_losses)
        true_flops = true_flops[valid_true]
        true_losses = true_losses[valid_true]

        if len(true_flops) > 0:
            true_x = true_flops

            # Clip true Pareto to benchmark optimizer FLOP range.
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
                    true_slope_txt = f" (m={true_fit_result.slope:.2f})"
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
                    label=f"True Pareto Front{true_slope_txt}",
                )
    if benchmark_xlim is not None:
        if flop_range is not None:
            ax.set_xlim(max(benchmark_xlim[0], flop_range[0]), max(benchmark_xlim[1], flop_range[1]))
        else:
            ax.set_xlim(benchmark_xlim[0], benchmark_xlim[1])
    ax.set_ylabel("Loss", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def plot_predicted_loss_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Predicted Loss Over Optimization",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    return_split: bool = True,
) -> Any:
    """Plot predicted loss over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_predicted_loss_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            target_flop=target_flop,
            predictor_type=predictor_type,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            true_pareto_info=true_pareto_info,
            return_split=False,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_predicted_loss_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                true_pareto_info=true_pareto_info,
                return_split=False,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Cumulative FLOPs",
                ylabel="Predicted Loss",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_predicted_loss_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                true_pareto_info=true_pareto_info,
                return_split=False,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Cumulative FLOPs",
                ylabel="Predicted Loss",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    from src.loss_predictor import LossPredictorTypes, predict_loss

    if predictor_type is None:
        predictor_type = LossPredictorTypes.KAPLAN

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)

    # Optional reference: true Pareto loss at closest FLOP to target.
    if true_pareto_info is not None:
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
                label=f"True Loss @ {closest_flop:.2e} FLOPs",
            )

    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed = seeds[0]
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

        pred_means: list[float] = []
        # pred_lowers: list[float] = []
        # pred_uppers: list[float] = []

        # Requirement: evaluate prefixes from 1..len(rows).
        for n_trials in range(1, len(rows) + 1):
            try:
                predicted_loss, _ = predict_loss(
                    predictor_type=predictor_type,
                    target_flop=target_flop,
                    rows=rows[:n_trials],
                )
                pred_means.append(float(predicted_loss))
                # pred_lowers.append(float(info.predicted_loss_lower))
                # pred_uppers.append(float(info.predicted_loss_upper))
            except Exception:
                pred_means.append(np.nan)
                # pred_lowers.append(np.nan)
                # pred_uppers.append(np.nan)

        x = np.asarray(cumulative_flops, dtype=float)
        y = np.asarray(pred_means, dtype=float)
        finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
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

        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)

        ax.plot(
            x,
            y,
            label=get_optimizer_label(opt_name),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        # ax.fill_between(
        #     x,
        #     y_low,
        #     y_high,
        #     alpha=0.2,
        #     color=color,
        # )

        x_series_for_limits.append(x)

    ax.set_xlabel("Cumulative FLOPs", fontsize=13)
    ax.set_ylabel("Predicted Loss", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if flop_range is not None:
        ax.set_xlim(flop_range[0], flop_range[1])
    elif x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    return fig, ax


def plot_predicted_n_param_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    target_flop: float,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Predicted n_param Over Optimization",
    flop_range: tuple[float, float] | None = None,
    true_pareto_info: dict[str, list] | None = None,
    return_split: bool = True,
) -> Any:
    """Plot predicted n_param over cumulative FLOPs using trial prefixes."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_predicted_n_param_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            target_flop=target_flop,
            predictor_type=predictor_type,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            true_pareto_info=true_pareto_info,
            return_split=False,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_predicted_n_param_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                true_pareto_info=true_pareto_info,
                return_split=False,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Cumulative FLOPs",
                ylabel="Predicted Model Size",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_predicted_n_param_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                target_flop=target_flop,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                true_pareto_info=true_pareto_info,
                return_split=False,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Cumulative FLOPs",
                ylabel="Predicted Model Size",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    from src.scale_param_predictor import ScaleParamPredictorTypes, predict_scale_parameters

    if predictor_type is None:
        predictor_type = ScaleParamPredictorTypes.POWER_LAW_FIT

    fig, ax = plt.subplots(figsize=figsize)
    x_series_for_limits: list[np.ndarray] = []

    base_path = Path(base_path)

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
            ax.axhline(
                y=closest_n_param,
                color="black",
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                label=f"True n_param @ {closest_flop:.2e} FLOPs",
            )

    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed = seeds[0]
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
            assert len(cumulative_flops) == len(learning_curve_rows), "Expected one learning curve row per trial for Chinchilla approach 1 optimizers"
            total_trials = len(cumulative_flops)
        else:
            source_rows = rows
            total_trials = len(rows)

        if total_trials <= 0:
            continue

        pred_n_params: list[float] = []
        for n_trials in range(1, total_trials + 1):
            try:
                (pred_n_param, _pred_n_data), _ = predict_scale_parameters(
                    predictor_type=predictor_type,
                    target_flop=target_flop,
                    rows=source_rows[:n_trials],
                )
                pred_n_params.append(float(pred_n_param))
            except Exception:
                pred_n_params.append(np.nan)

        x = np.asarray(cumulative_flops[:total_trials], dtype=float)
        y = np.asarray(pred_n_params, dtype=float)
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

        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)

        ax.plot(
            x,
            y,
            label=get_optimizer_label(opt_name),
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )
        x_series_for_limits.append(x)

    ax.set_xlabel("Cumulative FLOPs", fontsize=13)
    ax.set_ylabel("Predicted n_param", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if flop_range is not None:
        ax.set_xlim(flop_range[0], flop_range[1])
    elif x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

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
        print(f"maybe_cfgs: {maybe_cfgs}")
        cfg = {
            (k[len("config."):] if isinstance(k, str) and k.startswith("config.") else k): v
            for k, v in dict(maybe_cfgs[idx]).items()
        }
    print(cfg)
    losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    cfg["actual_loss"] = float(losses[idx]) if idx < len(losses) else float("nan")
    return cfg


def plot_predicted_hp_table_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    space: Any,
    target_flop: float,
    evaluate_pipeline: Callable,
    true_pareto_info: dict[str, list] | None = None,
    flop_range: tuple[float, float] | None = None,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Predicted Hyperparameters at Target FLOPs",
    return_split: bool = True,
) -> Any:
    """Create a NeurIPS-style table figure (no vertical lines) with predicted HP configs per optimizer."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            space=space,
            evaluate_pipeline=evaluate_pipeline,
            true_pareto_info=true_pareto_info,
            target_flop=target_flop,
            flop_range=flop_range,
            predictor_type=predictor_type,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            figsize=figsize,
            title=title,
            return_split=False,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                space=space,
                evaluate_pipeline=evaluate_pipeline,
                true_pareto_info=true_pareto_info,
                target_flop=target_flop,
                flop_range=flop_range,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (with priors)",
                return_split=False,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="",
                ylabel="",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                space=space,
                evaluate_pipeline=evaluate_pipeline,
                true_pareto_info=true_pareto_info,
                target_flop=target_flop,
                flop_range=flop_range,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (without priors)",
                return_split=False,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="",
                ylabel="",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    from src.hp_predictor import HPPredictorTypes, predict_hp_from_scaling_law

    if predictor_type is None:
        predictor_type = HPPredictorTypes.LINEAR_REGRESSION

    base_path = Path(base_path)

    optimizer_rows: list[str] = []
    predicted_configs_by_optimizer: dict[str, dict[str, Any]] = {}
    hp_columns: list[str] = []

    if hasattr(space, "searchables"):
        hp_columns = [str(k) for k in getattr(space, "searchables").keys()]

    for opt_name, seeds in optimizers.items():
        if not seeds:
            continue

        seed = seeds[0]
        try:
            losses, cumsum_costs, _max_cost, _rows, _learning_curve_rows, configs = process_seed(
                path=base_path / opt_name,
                seed=seed,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                return_incumbent=False,
            )
        except Exception as e:
            print(f"Warning: could not load data for {opt_name} seed {seed}: {e}")
            continue

        if not configs or not losses or not cumsum_costs:
            continue

        if flop_range is not None:
            cutoff_indices = np.where(np.asarray(cumsum_costs, dtype=float) < float(flop_range[1]))[0]
            if len(cutoff_indices) == 0:
                continue
            cutoff = int(cutoff_indices[-1]) + 1
            losses = losses[:cutoff]
            cumsum_costs = cumsum_costs[:cutoff]
            configs = configs[:cutoff]

        if not configs or not losses or not cumsum_costs:
            continue

        cumsum_arr = np.asarray(cumsum_costs, dtype=float)
        actual_flops = np.diff(cumsum_arr, prepend=0.0).tolist()
        pred_result = predict_hp_from_scaling_law(
            space=space,
            hp_predictor_type=predictor_type,
            target_flops=target_flop,
            configs=configs,
            losses=[float(v) for v in losses],
            flops=[float(v) for v in actual_flops],
        )
        predicted_cfg = cast(dict[str, Any], pred_result.get("extrapolated_config", {}))
        predicted_cfg_with_loss = dict(predicted_cfg)
        try:
            loss = evaluate_pipeline(**predicted_cfg)["objective_to_minimize"]
            predicted_cfg_with_loss["actual_loss"] = loss
        except Exception:
            predicted_cfg_with_loss["actual_loss"] = float("nan")
        

        if not hp_columns:
            hp_columns = list(predicted_cfg_with_loss.keys())

        optimizer_rows.append(get_optimizer_label(opt_name))
        predicted_configs_by_optimizer[opt_name] = predicted_cfg_with_loss

    if true_pareto_info is not None:
        true_cfg_row = _extract_true_pareto_config_row(
            true_pareto_info=true_pareto_info,
            target_flop=target_flop,
        )
        if true_cfg_row:
            optimizer_rows.append("True Pareto")
            predicted_configs_by_optimizer["__true_pareto__"] = true_cfg_row

    if predicted_configs_by_optimizer:
        inferred_cols: list[str] = []
        for cfg in predicted_configs_by_optimizer.values():
            for col in cfg.keys():
                if col not in inferred_cols and col != "actual_loss":
                    inferred_cols.append(col)
        hp_columns = inferred_cols
        if "actual_loss" not in hp_columns:
            hp_columns.append("actual_loss")

    if not optimizer_rows or not hp_columns:
        return _placeholder_plot(
            figsize=figsize,
            title=title,
            xlabel="",
            ylabel="",
        )

    fig_w = max(figsize[0], 1.6 + 1.4 * len(hp_columns))
    fig_h = max(figsize[1], 1.8 + 0.5 * len(optimizer_rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text: list[list[str]] = []
    for opt_name, _label in zip(predicted_configs_by_optimizer.keys(), optimizer_rows):
        cfg = predicted_configs_by_optimizer[opt_name]
        row_vals = [_format_hp_table_value(cfg.get(col, "")) for col in hp_columns]
        cell_text.append(row_vals)

    table = ax.table(
        cellText=cell_text,
        rowLabels=optimizer_rows,
        colLabels=hp_columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8 if r == 0 else 0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F5F5F5")
        if c == -1 and r > 0:
            cell.set_text_props(weight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    return fig, ax


def plot_hypervolume_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Hypervolume Over Optimization",
    flop_range: tuple[float, float] | None = None,
    normalize_flops: bool = True,
    return_split: bool = True,
) -> Any:
    """
    Plot hypervolume with mean and std bands (calculated per-seed).
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        title: Plot title
        flop_range: Tuple (min_flops, max_flops) to display on x-axis (trims data outside this range)
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_hypervolume_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            normalize_flops=normalize_flops,
            return_split=False,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_hypervolume_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Average Normalized Pareto Hypervolume",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_hypervolume_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                normalize_flops=normalize_flops,
                return_split=False,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)",
                ylabel="Average Normalized Pareto Hypervolume",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    fig, ax = plt.subplots(figsize=figsize)

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
            ax.fill_between(
                flop_counts,
                mean_hvs - std_hvs,
                mean_hvs + std_hvs,
                alpha=0.2,
                color=data['color'],
            )
    
    ax.set_xlabel("Normalized FLOPs (log scale)" if normalize_flops else "FLOPs (log scale)", fontsize=13)
    if effective_flop_range is not None:
        ax.set_xlim(effective_flop_range[0], effective_flop_range[1])
    elif normalize_flops:
        left, right = _get_common_xlim(x_series_for_limits)
        ax.set_xlim(left, right)
    ax.set_ylabel("Average Normalized Pareto Hypervolume", fontsize=13)
    # ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if not normalize_flops:
        ax.set_xscale("log")
    
    # Automatically scale y-axis to fit data tightly
    ax.autoscale(enable=True, axis='y', tight=True)
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


# def plot_relative_ranking_over_flops(
#     *,
#     base_path: str | Path,
#     optimizers: dict[str, list[int]] | dict[str, list[int | str]],
#     key_to_extract: str | None = None,
#     loss_key: str = "objective_to_minimize",
#     log_scale: bool = False,
#     figsize: tuple[int, int] = (10, 6),
#     title: str = "Optimizer Rankings Over Optimization",
#     flop_range: tuple[float, float] | None = None,
#     return_split: bool = True,
#     ranking_metric: str = "incumbent",
# ) -> Any:
#     """
#     Plot optimizer rankings (calculated per-seed, then aggregated with mean/std).
    
#     Args:
#         base_path: Base path where optimizer directories are located
#         optimizers: Dict mapping optimizer name to list of seeds
#         key_to_extract: Key to extract from results as cost
#         log_scale: Whether to use log scale for x-axis
#         figsize: Figure size
#         title: Plot title
#         flop_range: Tuple (min_flops, max_flops) to display on x-axis (trims data outside this range)
    
#     Returns:
#         fig: Matplotlib figure
#         ax: Axes object
#     """
#     if return_split:
#         all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

#         fig_all, ax_all = cast(tuple[Any, Any], plot_relative_ranking_over_flops(
#             base_path=base_path,
#             optimizers=all_opts,
#             key_to_extract=key_to_extract,
#             loss_key=loss_key,
#             log_scale=log_scale,
#             figsize=figsize,
#             title=title,
#             flop_range=flop_range,
#             return_split=False,
#             ranking_metric=ranking_metric,
#         ))

#         if with_priors:
#             fig_with, ax_with = cast(tuple[Any, Any], plot_relative_ranking_over_flops(
#                 base_path=base_path,
#                 optimizers=with_priors,
#                 key_to_extract=key_to_extract,
#                 loss_key=loss_key,
#                 log_scale=log_scale,
#                 figsize=figsize,
#                 title=f"{title} (with priors)",
#                 flop_range=flop_range,
#                 return_split=False,
#                 ranking_metric=ranking_metric,
#             ))
#         else:
#             fig_with, ax_with = _placeholder_plot(
#                 figsize=figsize,
#                 title=f"{title} (with priors)",
#                 xlabel="FLOPs (log scale)",
#                 ylabel="Relative Rank",
#             )

#         if without_priors:
#             fig_without, ax_without = cast(tuple[Any, Any], plot_relative_ranking_over_flops(
#                 base_path=base_path,
#                 optimizers=without_priors,
#                 key_to_extract=key_to_extract,
#                 loss_key=loss_key,
#                 log_scale=log_scale,
#                 figsize=figsize,
#                 title=f"{title} (without priors)",
#                 flop_range=flop_range,
#                 return_split=False,
#                 ranking_metric=ranking_metric,
#             ))
#         else:
#             fig_without, ax_without = _placeholder_plot(
#                 figsize=figsize,
#                 title=f"{title} (without priors)",
#                 xlabel="FLOPs (log scale)",
#                 ylabel="Relative Rank",
#             )

#         return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

#     fig, ax = plt.subplots(figsize=figsize)
    
#     rankings = calculate_rankings(
#         base_path=base_path,
#         optimizers=optimizers,
#         key_to_extract=key_to_extract,
#         loss_key=loss_key,
#         log_scale=log_scale,
#         ranking_metric=ranking_metric,
#     )
    
#     # Extract min_flops from flop_range if provided
#     min_flops_start = flop_range[0] if flop_range is not None else 0
    
#     for opt_name, rank_data in rankings.items():
#         color = get_optimizer_color(opt_name)
#         linestyle = get_optimizer_linestyle(opt_name)
#         flop_counts = rank_data["flop_counts"]
#         mean_ranks = rank_data["mean_ranks"]
#         std_ranks = rank_data["std_ranks"]
        
#         # Trim data to flop_range if specified
#         if flop_range is not None:
#             min_flops, max_flops = flop_range
#             mask = (flop_counts >= min_flops) & (flop_counts <= max_flops)
#             flop_counts = flop_counts[mask]
#             mean_ranks = mean_ranks[mask]
#             std_ranks = std_ranks[mask]
        
#         # Prepend a point at min_flops_start if the data starts after it
#         if len(flop_counts) > 0 and flop_counts[0] > min_flops_start + 1e-8:
#             flop_counts = np.concatenate([[min_flops_start], flop_counts])
#             mean_ranks = np.concatenate([[mean_ranks[0]], mean_ranks])
#             std_ranks = np.concatenate([[std_ranks[0]], std_ranks])
        
#         if len(flop_counts) > 0:
#             # Plot mean ranking
#             ax.plot(
#                 flop_counts,
#                 mean_ranks,
#                 label=opt_name,
#                 color=color,
#                 linestyle=linestyle,
#                 linewidth=2.5,
#             )
            
#             # Add std band
#             ax.fill_between(
#                 flop_counts,
#                 mean_ranks - std_ranks,
#                 mean_ranks + std_ranks,
#                 alpha=0.2,
#                 color=color,
#             )
    
#     ax.set_xlabel("FLOPs (log scale)", fontsize=13)
#     ax.set_ylabel("Relative Rank", fontsize=13)
#     ax.set_title(title, fontsize=14, fontweight="bold")
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.invert_yaxis()  # Invert so rank 1 is at top
#     ax.set_xscale("log")
    
#     # Set y-axis to linear scale with rank labels
#     num_optimizers = len(optimizers)
#     ax.set_xlim(flop_range[0], flop_range[1])
#     ax.set_ylim(num_optimizers + 0.5, 0.5)
#     ax.set_yticks(range(1, num_optimizers + 1))
#     ax.set_yticklabels(range(1, num_optimizers + 1))
    
#     # Place legend below with reduced padding
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
#                ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
#     plt.tight_layout(rect=[0, 0.10, 1, 1])
    
#     return fig, ax


def plot_normalized_relative_ranking_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Optimizer Rankings Over Optimization (Normalized)",
    flop_range: tuple[float, float] | None = None,
    return_split: bool = True,
    ranking_metric: str = "incumbent",
) -> Any:
    """
    Plot optimizer rankings with a normalized x-axis for a single benchmark.
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_normalized_relative_ranking_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            figsize=figsize,
            title=title,
            flop_range=flop_range,
            return_split=False,
            ranking_metric=ranking_metric,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_normalized_relative_ranking_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (with priors)",
                flop_range=flop_range,
                return_split=False,
                ranking_metric=ranking_metric,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Normalized FLOPs (log scale)",
                ylabel=f"Average {ranking_metric.capitalize()} Rank",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_normalized_relative_ranking_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=log_scale,
                figsize=figsize,
                title=f"{title} (without priors)",
                flop_range=flop_range,
                return_split=False,
                ranking_metric=ranking_metric,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Normalized FLOPs (log scale)",
                ylabel=f"Average {ranking_metric.capitalize()} Rank",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

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
    
    ax.set_xlabel("Normalized FLOPs (log scale)", fontsize=13)
    ax.set_ylabel(f"Average {ranking_metric.capitalize()} Rank", fontsize=13)
    # ax.set_title(title, fontsize=14, fontweight="bold")
    left, right = _get_common_xlim(x_series_for_limits)
    ax.set_xlim(left, right)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()
    
    num_optimizers = len(optimizers)
    ax.set_ylim(num_optimizers + 0.5, 0.5)
    ax.set_yticks(range(1, num_optimizers + 1))
    ax.set_yticklabels(range(1, num_optimizers + 1))
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


def plot_optimizer_rankings_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Optimizer Rankings Across Benchmarks",
    return_split: bool = True,
    ranking_metric: str = "incumbent",
) -> Any:
    """
    Plot mean optimizer rankings aggregated across all benchmarks with std bands.
    
    For each benchmark, calculate ranking curves. Then for each optimizer, aggregate
    their rankings across all benchmarks and plot mean ± std with confidence bands.
    
    Args:
        base_path: Base path where benchmark optimizer directories are located
        benchmarks: List of benchmark names
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        figsize: Figure size
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            figsize=figsize,
            title=title,
            return_split=False,
            ranking_metric=ranking_metric,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (with priors)",
                return_split=False,
                ranking_metric=ranking_metric,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="Normalized FLOPs (log scale)",
                ylabel=f"Average {ranking_metric.capitalize()} Rank",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_optimizer_rankings_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (without priors)",
                return_split=False,
                ranking_metric=ranking_metric,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="Normalized FLOPs (log scale)",
                ylabel=f"Average {ranking_metric.capitalize()} Rank",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

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
                log_scale=False,
                ranking_metric=ranking_metric,
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
        
        # Use flop counts from first benchmark as reference (they should be similar after interpolation)
        # All flop counts are log-spaced FLOP values (min_flop to max_flop) and have the same length
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
    
    ax.set_xlabel("Normalized FLOPs (log scale)", fontsize=13)
    left, right = _get_common_xlim(x_series_for_limits)
    ax.set_xlim(left, right)
    ax.set_ylabel(f"Average {ranking_metric.capitalize()} Rank", fontsize=13)
    # ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()  # Invert so rank 1 is at top
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax


def plot_loss_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_range: tuple[float, float] | None = None,
    return_split: bool = True,
) -> Any:
    """
    Create composite loss figures with all benchmarks + across-benchmark aggregate.
    
    Reuses existing plot_optimizer_loss function and combines plots in a single figure
    with shared legend.
    
    Args:
        base_path: Base path where benchmark optimizer directories are located
        benchmarks: List of benchmark names
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        loss_key: Loss metric key
        flop_range: Tuple (min_flops, max_flops) to trim data
        return_split: If True, returns 3 figures (all/with_priors/without_priors)
    
    Returns:
        If return_split=True: tuple of 3 figures
        If return_split=False: single figure (fig, axes)
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        result_all = cast(tuple[Any, Any], plot_loss_composite_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            flop_range=flop_range,
            return_split=False,
        ))

        if with_priors:
            result_with = cast(tuple[Any, Any], plot_loss_composite_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                flop_range=flop_range,
                return_split=False,
            ))
        else:
            fig_with, ax_with = plt.subplots(figsize=(12, 4 * (len(benchmarks) + 1)))
            ax_with.text(0.5, 0.5, "No optimizers in with_priors group", 
                        ha="center", va="center", transform=ax_with.transAxes)
            result_with = (fig_with, ax_with)

        if without_priors:
            result_without = cast(tuple[Any, Any], plot_loss_composite_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                flop_range=flop_range,
                return_split=False,
            ))
        else:
            fig_without, ax_without = plt.subplots(figsize=(12, 4 * (len(benchmarks) + 1)))
            ax_without.text(0.5, 0.5, "No optimizers in without_priors group",
                           ha="center", va="center", transform=ax_without.transAxes)
            result_without = (fig_without, ax_without)

        return result_all, result_with, result_without

    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    figsize = (12, 4 * num_subplots)
    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True)
    if num_subplots == 1:
        axes = [axes]
    
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
                figsize=(12, 4),
                title="",
                flop_range=flop_range,
                normalize_flops=True,
                return_split=False,
            ))
            ax.set_title(f"({letter}) {BENCHMARK_TITLES[benchmark_name]}", loc='left', fontsize=10, fontweight='bold')
            
            # Re-plot line data instead of reusing artists from another figure.
            _replot_lines(ax_bench, ax)
            
            # Copy axis properties
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            # ax.set_title(benchmark_name, fontsize=12, fontweight="bold")
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
            figsize=(12, 4),
            return_split=False,
            ranking_metric="incumbent",
        ))
        title="Across Benchmarks (Incumbent Ranking)"
        
        # Re-plot line data instead of reusing artists from another figure.
        _replot_lines(ax_agg, agg_ax)
        
        # Copy axis properties
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.set_title(f"({string.ascii_lowercase[0]}) {title}", loc='left', fontsize=10, fontweight='bold')
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        # agg_ax.invert_yaxis()
        plt.close(fig_agg)
        
    except Exception as e:
        print(f"Warning: Could not plot aggregate ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", 
                   ha="center", va="center", transform=agg_ax.transAxes)

    # Enforce one shared x-range across all composite panels.
    x_series_for_limits: list[np.ndarray] = []
    for ax in axes:
        for line in ax.get_lines():
            x_values = np.asarray(line.get_xdata(), dtype=float)
            if len(x_values) > 0:
                x_series_for_limits.append(x_values)
    if x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        axes[0].set_xlim(left, right)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add single shared legend at bottom
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=3.0)
    
    return fig, axes


def plot_hypervolume_composite_across_benchmarks(
    *,
    base_path: Path,
    benchmarks: list[str],
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    flop_range: tuple[float, float] | None = None,
    return_split: bool = True,
) -> Any:
    """
    Create composite hypervolume figures with all benchmarks + across-benchmark aggregate.
    
    Reuses existing plot_hypervolume_over_flops function and combines plots in a single figure
    with shared legend.
    
    Args:
        base_path: Base path where benchmark optimizer directories are located
        benchmarks: List of benchmark names
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        loss_key: Loss metric key
        flop_range: Tuple (min_flops, max_flops) to trim data
        return_split: If True, returns 3 figures (all/with_priors/without_priors)
    
    Returns:
        If return_split=True: tuple of 3 figures
        If return_split=False: single figure (fig, axes)
    """
    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        result_all = cast(tuple[Any, Any], plot_hypervolume_composite_across_benchmarks(
            base_path=base_path,
            benchmarks=benchmarks,
            optimizers=all_opts,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            flop_range=flop_range,
            return_split=False,
        ))

        if with_priors:
            result_with = cast(tuple[Any, Any], plot_hypervolume_composite_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=with_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                flop_range=flop_range,
                return_split=False,
            ))
        else:
            fig_with, ax_with = plt.subplots(figsize=(12, 4 * (len(benchmarks) + 1)))
            ax_with.text(0.5, 0.5, "No optimizers in with_priors group", 
                        ha="center", va="center", transform=ax_with.transAxes)
            result_with = (fig_with, ax_with)

        if without_priors:
            result_without = cast(tuple[Any, Any], plot_hypervolume_composite_across_benchmarks(
                base_path=base_path,
                benchmarks=benchmarks,
                optimizers=without_priors,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                flop_range=flop_range,
                return_split=False,
            ))
        else:
            fig_without, ax_without = plt.subplots(figsize=(12, 4 * (len(benchmarks) + 1)))
            ax_without.text(0.5, 0.5, "No optimizers in without_priors group",
                           ha="center", va="center", transform=ax_without.transAxes)
            result_without = (fig_without, ax_without)

        return result_all, result_with, result_without

    # Create composite figure with subplots for each benchmark + aggregate
    num_subplots = len(benchmarks) + 1
    figsize = (12, 4 * num_subplots)
    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True)
    if num_subplots == 1:
        axes = [axes]
    
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
                figsize=(12, 4),
                flop_range=flop_range,
                normalize_flops=True,
                return_split=False,
            ))
            letter = string.ascii_lowercase[idx + 1]
            
            # Re-plot line data instead of reusing artists from another figure.
            _replot_lines(ax_bench, ax)
            
            # Copy axis properties
            ax.set_title(f"({letter}) {BENCHMARK_TITLES[benchmark_name]}", loc='left', fontsize=10, fontweight='bold')
            ax.set_ylim(ax_bench.get_ylim())
            ax.set_xlabel(ax_bench.get_xlabel())
            ax.set_ylabel(ax_bench.get_ylabel())
            # ax.set_title(benchmark_name, fontsize=12, fontweight="bold")
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
            title="",
            figsize=(12, 4),
            return_split=False,
            ranking_metric="hypervolume",
        ))
        title="Across Benchmarks (Hypervolume Ranking)"

        
        # Re-plot line data instead of reusing artists from another figure.
        _replot_lines(ax_agg, agg_ax)
        plt.close(fig_agg)
        
        # Copy axis properties
        agg_ax.set_title(f"({string.ascii_lowercase[0]}) {title}", loc='left', fontsize=10, fontweight='bold')
        agg_ax.set_ylim(ax_agg.get_ylim())
        agg_ax.set_xlabel(ax_agg.get_xlabel())
        agg_ax.set_ylabel(ax_agg.get_ylabel())
        agg_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        # agg_ax.invert_yaxis()
        
    except Exception as e:
        print(f"Warning: Could not plot aggregate ranking: {e}")
        agg_ax.text(0.5, 0.5, f"Error plotting aggregate", 
                   ha="center", va="center", transform=agg_ax.transAxes)

    # Enforce one shared x-range across all composite panels.
    x_series_for_limits: list[np.ndarray] = []
    for ax in axes:
        for line in ax.get_lines():
            x_values = np.asarray(line.get_xdata(), dtype=float)
            if len(x_values) > 0:
                x_series_for_limits.append(x_values)
    if x_series_for_limits:
        left, right = _get_common_xlim(x_series_for_limits)
        axes[0].set_xlim(left, right)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add single shared legend at bottom
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(hspace=0.3)
    
    return fig, axes


if __name__ == "__main__":
    """
    Plot loss, hypervolume, and rankings for all benchmarks in configs.
    
    Per-seed data is kept throughout calculations, then aggregated with mean/std.
    """
    from pathlib import Path
    
    results_dir = Path("/work/dlclarge1/alipourn-benchmark_res")
    # loss_key = "objective_to_minimize"
    
    
    all_seeds = list(set(range(40, 51)))
    # all_seeds = [40]
    optimizers_dict = {
        "random_search": all_seeds,
        "random_search_with_priors": all_seeds,
        "chinchilla_approach1": all_seeds,
        "chinchilla_approach1_with_priors": all_seeds,
        "chinchilla_approach2": all_seeds,
        "chinchilla_approach2_with_priors": all_seeds,
        # "chinchilla_gemstones": all_seeds,
        "carbs": all_seeds,
        "carbs_with_priors": all_seeds,
    }
    
    # Try to import benchmark configs, fallback to hardcoded
    from src.benchmarks.configs import BenchmarkEnum
    from src.benchmarks.llm_space import LLM_TARGETS
    from src.benchmarks.openclip_space import OPENCLIP_TARGETS
    from src.plotting.plot_styles import BENCHMARK_TITLES
    from src.extrapolate import load_or_generate_pareto_info
    from src.loss_predictor import LossPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    
    benchmarks = [
        BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value
        ]
    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {benchmark_name}")
        print(f"{'='*60}")
        true_pareto_info = load_or_generate_pareto_info(benchmark_name)
        targets = {BenchmarkEnum.LLM.value: LLM_TARGETS, BenchmarkEnum.OPENCLIP.value: OPENCLIP_TARGETS}
        
        for target in targets[benchmark_name]:
            try:
                target = target.value
            except AttributeError:
                pass
            loss_key = f"extra.predictions.{target}.mean"
            # Create output directory
            plot_output_dir = results_dir / benchmark_name / "plots" / target
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            
            from src.benchmarks.configs import get_benchmark
            benchmark_cfg = get_benchmark(benchmark_name, False)
            flop_range = (5e16 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 9.99e18, get_benchmark(benchmark_name, False).max_evaluation_flops * 0.9 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1))
            
            pareto_flop_range = (
                get_benchmark(benchmark_name, False).min_flops * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1), 
                get_benchmark(benchmark_name, False).max_flops * 0.8 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1))
            
            
            
            flop_range_to_show_in_pareto = (
                get_benchmark(benchmark_name, False).min_flops * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1), 
                get_benchmark(benchmark_name, False).max_flops * 1 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1))
            

            pareto_target_flop = get_benchmark(benchmark_name, False).max_flops * 0.9 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1)
            print(f"  Plotting pareto front loss...")
            # (fig_pareto_all, _), (fig_pareto_with, _), (fig_pareto_without, _) = plot_pareto_front_over_flops(
            #     base_path=results_dir / benchmark_name,
            #     optimizers={k: [40] for k in optimizers_dict.keys()},
            #     key_to_extract="cost",
            #     log_scale=False,
            #     loss_key=loss_key,
            #     title=f"{BENCHMARK_TITLES[benchmark_name]}",
            #     flop_range=flop_range_to_show_in_pareto,
            #     pareto_flop_range=pareto_flop_range,
            #     true_pareto_info=true_pareto_info,
            #     pareto_target_flop=pareto_target_flop,
            # )
            # fig_pareto_all.savefig(plot_output_dir / "pareto_front_all.png", dpi=300, bbox_inches="tight")
            # fig_pareto_with.savefig(plot_output_dir / "pareto_front_with_priors.png", dpi=300, bbox_inches="tight")
            # fig_pareto_without.savefig(plot_output_dir / "pareto_front_without_priors.png", dpi=300, bbox_inches="tight")
            # plt.close(fig_pareto_all)
            # plt.close(fig_pareto_with)
            # plt.close(fig_pareto_without)
            # print(f"    ✓ Saved: {plot_output_dir / 'pareto_front_all.png'}")
            # print(f"    ✓ Saved: {plot_output_dir / 'pareto_front_with_priors.png'}")
            # print(f"    ✓ Saved: {plot_output_dir / 'pareto_front_without_priors.png'}")

            prediction_flop_range = (7e17 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 5e19, get_benchmark(benchmark_name, False).max_evaluation_flops * 0.9 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1))
            target_flop = get_benchmark(benchmark_name, False).max_flops * 0.9 * (1e9 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 1)
            # for predictor_type in LossPredictorTypes:
            #     predictor_slug = predictor_type.value.lower().replace(" ", "_")
            #     print(f"  Plotting {predictor_type.value} predicted loss at FLOP: {target_flop:.2e} over cumulative FLOPs...")
            #     (fig_pred_all, _), (fig_pred_with, _), (fig_pred_without, _) = plot_predicted_loss_over_flops(
            #         base_path=results_dir / benchmark_name,
            #         optimizers={k: [40] for k in optimizers_dict.keys()},
            #         target_flop=target_flop,
            #         predictor_type=predictor_type,
            #         key_to_extract="cost",
            #         loss_key=loss_key,
            #         title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
            #         flop_range=prediction_flop_range,
            #         true_pareto_info=true_pareto_info,
            #     )
            #     fig_pred_all.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_all.png", dpi=300, bbox_inches="tight")
            #     fig_pred_with.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_with_priors.png", dpi=300, bbox_inches="tight")
            #     fig_pred_without.savefig(plot_output_dir / f"{predictor_slug}_predicted_loss_without_priors.png", dpi=300, bbox_inches="tight")
            #     plt.close(fig_pred_all)
            #     plt.close(fig_pred_with)
            #     plt.close(fig_pred_without)
            #     print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_all.png'}")
            #     print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_with_priors.png'}")
            #     print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_loss_without_priors.png'}")

            for predictor_type in [ScaleParamPredictorTypes.CHINCHILLA3]:
                predictor_slug = predictor_type.value.lower().replace(" ", "_")
                print(f"  Plotting {predictor_type.value} predicted n_param at FLOP: {target_flop:.2e} over cumulative FLOPs...")
                (fig_nparam_all, _), (fig_nparam_with, _), (fig_nparam_without, _) = plot_predicted_n_param_over_flops(
                    base_path=results_dir / benchmark_name,
                    optimizers={k: [40] for k in optimizers_dict.keys()},
                    target_flop=target_flop,
                    predictor_type=predictor_type,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    title=f"{BENCHMARK_TITLES[benchmark_name]} ({predictor_type.value})",
                    flop_range=prediction_flop_range,
                    true_pareto_info=true_pareto_info,
                )
                fig_nparam_all.savefig(plot_output_dir / f"{predictor_slug}_predicted_n_param_all_{target_flop:.2e}.png", dpi=300, bbox_inches="tight")
                fig_nparam_with.savefig(plot_output_dir / f"{predictor_slug}_predicted_n_param_with_priors_{target_flop:.2e}.png", dpi=300, bbox_inches="tight")
                fig_nparam_without.savefig(plot_output_dir / f"{predictor_slug}_predicted_n_param_without_priors_{target_flop:.2e}.png", dpi=300, bbox_inches="tight")
                plt.close(fig_nparam_all)
                plt.close(fig_nparam_with)
                plt.close(fig_nparam_without)
                print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_n_param_all_{target_flop:.2e}.png'}")
                print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_n_param_with_priors_{target_flop:.2e}.png'}")
                print(f"    ✓ Saved: {plot_output_dir / f'{predictor_slug}_predicted_n_param_without_priors_{target_flop:.2e}.png'}")

            # from src.hp_predictor import HPPredictorTypes

            # for hp_predictor_type in [HPPredictorTypes.LINEAR_REGRESSION]:
            #     hp_predictor_slug = hp_predictor_type.value.lower().replace(" ", "_")
            #     print(f"  Building {hp_predictor_type.value} predicted HP table at FLOP: {target_flop:.2e}...")
            #     fig_hp_all, _ = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
            #         base_path=results_dir / benchmark_name,
            #         optimizers={k: [40] for k in optimizers_dict.keys()},
            #         space=convert_neps_to_classic_search_space(benchmark_cfg.space),
            #         evaluate_pipeline=benchmark_cfg.evaluate_pipeline,
            #         true_pareto_info=true_pareto_info,
            #         target_flop=target_flop,
            #         flop_range=prediction_flop_range,
            #         predictor_type=hp_predictor_type,
            #         key_to_extract="cost",
            #         loss_key=loss_key,
            #         title=f"{BENCHMARK_TITLES[benchmark_name]} ({hp_predictor_type.value})",
            #         return_split=False,
            #     ))
            #     fig_hp_all.savefig(plot_output_dir / f"{hp_predictor_slug}_predicted_hp_table_all_{target_flop:.2e}.png", dpi=300, bbox_inches="tight")
            #     plt.close(fig_hp_all)
            #     print(f"    ✓ Saved: {plot_output_dir / f'{hp_predictor_slug}_predicted_hp_table_all_{target_flop:.2e}.png'}")
            
    raise Exception("not ready")
    print(f"\n{'='*60}")
    print("Generating composite loss plots across all benchmarks...")
    print(f"{'='*60}")
    (fig_loss_comp_all, _), (fig_loss_comp_with, _), (fig_loss_comp_without, _) = plot_loss_composite_across_benchmarks(
        base_path=results_dir,
        benchmarks=benchmarks,
        optimizers=optimizers_dict,
        key_to_extract="cost",
    )
    
    # Save composite loss plots to main plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_loss_comp_all.savefig(plots_dir / "loss_composite_all.png", dpi=300, bbox_inches="tight")
    fig_loss_comp_with.savefig(plots_dir / "loss_composite_with_priors.png", dpi=300, bbox_inches="tight")
    fig_loss_comp_without.savefig(plots_dir / "loss_composite_without_priors.png", dpi=300, bbox_inches="tight")
    plt.close(fig_loss_comp_all)
    plt.close(fig_loss_comp_with)
    plt.close(fig_loss_comp_without)
    print(f"✓ Saved: {plots_dir / 'loss_composite_all.png'}")
    print(f"✓ Saved: {plots_dir / 'loss_composite_with_priors.png'}")
    print(f"✓ Saved: {plots_dir / 'loss_composite_without_priors.png'}")

   
    # Plot: Composite hypervolume plots (all benchmarks + aggregate in single figures)
    print(f"\n{'='*60}")
    print("Generating composite hypervolume plots across all benchmarks...")
    print(f"{'='*60}")
    try:
        (fig_hv_comp_all, _), (fig_hv_comp_with, _), (fig_hv_comp_without, _) = plot_hypervolume_composite_across_benchmarks(
            base_path=results_dir,
            benchmarks=benchmarks,
            optimizers=optimizers_dict,
            key_to_extract="cost",
        )
        
        # Save composite hypervolume plots to main plots directory
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_hv_comp_all.savefig(plots_dir / "hypervolume_composite_all.png", dpi=300, bbox_inches="tight")
        fig_hv_comp_with.savefig(plots_dir / "hypervolume_composite_with_priors.png", dpi=300, bbox_inches="tight")
        fig_hv_comp_without.savefig(plots_dir / "hypervolume_composite_without_priors.png", dpi=300, bbox_inches="tight")
        plt.close(fig_hv_comp_all)
        plt.close(fig_hv_comp_with)
        plt.close(fig_hv_comp_without)
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_all.png'}")
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_with_priors.png'}")
        print(f"✓ Saved: {plots_dir / 'hypervolume_composite_without_priors.png'}")
    except Exception as e:
        print(f"✗ Error plotting composite hypervolume: {e}")
        import traceback
        traceback.print_exc()
                
    # Plot 4: Parallel coordinate plot showing rankings across all benchmarks
    print(f"\n{'='*60}")
    print("Generating parallel coordinate plot across all benchmarks...")
    print(f"{'='*60}")
    for ranking_metric in ["incumbent", "hypervolume"]:
        print(f"\n  Plotting parallel coordinates for {ranking_metric} rankings...")
        try:
            (fig_parallel_all, _), (fig_parallel_with, _), (fig_parallel_without, _) = plot_optimizer_rankings_across_benchmarks(
                base_path=results_dir,
                benchmarks=benchmarks,
                optimizers=optimizers_dict,
                key_to_extract="cost",
                title="Optimizer Rankings Across Benchmarks",
                ranking_metric=ranking_metric,
            )
            
            # Save parallel coordinate plot to main plots directory
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_parallel_all.savefig(plots_dir / f"optimizer_rankings_parallel_{ranking_metric}_all.png", dpi=300, bbox_inches="tight")
            fig_parallel_with.savefig(plots_dir / f"optimizer_rankings_parallel_{ranking_metric}_with_priors.png", dpi=300, bbox_inches="tight")
            fig_parallel_without.savefig(plots_dir / f"optimizer_rankings_parallel_{ranking_metric}_without_priors.png", dpi=300, bbox_inches="tight")
            plt.close(fig_parallel_all)
            plt.close(fig_parallel_with)
            plt.close(fig_parallel_without)
            print(f"✓ Saved: {plots_dir / f'optimizer_rankings_parallel_{ranking_metric}_all.png'}")
            print(f"✓ Saved: {plots_dir / f'optimizer_rankings_parallel_{ranking_metric}_with_priors.png'}")
            print(f"✓ Saved: {plots_dir / f'optimizer_rankings_parallel_{ranking_metric}_without_priors.png'}")
        except Exception as e:
            print(f"✗ Error plotting parallel coordinates: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Plot generation completed!")
    print(f"Output directory: {results_dir / 'plots'}")
    print(f"{'='*60}")
