from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import neps
from neps.state.trial import State
from src.plotting.plot_styles import OPTIMIZER_COLORS


def get_optimizer_color(opt_name: str) -> str:
    """Get color for optimizer, with fallback to default."""
    return OPTIMIZER_COLORS.get(opt_name, OPTIMIZER_COLORS["default"])


def process_seed(
    *,
    path: str | Path,
    seed: str | int | None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
) -> tuple[list[float], list[float], float]:
    path = Path(path)
    if seed is not None:
        path = path / f"seed_{seed}"

    _fulldf, _summary = neps.status(path, print_summary=False)
    if _fulldf.empty:
        raise ValueError(f"No trials found in {path}")

    _fulldf = _fulldf.sort_values("time_sampled")

    def get_cost(idx: str | int) -> float:
        row = _fulldf.loc[idx]
        if key_to_extract and key_to_extract in row:
            return float(row[key_to_extract])
        return 1.0

    losses = []
    costs = []

    max_cost: float = -1.0

    for config_id, config_result in _fulldf.iterrows():
        if config_result["state"] != State.SUCCESS:
            continue

        cost = get_cost(config_id)

        loss = config_result[loss_key]
        try:
            loss = float(loss)
            # Keep NaN/inf values as NaN for post-processing
            if np.isinf(loss):
                loss = np.nan
            elif '_acc' in loss_key or 'recall' in loss_key:
                loss = 1.0 - loss
        except (ValueError, TypeError): # Handle NaN, NA, and inf values
            loss = np.nan
        losses.append(loss)
        costs.append(cost)
    
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
    return list(np.minimum.accumulate(losses)), cumsum_costs, max_cost


def process_optimizer_per_seed(
    *,
    base_path: str | Path,
    optimizer_name: str,
    seeds: list[int | str],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Process all seeds for an optimizer, keeping per-seed data.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizer_name: Name of the optimizer directory
        seeds: List of seed values to process
        key_to_extract: Key to extract from results as cost (typically 'flopcount')
        log_scale: Whether to interpolate on log scale for flop counts
    
    Returns:
        flop_grid: Common flop count grid (x-axis)
        per_seed_losses_dict: Dict mapping seed -> interpolated losses
        per_seed_costs_dict: Dict mapping seed -> per-sample cumulative costs
    """
    base_path = Path(base_path)
    optimizer_path = base_path / optimizer_name
    
    all_seeds_data = []
    valid_seeds = []
    
    # Process each seed
    for seed in seeds:
        try:
            losses, costs, max_cost = process_seed(
                path=optimizer_path,
                seed=seed,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
            )
            all_seeds_data.append({"losses": losses, "costs": costs, "max_cost": max_cost})
            valid_seeds.append(seed)
        except Exception as e:
            print(f"Warning: Could not process {optimizer_name} seed {seed}: {e}")
            continue
    
    if not all_seeds_data:
        raise ValueError(f"No valid seeds found for optimizer {optimizer_name}")
    
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
    
    per_seed_losses = {}
    per_seed_costs = {}
    
    for seed, seed_data in zip(valid_seeds, all_seeds_data):
        costs = np.array(seed_data["costs"])
        losses = np.array(seed_data["losses"])
        
        # Interpolate to common grid
        interp_losses = np.interp(flop_grid, costs, losses, left=losses[0], right=losses[-1])
        per_seed_losses[seed] = interp_losses
        per_seed_costs[seed] = costs
    
    return flop_grid, per_seed_losses, per_seed_costs


def process_optimizer(
    *,
    base_path: str | Path,
    optimizer_name: str,
    seeds: list[int | str],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flop_grid, per_seed_losses, _ = process_optimizer_per_seed(
        base_path=base_path,
        optimizer_name=optimizer_name,
        seeds=seeds,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
    )
    
    # Stack per-seed losses and calculate mean/standard error
    interpolated_losses_array = np.array(list(per_seed_losses.values()))
    
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
    optimizers: dict[str, list[int]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Loss Over Optimization",
    max_flops: float | None = None,
):
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
        max_flops: Maximum FLOPs to display on x-axis (trims data beyond this)
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Process all optimizers
    for opt_name, seeds in optimizers.items():
        color = get_optimizer_color(opt_name)
        
        flop_counts, mean_losses, std_losses = process_optimizer(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
        )
        
        # Trim data to max_flops if specified
        if max_flops is not None:
            mask = flop_counts <= max_flops
            flop_counts = flop_counts[mask]
            mean_losses = mean_losses[mask]
            std_losses = std_losses[mask]
        
        # Plot mean loss
        ax.plot(
            flop_counts,
            mean_losses,
            label=opt_name,
            color=color,
            linewidth=2.5,
        )
        
        # Add confidence band (standard error)
        ax.fill_between(
            flop_counts,
            mean_losses - std_losses,
            mean_losses + std_losses,
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel("FLOPs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Place legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(4, len(optimizers)), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, ax


def calculate_rankings(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
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
        )
        all_per_seed_data[opt_name] = {
            "flop_grid": flop_grid,
            "per_seed_losses": per_seed_losses,
        }
    
    # Calculate rankings per seed, then aggregate
    rankings = {}
    
    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        
        # Calculate ranking for each seed
        seed_rankings_list = []
        
        for seed, losses in per_seed_losses.items():
            seed_ranks = []
            
            for i in range(len(flop_grid)):
                # Get losses from all optimizers at this flop index
                losses_at_idx = []
                for other_opt in all_per_seed_data.keys():
                    try:
                        other_losses = all_per_seed_data[other_opt]["per_seed_losses"][seed]
                    except KeyError as e:
                        print(f"other_opt {other_opt} does not have seed {seed}, skipping for ranking at index {i}")
                        raise e
                    if i < len(other_losses):
                        losses_at_idx.append(other_losses[i])
                
                # Rank this optimizer (lower loss = rank 1)
                if losses_at_idx:
                    # Filter out NaN values for ranking
                    valid_losses_at_idx = [l for l in losses_at_idx if not np.isnan(l)]
                    
                    current_loss = losses[i]
                    if np.isnan(current_loss):
                        # If current loss is NaN, assign worst rank
                        rank = len(valid_losses_at_idx) + 1
                    elif valid_losses_at_idx:
                        # Count how many valid losses are better (lower)
                        rank = sum(1 for l in valid_losses_at_idx if l < current_loss) + 1
                    else:
                        # All other losses are NaN
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


def calculate_hypervolumes(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
) -> dict[str, dict]:
    """
    Calculate hypervolume per-seed, then aggregate with mean/std.
    
    Reference point is dynamically set to (max_cost, max_loss) across all optimizers.
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        log_scale: Whether to use log scale
    
    Returns:
        hvs_dict: Dict with optimizer names as keys
                Values: dicts with 'flop_counts', 'mean_hvs', 'std_hvs'
    """
    from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
    
    # First pass: collect all costs and losses to determine reference point
    max_cost = -1.0
    max_loss = -1.0
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
        
        # Track max cost and loss
        for seed, losses in per_seed_losses.items():
            max_cost = max(max_cost, flop_grid[-1])  # Last element is max flop
            max_loss = max(max_loss, float(np.max(losses)))
    
    # Use dynamic reference point (max_cost, max_loss)
    ref_point = (max_cost, max_loss)
    
    hvs_dict = {}
    
    # Second pass: calculate hypervolume with dynamic reference point
    for opt_name, data in all_per_seed_data.items():
        flop_grid = data["flop_grid"]
        per_seed_losses = data["per_seed_losses"]
        
        # Calculate hypervolume per seed
        seed_hvs_list = []
        
        for seed, losses in per_seed_losses.items():
            hvs = ScalingLawGuidedOptimizer._compute_hypervolume_over_accumulated_flops(
                losses.tolist(), 
                flop_grid.tolist(),
                ref_point=ref_point
            )
            seed_hvs_list.append(np.array(hvs))
        
        # Aggregate across seeds
        seed_hvs_array = np.array(seed_hvs_list)
        mean_hvs = np.nanmean(seed_hvs_array, axis=0)
        std_hvs = np.nanstd(seed_hvs_array, axis=0)
        
        # Count valid (non-NaN) values for each position
        valid_counts = np.sum(~np.isnan(seed_hvs_array), axis=0)
        # Avoid division by zero
        valid_counts = np.maximum(valid_counts, 1)
        se_hvs = std_hvs / np.sqrt(valid_counts)
        
        hvs_dict[opt_name] = {
            "flop_counts": flop_grid,
            "mean_hvs": mean_hvs.astype(float),
            "std_hvs": se_hvs.astype(float),
        }
    
    return hvs_dict


def plot_hypervolume_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Hypervolume Over Optimization",
    max_flops: float | None = None,
):
    """
    Plot hypervolume with mean and std bands (calculated per-seed).
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        title: Plot title
        max_flops: Maximum FLOPs to display on x-axis (trims data beyond this)
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    hvs_data = calculate_hypervolumes(
        base_path=base_path,
        optimizers=optimizers,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
    )
    
    for opt_name, data in hvs_data.items():
        color = get_optimizer_color(opt_name)
        flop_counts = data["flop_counts"]
        mean_hvs = data["mean_hvs"]
        std_hvs = data["std_hvs"]
        
        # Trim data to max_flops if specified
        if max_flops is not None:
            mask = flop_counts <= max_flops
            flop_counts = flop_counts[mask]
            mean_hvs = mean_hvs[mask]
            std_hvs = std_hvs[mask]
        
        # Plot mean hypervolume
        ax.plot(
            flop_counts,
            mean_hvs,
            label=opt_name,
            color=color,
            linewidth=2.5,
        )
        
        # Add std band
        ax.fill_between(
            flop_counts,
            mean_hvs - std_hvs,
            mean_hvs + std_hvs,
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel("FLOPs", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")  # Linear scale for y-axis to show variations clearly
    
    # Place legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(4, len(optimizers)), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, ax


def plot_relative_ranking_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Optimizer Rankings Over Optimization",
    max_flops: float | None = None,
):
    """
    Plot optimizer rankings (calculated per-seed, then aggregated with mean/std).
    
    Args:
        base_path: Base path where optimizer directories are located
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        title: Plot title
        max_flops: Maximum FLOPs to display on x-axis (trims data beyond this)
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    rankings = calculate_rankings(
        base_path=base_path,
        optimizers=optimizers,
        key_to_extract=key_to_extract,
        loss_key=loss_key,
        log_scale=log_scale,
    )
    
    for opt_name, rank_data in rankings.items():
        color = get_optimizer_color(opt_name)
        flop_counts = rank_data["flop_counts"]
        mean_ranks = rank_data["mean_ranks"]
        std_ranks = rank_data["std_ranks"]
        
        # Trim data to max_flops if specified
        if max_flops is not None:
            mask = flop_counts <= max_flops
            flop_counts = flop_counts[mask]
            mean_ranks = mean_ranks[mask]
            std_ranks = std_ranks[mask]
        
        # Plot mean ranking
        ax.plot(
            flop_counts,
            mean_ranks,
            label=opt_name,
            color=color,
            linewidth=2.5,
        )
        
        # Add std band
        ax.fill_between(
            flop_counts,
            mean_ranks - std_ranks,
            mean_ranks + std_ranks,
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel("FLOPs", fontsize=12)
    ax.set_ylabel("Relative Rank (1=best)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert so rank 1 is at top
    ax.set_xscale("log")
    
    # Set y-axis to linear scale with rank labels
    num_optimizers = len(optimizers)
    ax.set_ylim(num_optimizers + 0.5, 0.5)
    ax.set_yticks(range(1, num_optimizers + 1))
    ax.set_yticklabels(range(1, num_optimizers + 1))
    
    # Place legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(4, len(optimizers)), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, ax


def plot_optimizer_rankings_across_benchmarks(
    *,
    base_path: str | Path,
    benchmarks: dict[str, any],
    optimizers: dict[str, list[int | str]],
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Optimizer Rankings Across Benchmarks",
):
    """
    Plot mean optimizer rankings aggregated across all benchmarks with std bands.
    
    For each benchmark, calculate ranking curves. Then for each optimizer, aggregate
    their rankings across all benchmarks and plot mean ± std with confidence bands.
    
    Args:
        base_path: Base path where benchmark optimizer directories are located
        benchmarks: Dict mapping benchmark name to benchmark config
        optimizers: Dict mapping optimizer name to list of seeds
        key_to_extract: Key to extract from results as cost
        figsize: Figure size
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
        ax: Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    benchmark_names = list(benchmarks.keys())
    optimizer_names = list(optimizers.keys())
    
    # Collect all ranking curves for each optimizer across benchmarks
    optimizer_all_ranks = {opt: [] for opt in optimizer_names}
    optimizer_all_flop_counts = {opt: [] for opt in optimizer_names}
    
    for benchmark_name in benchmark_names:
        try:
            # Calculate rankings for this benchmark
            rankings = calculate_rankings(
                base_path=base_path,
                optimizers=optimizers,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                log_scale=False,
            )
            
            # Collect ranking curves for each optimizer from this benchmark
            for opt_name in optimizer_names:
                if opt_name in rankings:
                    seed_ranks = rankings[opt_name]["seed_ranks"]
                    flop_counts = rankings[opt_name]["flop_counts"]
                    optimizer_all_ranks[opt_name].extend(seed_ranks)
                    optimizer_all_flop_counts[opt_name].append(flop_counts)
        
        except Exception as e:
            print(f"Warning: Could not calculate rankings for {benchmark_name}: {e}")
            continue
    
    # For each optimizer, aggregate ranks across benchmarks
    for opt_name in optimizer_names:
        if not optimizer_all_ranks[opt_name]:
            print(f"Warning: No ranking data for optimizer {opt_name}")
            continue
        
        color = get_optimizer_color(opt_name)
        
        # Stack all ranking curves from different benchmarks
        all_ranks_array = np.array(optimizer_all_ranks[opt_name])
        
        # Calculate mean and std across benchmarks
        mean_ranks = np.mean(all_ranks_array, axis=0)
        std_ranks = np.std(all_ranks_array, axis=0)
        se_ranks = std_ranks / np.sqrt(len(optimizer_all_ranks[opt_name]))
        
        # Use flop counts from first benchmark as reference (they should be similar after interpolation)
        flop_counts = optimizer_all_flop_counts[opt_name][0]
        
        # Plot mean ranking
        ax.plot(
            flop_counts,
            mean_ranks,
            label=opt_name,
            color=color,
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
    
    ax.set_xlabel("FLOPs", fontsize=12)
    ax.set_ylabel("Relative Rank (1=best)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert so rank 1 is at top
    
    # Place legend below
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(4, len(optimizers)), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, ax


if __name__ == "__main__":
    """
    Plot loss, hypervolume, and rankings for all benchmarks in configs.
    
    Per-seed data is kept throughout calculations, then aggregated with mean/std.
    """
    from pathlib import Path
    
    results_dir = Path("/work/dlclarge1/alipourn-benchmark_res")
    # loss_key = "objective_to_minimize"
    
    
    # Seeds to process
    
    # Optimizers available (map optimizer name to seeds)
    # all_seeds = list(set(range(40, 47)))
    all_seeds = [40]
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
    
    # for benchmark_name in [BenchmarkEnum.LLM, BenchmarkEnum.OPENCLIP]:
    for benchmark_name in [BenchmarkEnum.OPENCLIP]:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {benchmark_name}")
        print(f"{'='*60}")
        targets = {BenchmarkEnum.LLM: LLM_TARGETS, BenchmarkEnum.OPENCLIP: OPENCLIP_TARGETS}
        for target in targets[benchmark_name]:
        # for target in ["arc_challenge_acc"]:
            try:
                target = target.value
            except AttributeError:
                pass
            loss_key = f"extra.predictions.{target}.mean"
            try:
                # Create output directory
                plot_output_dir = results_dir / benchmark_name / "plots" / target
                plot_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Plot 1: Loss over FLOPs
                print(f"  Plotting loss over FLOPs...")
                # try:
                from src.benchmarks.configs import get_benchmark
                max_flops_display = get_benchmark(benchmark_name, False).max_evaluation_flops * 0.6
                fig_loss, ax_loss = plot_optimizer_loss(
                    base_path=results_dir / benchmark_name,
                    optimizers=optimizers_dict,
                    key_to_extract="cost",
                    log_scale=False,
                    loss_key=loss_key,
                    title=f"{benchmark_name} - Loss Over Optimization",
                    max_flops=max_flops_display,
                )
                fig_loss.savefig(plot_output_dir / "loss.png", dpi=300, bbox_inches="tight")
                plt.close(fig_loss)
                print(f"    ✓ Saved: {plot_output_dir / 'loss.png'}")
                # except Exception as e:
                #     print(f"    ✗ Error plotting loss: {e}")
                
                # Plot 2: Hypervolume
                print(f"  Plotting hypervolume...")
                try:
                    fig_hv, ax_hv = plot_hypervolume_over_flops(
                        base_path=results_dir / benchmark_name,
                        optimizers=optimizers_dict,
                        key_to_extract="cost",
                        log_scale=False,
                        loss_key=loss_key,
                        title=f"{benchmark_name} - Hypervolume",
                        max_flops=max_flops_display,
                    )
                    fig_hv.savefig(plot_output_dir / "hypervolume.png", dpi=300, bbox_inches="tight")
                    plt.close(fig_hv)
                    print(f"    ✓ Saved: {plot_output_dir / 'hypervolume.png'}")
                except Exception as e:
                    print(f"    ✗ Error plotting hypervolume: {e}")
                
                # Plot 3: Rankings
                print(f"  Plotting rankings...")
                # try:
                fig_rank, ax_rank = plot_relative_ranking_over_flops(
                    base_path=results_dir / benchmark_name,
                    optimizers=optimizers_dict,
                    key_to_extract="cost",
                    log_scale=False,
                    loss_key=loss_key,
                    title=f"{benchmark_name} - Optimizer Rankings",
                    max_flops=max_flops_display,
                )
                fig_rank.savefig(plot_output_dir / "rankings.png", dpi=300, bbox_inches="tight")
                plt.close(fig_rank)
                print(f"    ✓ Saved: {plot_output_dir / 'rankings.png'}")
                # except Exception as e:
                #     print(f"    ✗ Error plotting rankings: {e}")
                
                print(f"  ✓ All plots generated for {benchmark_name}")
                
            except Exception as e:
                print(f"  ✗ Error generating plots for {benchmark_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Plot 4: Parallel coordinate plot showing rankings across all benchmarks
    print(f"\n{'='*60}")
    print("Generating parallel coordinate plot across all benchmarks...")
    print(f"{'='*60}")
    # try:
    #     fig_parallel, ax_parallel = plot_optimizer_rankings_across_benchmarks(
    #         base_path=results_dir,
    #         benchmarks=benchmarks_dict,
    #         optimizers=optimizers_dict,
    #         key_to_extract="cost",
    #         title="Optimizer Rankings Across Benchmarks",
    #     )
        
    #     # Save parallel coordinate plot to main plots directory
    #     plots_dir = results_dir / "plots"
    #     plots_dir.mkdir(parents=True, exist_ok=True)
    #     fig_parallel.savefig(plots_dir / "optimizer_rankings_parallel.png", dpi=300, bbox_inches="tight")
    #     plt.close(fig_parallel)
    #     print(f"✓ Saved: {plots_dir / 'optimizer_rankings_parallel.png'}")
    # except Exception as e:
    #     print(f"✗ Error plotting parallel coordinates: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Plot generation completed!")
    print(f"Output directory: {results_dir / 'plots'}")
    print(f"{'='*60}")
