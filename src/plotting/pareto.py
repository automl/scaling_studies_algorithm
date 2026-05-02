from pathlib import Path
import json
from collections import defaultdict
import neps
import numpy as np
from neps.state.trial import State
from src.benchmarks.llm_space import LLM_TARGETS
from src.benchmarks.openclip_space import OPENCLIP_TARGETS
from src.benchmarks.configs import BenchmarkEnum

def find_pareto_optimal_indices(losses, costs):
    """Find indices of Pareto optimal points (minimize both loss and costs).
    
    A point is Pareto optimal if no other point is better in all objectives.
    """
    # Filter out indices with None or NaN values
    valid_indices = [i for i in range(len(losses)) 
                     if losses[i] is not None and not (isinstance(losses[i], float) and np.isnan(losses[i]))
                     and costs[i] is not None and not (isinstance(costs[i], float) and np.isnan(costs[i]))]
    
    pareto_indices = []
    
    for i in valid_indices:
        is_dominated = False
        for j in valid_indices:
            if i == j:
                continue
            # Point j dominates point i if j is better in both objectives (lower loss and costs)
            if (losses[j] < losses[i] or costs[j] < costs[i]) and (losses[j] <= losses[i] and costs[j] <= costs[i]):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices

def plot_pareto_front(losses, costs, save_dir: Path = None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(costs, losses, c='red', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cost (FLOPs)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Pareto Front: Loss vs Cost', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "pareto_front.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pareto front plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
def plot_hp_scale(hp_values, objectives, flops, params, hp_name: str, save_dir: Path = None, pareto_indices: list = None):
    """Plot hyperparameter vs loss and vs flops with parameter colorbar.
    
    Args:
        hp_values: List of hyperparameter values (all points)
        objectives: List of loss values (all points)
        flops: List of FLOP values (all points)
        params: List of parameter counts for coloring (all points)
        hp_name: Name of the hyperparameter
        save_dir: Directory to save plots. If provided, saves plots instead of showing them.
        pareto_indices: Optional list of indices for Pareto optimal points to highlight with black circles and fit lines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
    
    # Setup colormap and normalization based on objectives (linear scale)
    # Filter out None and NaN values
    valid_objectives = np.array([obj for obj in objectives if obj is not None and not np.isnan(obj)])
    norm = Normalize(vmin=valid_objectives.min().item(), vmax=valid_objectives.max().item())
    cmap = plt.colormaps['viridis']
    
    scatter = axes.scatter(flops, hp_values, c=objectives, cmap=cmap, norm=norm, s=50, alpha=0.7, label='All points')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('FLOPs', fontsize=12)
    axes.set_ylabel(hp_name, fontsize=12)
    axes.set_title(f'FLOPs vs {hp_name}', fontsize=12)
    axes.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight Pareto optimal points and fit lines
    if pareto_indices is not None and len(pareto_indices) > 0:
        pareto_hp = np.array([hp_values[i] for i in pareto_indices])
        pareto_objectives = np.array([objectives[i] for i in pareto_indices])
        pareto_flops = np.array([flops[i] for i in pareto_indices])
        
        # Add black circles around Pareto optimal points
        # ax1.scatter(pareto_hp, pareto_objectives, facecolors='none', edgecolors='black', s=150, linewidth=2, label='Pareto optimal')
        axes.scatter(pareto_flops, pareto_hp, facecolors='none', edgecolors='black', s=150, linewidth=2, label='Pareto optimal')
        
        # Fit linear regression in log-log space for Pareto optimal points
        if len(pareto_indices) >= 2:
            # For Loss vs HP
            log_hp = np.log(pareto_hp)
            # log_objectives = np.log(pareto_objectives)
            # coeffs_loss = np.polyfit(log_hp, log_objectives, 1)
            # slope_loss = coeffs_loss[0]
            # intercept_loss = coeffs_loss[1]
            
            # Generate fitted line for Loss
            hp_range_log = np.linspace(np.log(pareto_hp.min()), np.log(pareto_hp.max()), 100)
            flops_range_log = np.linspace(np.log(pareto_flops.min()), np.log(pareto_flops.max()), 100)
            hp_range = np.exp(hp_range_log)
            # fitted_loss = np.exp(slope_loss * hp_range_log + intercept_loss)
            # ax1.plot(hp_range, fitted_loss, 'r--', linewidth=2.5, alpha=0.8, label=f'Linear fit (slope={slope_loss:.3f})')
            
            # For FLOPs vs HP
            log_flops = np.log(pareto_flops)
            coeffs_flops = np.polyfit(log_flops, log_hp, 1)
            slope_flops = coeffs_flops[0]
            intercept_flops = coeffs_flops[1]
            
            # Generate fitted line for FLOPs (correct: apply fit to log_flops range to get log_hp)
            fitted_hp_log = slope_flops * flops_range_log + intercept_flops
            fitted_hp = np.exp(fitted_hp_log)
            flops_range = np.exp(flops_range_log)
            axes.plot(flops_range, fitted_hp, 'r--', linewidth=2.5, alpha=0.8, label=f'Linear fit (slope={slope_flops:.3f})')
        
        # ax1.legend(loc='best', fontsize=10)
        axes.legend(loc='best', fontsize=10)
    
    # Add shared colorbar for parameters (positioned horizontally below plots)
    cbar = plt.colorbar(scatter, label='Loss',)
    
    plt.tight_layout()
    
    if save_dir:
        # Create save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot as PNG
        save_path = save_dir / f"{hp_name}_vs_flops.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_aspect_ratio_vs_loss(aspect_ratios, losses, save_dir: Path = None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(aspect_ratios, losses, c='blue', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Aspect Ratio (d_model / num_layers)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Aspect Ratio', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "aspect_ratio_vs_loss.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Aspect ratio plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # path = Path("scripts/results/openclip_grid_search/grid_search/seed_40")
    bench = BenchmarkEnum.OPENCLIP
    pathes = {
        BenchmarkEnum.OPENCLIP:Path("/work/dlclarge1/alipourn-scan_suite/openclip/carbs"),
        # BenchmarkEnum.LLM: Path("/work/dlclarge1/alipourn-scan_suite/llm/grid_search")
    }
    path = pathes[bench]
    targets = ["val_loss"]
    losses = defaultdict(list)
    costs = []
    confs = []
    params = []
    datapoints = []
    max_cost: float = -1.0
    seed_dir = path / "seed_40"
    _fulldf, _summary = neps.status(seed_dir, print_summary=False)
    if _fulldf.empty:
        raise ValueError(f"No trials found in {seed_dir}")

    _fulldf = _fulldf.sort_values("time_sampled")

    def get_cost(idx: str | int) -> float:
        row = _fulldf.loc[idx]
        return float(row['cost'])

    for config_id, config_result in _fulldf.iterrows():
        if config_result["state"] != State.SUCCESS:
            continue

        cost = get_cost(config_id)
        conf = {key: value for key, value in config_result.items() if key.startswith("config.")}
        for target in targets:
            # try:
            #     target = target.value
            # except AttributeError:
            #     pass
            loss_key = f"extra.predictions.{target}.mean"
            try:
                if '_acc' in target or 'recall' in target:
                    loss = 1.0 - float(config_result["objective_to_minimize"])
                else:
                    loss = float(config_result["objective_to_minimize"])
            except TypeError:
                print(f"Skipping config {config_id} due to invalid loss value: {config_result[loss_key]}")
                continue
            losses[target].append(loss)
        costs.append(cost)
        confs.append(conf)
        params.append(float(config_result["extra.n_param"]))
        datapoints.append(float(config_result["extra.n_data"]))

    # Keep all data for plotting
    all_losses = losses["val_loss"] if "val_loss" in losses else losses["valid_loss"] if "valid_loss" in losses else []
    all_costs = costs
    all_confs = confs
    all_params = params
    all_datapoints = datapoints
    
    
    save_dir = path / "plotting"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find Pareto optimal points (but keep all data)
    
    for target in targets:
        print(f"Target: {target}")
        target_losses = losses[target]
        pareto_indices = find_pareto_optimal_indices(target_losses, costs)
        print(f"Found {len(pareto_indices)} Pareto optimal points out of {len(target_losses)} total points")
            
        losses_pareto = [target_losses[i] for i in pareto_indices]
        costs_pareto = [costs[i] for i in pareto_indices]
        confs_pareto = [confs[i] for i in pareto_indices]
        params_pareto = [params[i] for i in pareto_indices]
        datapoints_pareto = [datapoints[i] for i in pareto_indices]

        assert len(losses_pareto) == len(costs_pareto) == len(confs_pareto) == len(params_pareto) == len(datapoints_pareto)
        plot_pareto_front(losses_pareto, costs_pareto, save_dir=save_dir / target)

    # Save pareto optimal points to JSON
    pareto_data = [
        {
            "loss": loss,
            "cost": cost,
            "config": conf
        }
        for loss, cost, conf in zip(losses_pareto, costs_pareto, confs_pareto)
    ]
    
    json_path = save_dir / "pareto_optimal_points.json"
    with open(json_path, "w") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"Pareto optimal points saved to {json_path}")
