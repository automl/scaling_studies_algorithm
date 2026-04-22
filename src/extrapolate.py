import json
import logging
import time
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from neps.optimizers.optimizer import AskFunction
from neps.runtime import _save_optimizer_artifacts

from neps.status.status import post_run_csv
from neps.state import NePSState
from neps.optimizers.scaling_law_guided import fit_pareto_front_loglog


from src.benchmarks.configs import BenchmarkEnum, get_benchmark
from src.baselines.configs import get_optimizer_wrapper, OptimizerEnum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_neps_state(root_directory: Path | str) -> tuple[Any, Any]:
    """Load NePSState and trials from a directory."""
    root_directory = Path(root_directory)

    if not root_directory.exists():
        raise FileNotFoundError(
            f"No neps state found at: {root_directory}\n"
            "Please provide a valid neps results directory."
        )
    try:
        state = NePSState.create_or_load(path=root_directory, load_only=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load neps state from: {root_directory}"
        ) from e

    space = state.lock_and_get_search_space()
    trials = state._trial_repo.latest(refresh_cache=True)

    if space is None:
        raise RuntimeError("Could not load search space from neps state.")

    return state, trials


def _load_neps_state_with_retry(root_directory: Path | str, max_retries: int = 10) -> tuple[Any, Any]:
    """Load NePSState with retry logic to handle concurrent file access.
    
    Args:
        root_directory: Path to neps study directory
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (state, trials)
        
    Raises:
        Exception: If all retries are exhausted
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to load neps state from {root_directory}")
            return _load_neps_state(root_directory)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Failed to load neps state (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
            continue
    
    raise RuntimeError(
        f"Failed to load neps state after {max_retries} attempts. Last error: {last_error}"
    )


def get_true_pareto_front(root_directory: Path | str) -> dict[str, list]:
    """Extract and return Pareto front data from a neps study."""
    _, trials = _load_neps_state_with_retry(root_directory)
    
    result = fit_pareto_front_loglog(trials, lc_aware=False)
    pareto_rows = result[5]
    
    return {
        "loss": [r[2] for r in pareto_rows],
        "flops": [r[3] for r in pareto_rows],
        "n_params": [float(r[0]) for r in pareto_rows],
        "n_data": [float(r[1]) for r in pareto_rows],
    }

def save_pareto_info(pareto_info: dict, output_dir: Path | str) -> None:
    """Save Pareto front information to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "pareto_info.json"
    with open(output_file, 'w') as f:
        json.dump(pareto_info, f, indent=2)
    logger.info(f"Saved pareto_info to {output_file}")

def load_pareto_info(output_dir: Path | str) -> dict[str, list] | None:
    """Load Pareto front information from JSON file if it exists."""
    output_dir = Path(output_dir)
    output_file = output_dir / "pareto_info.json"
    
    if output_file.exists():
        with open(output_file, 'r') as f:
            logger.info(f"Loaded pareto_info from {output_file}")
            return json.load(f)
    return None

def extrapolate_all(
    root_directories: list[Path] | list[str],
    optimizer_ask,  # Can be AskFunction or optimizer instance
    true_pareto_info: dict,
) -> list[dict[str, Any]]:
    """Extrapolate for multiple studies and return all results."""
    results = []
    for root_directory in root_directories:
        result = extrapolate(root_directory, optimizer_ask, true_pareto_info)
        if result is not None:
            results.append(result)
    return results

def create_aggregated_pareto_plots(
    root_directories: list[Path] | list[str],
    true_pareto_info: dict,
    summary_dir: Path | str,
) -> None:
    """Create aggregated param-flops and loss-flops plots across all studies.
    
    Args:
        root_directories: List of study directories
        true_pareto_info: True pareto front information
        summary_dir: Directory to save aggregated plots
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data from all studies
    all_flops = []
    all_params = []
    all_data = []
    all_loss = []
    study_labels = []
    
    for idx, root_directory in enumerate(root_directories):
        try:
            state, trials = _load_neps_state_with_retry(root_directory)
            
            # Extract pareto front data directly from trials using fit_pareto_front_loglog
            result = fit_pareto_front_loglog(trials, lc_aware=False)
            pareto_rows = result[5]
            
            flops = [r[3] for r in pareto_rows]
            params = [float(r[0]) for r in pareto_rows]
            data_pts = [float(r[1]) for r in pareto_rows]
            loss = [r[2] for r in pareto_rows]
            
            all_flops.extend(flops)
            all_params.extend(params)
            all_data.extend(data_pts)
            all_loss.extend(loss)
            study_labels.extend([f"Study {idx}"] * len(flops))
            
        except Exception as e:
            logger.warning(f"Failed to extract data from {root_directory}: {e}")
            continue
    
    if not all_flops:
        logger.warning("No data collected from any studies")
        return
    
    # Create aggregated FLOPs vs Params plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter1 = ax1.scatter(all_flops, all_params, c=range(len(all_flops)), cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('FLOPs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    ax1.set_title('Aggregated: FLOPs vs Parameters (All Studies)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter1, ax=ax1, label='Point Index')
    plt.tight_layout()
    fig1.savefig(summary_dir / "aggregated_flops_vs_params.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved aggregated flops vs params plot to {summary_dir / 'aggregated_flops_vs_params.png'}")
    plt.close(fig1)
    
    # Create aggregated FLOPs vs Loss plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter2 = ax2.scatter(all_flops, all_loss, c=range(len(all_flops)), cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('FLOPs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Aggregated: FLOPs vs Loss (All Studies)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter2, ax=ax2, label='Point Index')
    plt.tight_layout()
    fig2.savefig(summary_dir / "aggregated_flops_vs_loss.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved aggregated flops vs loss plot to {summary_dir / 'aggregated_flops_vs_loss.png'}")
    plt.close(fig2)
    
    # Create aggregated FLOPs vs Data plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    scatter3 = ax3.scatter(all_flops, all_data, c=range(len(all_flops)), cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('FLOPs', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Data Points', fontsize=12, fontweight='bold')
    ax3.set_title('Aggregated: FLOPs vs Data (All Studies)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter3, ax=ax3, label='Point Index')
    plt.tight_layout()
    fig3.savefig(summary_dir / "aggregated_flops_vs_data.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved aggregated flops vs data plot to {summary_dir / 'aggregated_flops_vs_data.png'}")
    plt.close(fig3)

def extrapolate(
    root_directory: Path | str,
    optimizer_ask,  # Can be AskFunction or optimizer instance with extrapolate method
    true_pareto_info: dict,
) -> dict[str, Any] | None:
    """Extrapolate best configuration using the optimizer."""
    state, trials = _load_neps_state_with_retry(root_directory)

    # Check if optimizer has extrapolate method
    if not hasattr(optimizer_ask, "extrapolate"):
        logger.info(
            f"Optimizer {type(optimizer_ask)} does not support extrapolation."
        )
        return None

    extrapolation_data = optimizer_ask.extrapolate(trials=trials, true_pareto_info=true_pareto_info)
    
    # Persist optimizer artifacts if available
    if hasattr(optimizer_ask, 'get_trial_artifacts'):
        try:
            artifacts = optimizer_ask.get_trial_artifacts(trials)
            if artifacts is not None:
                _save_optimizer_artifacts(artifacts, Path(root_directory) / "summary")
        except Exception as e:
            logger.error(f"Failed to persist optimizer artifacts: {e}", exc_info=True)
    
    post_run_csv(root_directory)
    return extrapolation_data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrapolate best configuration from a neps study."
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=[bench.value for bench in BenchmarkEnum],
        help='Name of the study/root_dir to be created with neps',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default="",
        choices=[opt.value for opt in OptimizerEnum],
        help='Name of the optimizer to be used',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for reproducability',
    )
    parser.add_argument(
        '--use-priors',
        action='store_true',
        help='Use prior for the sampling approach',
    )

    args = parser.parse_args()
    seed_range = list(range(40, 51)) if args.seed == 0 else [args.seed]
    optimizers = [
        OptimizerEnum.CHINCHILLA_APPROACH1, 
        OptimizerEnum.CHINCHILLA_APPROACH2,
        OptimizerEnum.CHINCHILLA_GEMSTONES, 
        OptimizerEnum.RANDOM_SEARCH,
        OptimizerEnum.CARBS,
        # OptimizerEnum.GRID_SEARCH
    ] if args.optimizer == "" else [OptimizerEnum(args.optimizer)]

    optimizer_suffix = "_with_priors" if args.use_priors else ""
    root_dirs = {}
    optimizer_root_dir_for_seeds = []
    for optimizer in optimizers:
        for seed in seed_range:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            optimizer_name = OptimizerEnum(optimizer)
            root_dir = (
                f"/work/dlclarge1/alipourn-benchmark_res/{args.benchmark}/"
                f"{optimizer_name}{optimizer_suffix}/seed_{seed}")
            optimizer_root_dir_for_seeds.append(root_dir)
            benchmark = get_benchmark(
                BenchmarkEnum(args.benchmark),
                lc_aware=False
                in [
                    OptimizerEnum.CHINCHILLA_APPROACH1.value,
                    OptimizerEnum.CHINCHILLA_GEMSTONES.value,
                ],
            )
            optimizer_ask, _ = get_optimizer_wrapper(
                pipeline_space=benchmark.space,
                evaluate_pipeline=benchmark.evaluate_pipeline,
                param_estimator=benchmark.params_estimator,
                seen_datapoints_estimator=benchmark.seen_datapoints_estimator,
                flops_estimator=benchmark.flops_estimator,
                aspect_ratio_fn=benchmark.aspect_ratio_fn,
                data_parameter_name=benchmark.data_parameter_name,
                # type: ignore
            )(
                optimizer_name,
                root_dir=root_dir,
                max_evaluation_flops=benchmark.max_evaluation_flops,
            )
            optimizer_ask = optimizer_ask.optimizer(benchmark.space)

            # Single seed extrapolation
            
            comprehensive_study_root_dir = (
                f"/work/dlclarge1/alipourn-benchmark_res/{args.benchmark}/"
                f"{OptimizerEnum.GRID_SEARCH}/seed_40"
            )
            summary_dir = Path(comprehensive_study_root_dir) / "summary"
            
            # Try to load pareto_info from JSON, or generate and save it
            pareto_info = load_pareto_info(summary_dir)
            if pareto_info is None:
                pareto_info = get_true_pareto_front(comprehensive_study_root_dir)
                save_pareto_info(pareto_info, summary_dir)
            
            extrapolation_data = extrapolate(
                root_dir, optimizer_ask, pareto_info, 
            )
            print(f"extrapolation_data {extrapolation_data}")
    # Aggregate heatmaps across all seeds
    # from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LogNorm
    
    # all_loss_heatmaps = []
    # all_params_heatmaps = []
    # spent_ratios_list_global = None
    
    # for root_dir in optimizer_root_dir_for_seeds:
    #     json_path = Path(root_dir) / "summary" / "extrapolation_results_all_ratios.json"
        
    #     if not json_path.exists():
    #         logger.warning(f"Extrapolation results not found at {json_path}")
    #         continue
        
    #     try:
    #         # Load the extrapolation results
    #         with open(json_path, 'r') as f:
    #             extrapolation_results_by_spent_ratio = json.load(f)
            
    #         logger.info(f"Loaded extrapolation results from {json_path}")
            
    #         # Store spent ratios from first seed
    #         if spent_ratios_list_global is None:
    #             spent_ratios_list_global = sorted([float(k) for k in extrapolation_results_by_spent_ratio.keys()])
            
    #         # Call the classmethod to get heatmaps
    #         loss_heatmap, params_heatmap = Chinchilla_Guided_Scaling.aggregate_extrapolation_errors(
    #             extrapolation_results_by_spent_ratio
    #         )
            
    #         all_loss_heatmaps.append(loss_heatmap)
    #         all_params_heatmaps.append(params_heatmap)
            
    #         logger.info(f"Aggregated heatmaps from {json_path}")
    #     except Exception as e:
    #         logger.warning(f"Failed to process {json_path}: {e}")
    #         continue
    
    # # Compute mean heatmaps across all seeds
    # if all_loss_heatmaps and all_params_heatmaps:
    #     mean_loss_heatmap = np.mean(all_loss_heatmaps, axis=0)
    #     mean_params_heatmap = np.mean(all_params_heatmaps, axis=0)
        
    #     logger.info(f"Computed mean heatmaps from {len(all_loss_heatmaps)} seeds")
        
    #     # Save aggregated heatmaps
    #     summary_dir_base = Path(optimizer_root_dir_for_seeds[0]).parent / "plots" / "summary_aggregated"
    #     summary_dir_base.mkdir(parents=True, exist_ok=True)
        
    #     target_ratios_list = sorted(Chinchilla_Guided_Scaling.target_over_spent_flops_ratios)
        
    #     # Plot and save mean loss heatmap
    #     fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
        
    #     positive_values = mean_loss_heatmap[mean_loss_heatmap > 0]
    #     vmin = max(positive_values.min(), 0.01) if len(positive_values) > 0 else 0.01
    #     vmax = max(positive_values.max(), 0.1) if len(positive_values) > 0 else 1.0
        
    #     im_loss = ax_loss.imshow(
    #         mean_loss_heatmap,
    #         aspect='auto',
    #         cmap='RdYlGn_r',
    #         norm=LogNorm(vmin=vmin, vmax=vmax),
    #         interpolation='nearest'
    #     )
        
    #     ax_loss.set_xticks(range(len(target_ratios_list)))
    #     ax_loss.set_yticks(range(len(spent_ratios_list_global)))
    #     ax_loss.set_xticklabels([f'{r:.2f}' for r in target_ratios_list], rotation=45, ha='right')
    #     ax_loss.set_yticklabels([f'{r:.3f}' for r in spent_ratios_list_global])
        
    #     ax_loss.set_xlabel('Target / Spent FLOPS Ratio (Interpolation/Extrapolation Distance)', fontsize=12, fontweight='bold')
    #     ax_loss.set_ylabel('Spent / Max Space FLOPS Ratio (Budget Consumption)', fontsize=12, fontweight='bold')
    #     ax_loss.set_title('Mean Loss Prediction Quality Heatmap Across All Seeds', fontsize=13, fontweight='bold')
        
    #     cbar_loss = plt.colorbar(im_loss, ax=ax_loss, label='Mean Relative Prediction Error')
        
    #     # Annotate cells with mean values
    #     for spent_idx in range(len(spent_ratios_list_global)):
    #         for target_idx in range(len(target_ratios_list)):
    #             error = mean_loss_heatmap[spent_idx, target_idx]
    #             if error > 0:
    #                 ax_loss.text(target_idx, spent_idx, f'{error:.2f}',
    #                             ha="center", va="center", color="black", fontsize=8, weight='bold')
        
    #     plt.tight_layout()
    #     fig_loss.savefig(summary_dir_base / "mean_loss_heatmap.png", dpi=150, bbox_inches='tight')
    #     logger.info(f"Saved mean loss heatmap to {summary_dir_base / 'mean_loss_heatmap.png'}")
    #     plt.close(fig_loss)
        
    #     # Plot and save mean params heatmap
    #     fig_params, ax_params = plt.subplots(figsize=(12, 8))
        
    #     positive_values_params = mean_params_heatmap[mean_params_heatmap > 0]
    #     vmin_params = max(positive_values_params.min(), 0.01) if len(positive_values_params) > 0 else 0.01
    #     vmax_params = max(positive_values_params.max(), 0.1) if len(positive_values_params) > 0 else 1.0
        
    #     im_params = ax_params.imshow(
    #         mean_params_heatmap,
    #         aspect='auto',
    #         cmap='RdYlGn_r',
    #         norm=LogNorm(vmin=vmin_params, vmax=vmax_params),
    #         interpolation='nearest'
    #     )
        
    #     ax_params.set_xticks(range(len(target_ratios_list)))
    #     ax_params.set_yticks(range(len(spent_ratios_list_global)))
    #     ax_params.set_xticklabels([f'{r:.2f}' for r in target_ratios_list], rotation=45, ha='right')
    #     ax_params.set_yticklabels([f'{r:.3f}' for r in spent_ratios_list_global])
        
    #     ax_params.set_xlabel('Target / Spent FLOPS Ratio (Interpolation/Extrapolation Distance)', fontsize=12, fontweight='bold')
    #     ax_params.set_ylabel('Spent / Max Space FLOPS Ratio (Budget Consumption)', fontsize=12, fontweight='bold')
    #     ax_params.set_title('Mean Parameter Prediction Quality Heatmap Across All Seeds', fontsize=13, fontweight='bold')
        
    #     cbar_params = plt.colorbar(im_params, ax=ax_params, label='Mean Relative Prediction Error')
        
    #     # Annotate cells with mean values
    #     for spent_idx in range(len(spent_ratios_list_global)):
    #         for target_idx in range(len(target_ratios_list)):
    #             error = mean_params_heatmap[spent_idx, target_idx]
    #             if error > 0:
    #                 ax_params.text(target_idx, spent_idx, f'{error:.2f}',
    #                             ha="center", va="center", color="black", fontsize=8, weight='bold')
    #     plt.tight_layout()
    #     fig_params.savefig(summary_dir_base / "mean_params_heatmap.png", dpi=150, bbox_inches='tight')
    #     logger.info(f"Saved mean params heatmap to {summary_dir_base / 'mean_params_heatmap.png'}")
    #     plt.close(fig_params)
        
    #     # Save mean heatmaps as JSON
    #     mean_heatmap_data = {
    #         "loss_heatmap": mean_loss_heatmap.tolist(),
    #         "params_heatmap": mean_params_heatmap.tolist(),
    #         "num_seeds": len(all_loss_heatmaps),
    #         "target_ratios": [float(r) for r in target_ratios_list],
    #     }
    #     with open(summary_dir_base / "mean_heatmaps.json", 'w') as f:
    #         json.dump(mean_heatmap_data, f, indent=2)
    #     logger.info(f"Saved mean heatmaps data to {summary_dir_base / 'mean_heatmaps.json'}")
    # else:
    #     logger.warning("No heatmaps collected from any seeds")
    
    # Aggregated plots across multiple seeds
    # if len(optimizers) > 1:
    #     comprehensive_study_root_dir = (
    #         f"/work/dlclarge1/alipourn-benchmark_res/{args.benchmark}/"
    #         f"{OptimizerEnum.GRID_SEARCH}/seed_40"
    #     )
    #     summary_dir = Path(comprehensive_study_root_dir) / "summary"
        
    #     # Try to load pareto_info from JSON, or generate and save it
    #     pareto_info = load_pareto_info(summary_dir)
    #     if pareto_info is None:
    #         pareto_info = get_true_pareto_front(comprehensive_study_root_dir)
    #         save_pareto_info(pareto_info, summary_dir)
        
    #     # Create aggregated plots
    #     create_aggregated_pareto_plots(
    #         [root_dirs[optimizer_name] for optimizer_name in optimizers],
    #         pareto_info,
    #         summary_dir
    #     )
