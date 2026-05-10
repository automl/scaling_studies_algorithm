from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from plotting.prediction_plots import get_optimizer_color, get_optimizer_linestyle
from src.plotting.standalone_plots import process_optimizer
from src.extrapolate import extrapolate_single
from src.baselines.configs import OptimizerEnum
from src.benchmarks.configs import get_benchmark

def plot_optimizer_extrapolation(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]],
    optimizer_asks: dict[str, any],
    true_pareto_info: dict,
    target_flops: float,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    log_scale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Loss Over Optimization (with Extrapolation)",
    flop_range: tuple[float, float] | None = None,
):
    """
    Plot optimizer loss over FLOPs, including extrapolated data up to 5x max_flops.
    
    For each optimizer, extrapolates to 5 * benchmark_max_flops and extracts loss data:
    - Chinchilla approaches: result["configs"][k]["param_data"]["loss"]
    - CARBS methods: result["configs"][k]["loss"]
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate target FLOPs for extrapolation
    
    all_data = []
    all_first_flops = []
    
    for opt_name, seeds in optimizers.items():
        color = get_optimizer_color(opt_name)
        linestyle = get_optimizer_linestyle(opt_name)
        
        # Get actual observed data
        flop_counts, mean_losses, std_losses = process_optimizer(
            base_path=base_path,
            optimizer_name=opt_name,
            seeds=seeds,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            log_scale=log_scale,
            interpolate=False,
        )
        
        # Trim data to flop_range if specified
        if flop_range is not None:
            _, max_flops = flop_range
            mask = 0 < flop_counts <= max_flops - 1e-8
            flop_counts = flop_counts[mask]
            mean_losses = mean_losses[mask]
            std_losses = std_losses[mask]
        
        # Extrapolate data
        extrapolation_results = []
        for seed in seeds:
            root_dir = base_path / opt_name / f"seed_{seed}"
            if optimizer_asks.get(opt_name) is not None:
                for i in range(len(flop_counts)):
                    # try:
                    print(f"step {i}")
                    result = extrapolate_single(
                        root_directory=root_dir,
                        optimizer_ask=optimizer_asks[opt_name],
                        last_trial_id=i,
                        target_flops=target_flops,
                    )
                    if result is not None:
                        extrapolation_results.append(result)
                    # except Exception as e:
                    #     print(f"Warning: Could not extrapolate for {opt_name} seed {seed}: {e}")
                    #     continue
        
        # Extract extrapolated losses based on optimizer type
        extrapolated_flops = []
        extrapolated_losses = []
        
        if extrapolation_results:
            for result in extrapolation_results:
                if "configs" in result:
                    for k, config_data in result["configs"].items():
                        # Determine loss extraction based on optimizer type
                        if "chinchilla" in opt_name.lower():
                            # Chinchilla approach: extract from param_data
                            if "param_data" in config_data and "loss" in config_data["param_data"]:
                                loss = config_data["param_data"]["loss"]
                            else:
                                continue
                        else:
                            # CARBS methods: extract directly from config
                            if "loss" in config_data:
                                loss = config_data["loss"]
                            else:
                                continue
                        
                        # Extract FLOPs value (k is typically the flops value)
                        try:
                            flops_val = float(k)
                            extrapolated_flops.append(flops_val)
                            extrapolated_losses.append(loss)
                        except (ValueError, TypeError):
                            continue
        
        # Combine observed and extrapolated data
        if len(flop_counts) > 0:
            combined_flops = list(flop_counts)
            combined_losses = list(mean_losses)
            is_extrapolated_list = [False] * len(flop_counts)
            
            if extrapolated_flops:
                # Sort extrapolated data by FLOPs
                sorted_idx = np.argsort(extrapolated_flops)
                for idx in sorted_idx:
                    combined_flops.append(extrapolated_flops[idx])
                    combined_losses.append(extrapolated_losses[idx])
                    is_extrapolated_list.append(True)
            
            combined_flops = np.array(combined_flops)
            combined_losses = np.array(combined_losses)
            is_extrapolated_array = np.array(is_extrapolated_list, dtype=bool)
            
            # Sort combined data
            sort_idx = np.argsort(combined_flops)
            combined_flops = combined_flops[sort_idx]
            combined_losses = combined_losses[sort_idx]
            is_extrapolated_array = is_extrapolated_array[sort_idx]
            
            if len(combined_flops) > 0:
                all_data.append({
                    'opt_name': opt_name,
                    'color': color,
                    'linestyle': linestyle,
                    'flop_counts': combined_flops,
                    'mean_losses': combined_losses,
                    'is_extrapolated': is_extrapolated_array,
                })
                
                non_zero_flops = combined_flops[combined_flops > 1e-8]
                if len(non_zero_flops) > 0:
                    all_first_flops.append(non_zero_flops[0])
    
    # Calculate common starting point
    common_start = max(all_first_flops) if all_first_flops else 0
    
    # Plot data
    for data in all_data:
        flop_counts = data['flop_counts']
        mean_losses = data['mean_losses']
        is_extrapolated = data.get('is_extrapolated', np.zeros(len(flop_counts), dtype=bool))
        
        # Trim data to common_start
        mask = flop_counts >= common_start - 1e-8
        flop_counts = flop_counts[mask]
        mean_losses = mean_losses[mask]
        is_extrapolated = is_extrapolated[mask]
        
        if len(flop_counts) > 0:
            # Plot observed data (solid line)
            observed_mask = ~is_extrapolated
            if np.any(observed_mask):
                ax.plot(
                    flop_counts[observed_mask],
                    mean_losses[observed_mask],
                    label=data['opt_name'],
                    color=data['color'],
                    linestyle=data['linestyle'],
                    linewidth=2.5,
                )
            
            # Plot extrapolated data (dotted continuation)
            extrapolated_mask = is_extrapolated
            if np.any(extrapolated_mask):
                ax.plot(
                    flop_counts[extrapolated_mask],
                    mean_losses[extrapolated_mask],
                    color=data['color'],
                    linestyle=':',  # Dotted line for extrapolated
                    linewidth=2.5,
                    alpha=0.7,
                )
    
    
    ax.set_xlabel("FLOPs (log scale)", fontsize=13)
    ax.set_ylabel("Loss (log scale)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Place legend below with reduced padding
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=min(4, len(optimizers)), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    
    return fig, ax



if __name__ == "__main__":
    from pathlib import Path
    
    results_dir = Path("/work/dlclarge1/alipourn-scan_suite")
    loss_key = "objective_to_minimize"
    
    # Seeds to process
    # all_seeds = list(set(range(40, 46)))
    all_seeds = [42]
    optimizers_dict = {
        # "random_search": all_seeds,
        # "random_search_with_priors": all_seeds,
        "chinchilla_approach1": all_seeds,
        # "chinchilla_approach1_with_priors": all_seeds,
        # "chinchilla_approach2": all_seeds,
        # "chinchilla_approach2_with_priors": all_seeds,
        # "carbs": all_seeds,
        # "carbs_with_priors": all_seeds,
    }
    
    from src.benchmarks.configs import BenchmarkEnum
    from src.benchmarks.llm_space import LLM_TARGETS
    from src.benchmarks.openclip_space import OPENCLIP_TARGETS
    from src.plotting.plot_styles import BENCHMARK_TITLES
    
    benchmarks = [BenchmarkEnum.OPENCLIP.value]
    
    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Generating extrapolation plots for: {benchmark_name}")
        print(f"{'='*60}")
        
        targets = {BenchmarkEnum.LLM.value: LLM_TARGETS, BenchmarkEnum.OPENCLIP.value: OPENCLIP_TARGETS}
        flop_range = (5e16 if benchmark_name == BenchmarkEnum.OPENCLIP.value else 9.99e18, get_benchmark(benchmark_name, False).max_evaluation_flops * 0.9)
        for target in targets[benchmark_name]:
            try:
                target_val = target.value if hasattr(target, 'value') else target
            except AttributeError:
                target_val = target
            if not target_val.startswith("val"):
                continue
            
            plot_output_dir = results_dir / benchmark_name / "plots" / target_val
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Plotting extrapolation for {target_val}...")
            
            from src.benchmarks.configs import get_benchmark
            benchmark = get_benchmark(benchmark_name, False)
            
            # Load pareto info
            comprehensive_study_root_dir = results_dir / benchmark_name / OptimizerEnum.GRID_SEARCH.value / "seed_40"
            summary_dir = Path(comprehensive_study_root_dir) / "summary"
            from src.extrapolate import load_pareto_info, get_true_pareto_front
            
            pareto_info = load_pareto_info(summary_dir)
            if pareto_info is None:
                pareto_info = get_true_pareto_front(comprehensive_study_root_dir)
            
            # Build optimizer_asks dictionary
            from src.baselines.configs import get_optimizer_wrapper, OptimizerEnum
            optimizer_asks = {}
            
            for opt_name in optimizers_dict.keys():
                base_opt_name = opt_name.replace("_with_priors", "")
                has_priors = "_with_priors" in opt_name
                
                try:
                    optimizer_enum = OptimizerEnum(base_opt_name)
                    root_dir = results_dir / benchmark_name / opt_name / "seed_40"
                    
                    optimizer_ask, _ = get_optimizer_wrapper(
                        pipeline_space=benchmark.space,
                        evaluate_pipeline=benchmark.evaluate_pipeline,
                        param_estimator=benchmark.params_estimator,
                        seen_datapoints_estimator=benchmark.seen_datapoints_estimator,
                        flops_estimator=benchmark.flops_estimator,
                        aspect_ratio_fn=benchmark.aspect_ratio_fn,
                        data_parameter_name=benchmark.data_parameter_name,
                    )(
                        optimizer_enum,
                        root_dir=root_dir,
                        max_evaluation_flops=benchmark.max_evaluation_flops,
                    )
                    optimizer_asks[opt_name] = optimizer_ask.optimizer(benchmark.space)
                except Exception as e:
                    print(f"    Warning: Could not load optimizer {opt_name}: {e}")
                    optimizer_asks[opt_name] = None
            
            # Plot extrapolation
            try:
                fig_extrap, ax_extrap = plot_optimizer_extrapolation(
                    base_path=results_dir / benchmark_name,
                    optimizers=optimizers_dict,
                    optimizer_asks=optimizer_asks,
                    true_pareto_info=pareto_info,
                    target_flops= benchmark.max_flops * 0.7,
                    key_to_extract="cost",
                    loss_key=loss_key,
                    title=f"{BENCHMARK_TITLES[benchmark_name]} (with Extrapolation)",
                )
                fig_extrap.savefig(plot_output_dir / "loss_extrapolation.png", dpi=300, bbox_inches="tight")
                plt.close(fig_extrap)
                print(f"    ✓ Saved: {plot_output_dir / 'loss_extrapolation.png'}")
            except Exception as e:
                print(f"    ✗ Error plotting extrapolation: {e}")
                import traceback
                traceback.print_exc()
