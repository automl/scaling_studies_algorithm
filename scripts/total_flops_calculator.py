
import sys
from pathlib import Path
import numpy as np
import neps
from neps.state.trial import State

# Add the src directory to the Python path to allow for importing modules from there
sys.path.append(str(Path(__file__).parent.parent))

from src.benchmarks.configs import BenchmarkEnum, get_benchmark
from src.benchmarks.llm_space import LLM_TARGETS
from src.benchmarks.openclip_space import OPENCLIP_TARGETS
from src.benchmarks.tabpfn_space import TABPFN_TARGETS
from src.benchmarks.configs import BenchmarkEnum, get_benchmark
def get_total_flops_for_seed(
    *,
    path: str | Path,
    bench_max_flops: float,
    seed: str | int | None,
    key_to_extract: str | None = None,
) -> float:
    """
    Calculates the total FLOPs for a single seed by summing the costs.
    """
    path = Path(path)
    if seed is not None:
        path = path / f"seed_{seed}"

    try:
        _fulldf, _ = neps.status(path, print_summary=False)
        if _fulldf.empty:
            print(f"  - No trials found in {path}")
            return 0.0

        _fulldf = _fulldf.sort_values("time_sampled")

        def get_cost(idx: str | int) -> float:
            row = _fulldf.loc[idx]
            if key_to_extract and key_to_extract in row:
                return float(row[key_to_extract])
            return 1.0

        costs = []
        
        for config_id, config_result in _fulldf.iterrows():
            if config_result["state"] != State.SUCCESS:
                continue
            costs.append(get_cost(config_id))
        costs_arr = np.array(costs, dtype=float)
        mask = costs_arr > (bench_max_flops * 1.1)
        if np.any(mask):
            print(f"  - Warning: Some costs exceed the benchmark's max FLOPs ({bench_max_flops:.2e}). Capping those costs.")
            raise Exception(f"Costs exceeding max FLOPs: {costs_arr[mask]}")
        total_cost = sum(costs)
        print(f"  - Processed seed {seed} for {path.parent.name}: {total_cost:.2e} FLOPs")
        return total_cost

    except FileNotFoundError:
        print(f"  - Directory not found: {path}")
        return 0.0
    except Exception as e:
        print(f"  - Error processing {path}: {e}")
        return 0.0

if __name__ == "__main__":
    results_dir = Path("/work/dlclarge1/alipourn-scan_suite")
    
    all_seeds = list(set(range(40, 51)))
    optimizers_dict = {
        # "random_search": all_seeds,
        # "random_search_with_priors": all_seeds,
        # "chinchilla_approach1": all_seeds,
        # "chinchilla_approach1_with_priors": all_seeds,
        # "chinchilla_approach2": all_seeds,
        # "chinchilla_approach2_with_priors": all_seeds,
        "bo_space_expansion_jes": all_seeds,
        # "carbs_with_priors": all_seeds,
    }

    benchmarks = [
        # BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value,
        # BenchmarkEnum.TABPFN.value
        ]
    
    grand_total_flops = 0
    
    print(f"{'='*60}")
    print("Calculating Total FLOPs Across All Benchmarks and Seeds")
    print(f"{'='*60}\n")

    for benchmark_name in benchmarks:
        bench = get_benchmark(benchmark_name, lc_aware=False)  # Ensure benchmark config is loaded
        benchmark_total_flops = 0
        print(f"Processing benchmark: {benchmark_name}")
        
        targets = {
            BenchmarkEnum.LLM.value: LLM_TARGETS,
            BenchmarkEnum.OPENCLIP.value: OPENCLIP_TARGETS,
            BenchmarkEnum.TABPFN.value: TABPFN_TARGETS,
        }

        for target in targets.get(benchmark_name, []):
            try:
                target_value = target.value
            except AttributeError:
                target_value = target
            
            print(f"  Target: {target_value}")
            
            for opt_name, seeds in optimizers_dict.items():
                optimizer_path = results_dir / benchmark_name / opt_name
                
                for seed in seeds:
                    total_flops = get_total_flops_for_seed(
                        path=optimizer_path,
                        seed=seed,
                        key_to_extract="cost",
                        bench_max_flops =bench.max_flops,
                    )
                    benchmark_total_flops += total_flops
        
        print(f"\nTotal FLOPs for {benchmark_name}: {benchmark_total_flops:.2e}\n")
        grand_total_flops += benchmark_total_flops

    print(f"{'='*60}")
    print(f"Grand Total FLOPs Across All Benchmarks: {grand_total_flops:.2e}")
    print(f"{'='*60}")
