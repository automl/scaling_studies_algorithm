# given a search space and grid parameters, generate a list of configurations and calculate their flops using the get_total_flops function
from typing import Any, List, Union

import neps
from neps.optimizers.utils.grid import make_grid


def get_grids_total_flops(
    search_space: neps.PipelineSpace,
    flop_estimator: callable,
    size_per_numerical_hp: int | dict[str, int] = 5,
) -> List[Union[dict[str, Any], dict[str, Any]]]:
    
    confs = make_grid(search_space, size_per_numerical_hp=size_per_numerical_hp)
    total_flops = 0
    for conf in confs:
        print(conf)
        total_flops += flop_estimator(**conf)
    return confs, total_flops

if __name__ == "__main__":
    from src.benchmarks.synthetic_mlp_space import get_total_flops, max_target_flops

    pipeline_space = neps.SearchSpace(dict(
        batch_size = neps.HPOInteger(lower=64, upper=512, is_scaling=True),
        num_layers = neps.HPOInteger(lower=2, upper=4, is_scaling=True),
        max_units = neps.HPOInteger(lower=64, upper=512, is_scaling=True),
        learning_rate = neps.HPOFloat(lower=0.0001, upper=0.1),
    ))

    max_size_per_numerical_hp = {
        "batch_size": 4,
        "num_layers": 3,
        "max_units": 3,
        "learning_rate": 3,
    }

    # Test 1: Get FLOPs for full grid
    print("=" * 70)
    print("TEST 1: Full grid with max sizes")
    print("=" * 70)
    confs, total_flops = get_grids_total_flops(pipeline_space, get_total_flops, size_per_numerical_hp=max_size_per_numerical_hp)
    print(f"Total FLOPs for full grid: {total_flops:.2e}\n")
    
    # Test 2: Find grid that matches a target FLOPs (80% of full grid)
    print("=" * 70)
    print("TEST 2: Find grid matching 80% of full grid FLOPs")
    print("=" * 70)
    target_flops = total_flops * 0.8
    optimal_sizes = find_the_fit_grid_for_total_flops(
        target_flops=target_flops,
        space=pipeline_space,
        max_size_per_numerical_hp=max_size_per_numerical_hp,
        flop_estimator=get_total_flops,
        jitter=0.01,  # 1% tolerance
    )
    
    # Verify the result
    confs, achieved_flops = get_grids_total_flops(pipeline_space, get_total_flops, size_per_numerical_hp=optimal_sizes)
    ratio = achieved_flops / target_flops
    print(f"\n{'='*70}")
    print(f"VERIFICATION: Achieved FLOPs ratio: {ratio:.6f}")
    print(f"Target: {target_flops:.2e}, Achieved: {achieved_flops:.2e}")
    print(f"{'='*70}")

def find_the_fit_grid_for_total_flops(
    target_flops: float,
    space: neps.PipelineSpace,
    max_size_per_numerical_hp: dict[str, int],
    flop_estimator: callable,
    jitter: float = 0.001,
) -> dict[str, int]:
    """
    Smart grid search to find size_per_numerical_hp that matches target_flops within jitter tolerance.
    
    Strategy:
    1. Start with all sizes at (max_size - 1) - gives lower bound
    2. Calculate FLOPs for baseline
    3. If too low, track impact of each hyperparameter on FLOPs
    4. Greedily increase hyperparameters that keep us in the acceptable range
    5. Return configuration matching: (1 - jitter) < calculated_flops / target_flops < (1 + jitter)
    
    Args:
        target_flops: Target total FLOPs
        space: neps SearchSpace
        max_size_per_numerical_hp: Dict mapping hp names to max grid sizes
        flop_estimator: Function to calculate total FLOPs for a grid config
        jitter: Tolerance factor (default 0.001 = 0.1%)
    
    Returns:
        Dict mapping hyperparameter names to grid sizes that fit the target
    """
    lower_bound = target_flops * (1 - jitter)
    upper_bound = target_flops * (1 + jitter)
    
    # Start with all sizes at (max - 1)
    current_sizes = {hp: max(1, max_size_per_numerical_hp[hp] - 1) 
                     for hp in max_size_per_numerical_hp}
    
    # Calculate baseline FLOPs
    baseline_flops, _ = get_grids_total_flops(space, flop_estimator, size_per_numerical_hp=current_sizes)
    
    print(f"Target FLOPs: {target_flops:.2e} (range: {lower_bound:.2e} - {upper_bound:.2e})")
    print(f"Baseline FLOPs (all sizes - 1): {baseline_flops:.2e}")
    print(f"Current sizes: {current_sizes}")
    
    # Check if baseline is already in range
    if lower_bound <= baseline_flops <= upper_bound:
        print(f"✓ Baseline configuration fits! FLOPs ratio: {baseline_flops / target_flops:.4f}")
        return current_sizes
    
    # If baseline is too high, we can't fit by only decreasing
    if baseline_flops > upper_bound:
        print(f"✗ Baseline FLOPs already exceeds upper bound. Cannot fit by only decreasing.")
        return current_sizes
    
    # Baseline is too low - now incrementally increase sizes
    print(f"\nBaseline too low by {(target_flops - baseline_flops) / target_flops * 100:.1f}%")
    print("Searching for optimal configuration by increasing sizes...\n")
    
    best_config = current_sizes.copy()
    best_flops = baseline_flops
    best_ratio = best_flops / target_flops
    
    # Track the FLOPs impact of increasing each hyperparameter
    hp_impact = {}
    
    # Iteratively increase grid sizes
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # If current config is in range, we're done
        if lower_bound <= best_flops <= upper_bound:
            print(f"\n✓ Found fitting configuration!")
            print(f"  FLOPs: {best_flops:.2e}, Ratio: {best_ratio:.6f}")
            print(f"  Sizes: {best_config}")
            return best_config
        
        # Try increasing each hyperparameter by 1 and see which ones fit
        candidates = []
        
        for hp_name in current_sizes:
            # Skip if already at max
            if current_sizes[hp_name] >= max_size_per_numerical_hp[hp_name]:
                continue
            
            # Try increasing this hyperparameter
            test_sizes = current_sizes.copy()
            test_sizes[hp_name] += 1
            
            test_flops, _ = get_grids_total_flops(space, flop_estimator, size_per_numerical_hp=test_sizes)
            ratio = test_flops / target_flops
            
            # Calculate impact
            impact = test_flops - best_flops
            hp_impact[hp_name] = impact
            
            # Check if it fits
            fits = lower_bound <= test_flops <= upper_bound
            
            # Store candidate: (fits, distance_to_target, impact, hp_name, config, flops, ratio)
            distance_to_target = abs(test_flops - target_flops)
            candidates.append((fits, distance_to_target, impact, hp_name, test_sizes, test_flops, ratio))
        
        if not candidates:
            print(f"✗ Reached maximum sizes, cannot fit target FLOPs")
            print(f"  Best found: {best_flops:.2e}, Ratio: {best_ratio:.6f}")
            return best_config
        
        # Sort candidates: prioritize those that fit, then by distance to target
        candidates.sort(key=lambda x: (not x[0], x[1]))
        
        # Pick the best candidate
        fits, distance, impact, hp_name, test_config, test_flops, ratio = candidates[0]
        
        # Update best config
        current_sizes = test_config
        best_config = test_config
        best_flops = test_flops
        best_ratio = ratio
        
        print(f"Iter {iteration:3d}: Increased {hp_name:15s} -> "
              f"FLOPs: {test_flops:.2e}, Ratio: {ratio:.6f}, "
              f"Fits: {'✓' if fits else '✗'}, Distance: {distance:.2e}")
        
        if iteration % 50 == 0:
            print(f"  Current config: {current_sizes}")
    
    print(f"\n⚠ Max iterations reached ({max_iterations})")
    print(f"Best found: {best_flops:.2e}, Ratio: {best_ratio:.6f}")
    print(f"Sizes: {best_config}")
    
    return best_config

