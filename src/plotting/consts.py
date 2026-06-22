from src.benchmarks.configs import get_benchmark, BenchmarkEnum
from src.extrapolate import load_or_generate_pareto_info
from pathlib import Path

acc_flop_ranges_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (4e16, get_benchmark(BenchmarkEnum.OPENCLIP.value, False).max_evaluation_flops * 0.9),
    BenchmarkEnum.LLM.value: (5.1e18, get_benchmark(BenchmarkEnum.LLM.value, False).max_evaluation_flops * 0.9),
    BenchmarkEnum.TABPFN.value: (1e17, get_benchmark(BenchmarkEnum.TABPFN.value, False).max_evaluation_flops * 0.9),
}

acc_pred_flop_ranges_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (4e16, get_benchmark(BenchmarkEnum.OPENCLIP.value, False).max_evaluation_flops * 0.9),
    BenchmarkEnum.LLM.value: (5.1e18, get_benchmark(BenchmarkEnum.LLM.value, False).max_evaluation_flops * 0.9),
    BenchmarkEnum.TABPFN.value: (1e17, get_benchmark(BenchmarkEnum.TABPFN.value, False).max_evaluation_flops * 0.9),
}

selected_best_seed_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: 49,
    BenchmarkEnum.LLM.value: 44,
    BenchmarkEnum.TABPFN.value: 49,
}

flop_range_to_show_in_pareto_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (
        get_benchmark(BenchmarkEnum.OPENCLIP.value, False).min_flops,
        get_benchmark(BenchmarkEnum.OPENCLIP.value, False).max_flops,
    ),
    BenchmarkEnum.LLM.value: (
        get_benchmark(BenchmarkEnum.LLM.value, False).min_flops,
        get_benchmark(BenchmarkEnum.LLM.value, False).max_flops,
    ),
    BenchmarkEnum.TABPFN.value: (
        get_benchmark(BenchmarkEnum.TABPFN.value, False).min_flops,
        get_benchmark(BenchmarkEnum.TABPFN.value, False).max_flops,
    ),
}

target_flop_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: get_benchmark(BenchmarkEnum.OPENCLIP.value, False).max_flops * 2,
    BenchmarkEnum.LLM.value: get_benchmark(BenchmarkEnum.LLM.value, False).max_flops * 2,
    BenchmarkEnum.TABPFN.value: get_benchmark(BenchmarkEnum.TABPFN.value, False).max_flops * 2,
}
hp_target_flop_ratios = (0.5, 1.0, 5.0)
acc_pred_flop_ranges_per_benchmark_by_ratio = {
    benchmark_name: {
        ratio: (
            acc_pred_flop_ranges_per_benchmark[benchmark_name][0],
            ratio * target_flop,
        )
        for ratio in hp_target_flop_ratios
    }
    for benchmark_name, target_flop in target_flop_per_benchmark.items()
    if benchmark_name in acc_pred_flop_ranges_per_benchmark
}
true_pareto_info_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: load_or_generate_pareto_info(BenchmarkEnum.OPENCLIP.value),
    BenchmarkEnum.LLM.value: load_or_generate_pareto_info(BenchmarkEnum.LLM.value),
    BenchmarkEnum.TABPFN.value: load_or_generate_pareto_info(BenchmarkEnum.TABPFN.value),
}

bench_confs = {
    BenchmarkEnum.OPENCLIP.value: get_benchmark(BenchmarkEnum.OPENCLIP.value, False),
    BenchmarkEnum.LLM.value: get_benchmark(BenchmarkEnum.LLM.value, False),
    BenchmarkEnum.TABPFN.value: get_benchmark(BenchmarkEnum.TABPFN.value, False),
}

results_dir = Path("/work/dlclarge1/alipourn-scan_suite")
loss_key = "objective_to_minimize"


all_seeds = list(set(range(40, 51)))
# all_seeds = [40, 41, 42, 43, 44, 47, 48]
# all_seeds = [40]
optimizers_dict = {
    "random_search": all_seeds,
    # "random_search_with_priors": all_seeds,
    "chinchilla_approach1": all_seeds,
    # "chinchilla_approach1_with_priors": all_seeds,
    "chinchilla_approach2": all_seeds,
    # "chinchilla_approach2_with_priors": all_seeds,
    # "chinchilla_gemstones": all_seeds,
    "carbs": all_seeds,
    # "carbs_with_priors": all_seeds,
    # "bo_space_expansion_mes": all_seeds,
    # "bo_space_expansion_ei": all_seeds,
    # "bo_space_expansion_jes": all_seeds,
    # "bo_left_budget": all_seeds,
    "bo_left_budget_ei": all_seeds,
    "bo_left_budget_jes": all_seeds,
    "bo_left_budget_mes": all_seeds,
    # "bo_left_budget_with_priors": all_seeds,
    # "bo_left_budget_zero_mean_ei": all_seeds,
    # "bo_left_budget_zero_mean_ei_with_priors": all_seeds,
    "bo_space_expansion_modified_jes": all_seeds,
    "bo_left_budget_modified_jes": all_seeds,
}
