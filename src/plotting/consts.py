import os

from src.benchmarks.configs import get_benchmark, get_predictor_type, get_results_dir, BenchmarkEnum
from src.extrapolate import load_or_generate_pareto_info
from src.plotting.utils import target_loss_key
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
from pathlib import Path

predictor_type = get_predictor_type()

# Downstream target to plot instead of each run's own objective, selected the same
# way as the predictor (env var, so every module building constants at import time
# agrees). Unset -> the objective the study actually optimized (upstream loss).
# The target must have been written into the reports first, by
# `python -m src.posthoc_downstream_eval --target <target> ...`.
TARGET_ENV_VAR = "SCAN_TARGET"
target = os.environ.get(TARGET_ENV_VAR) or None

acc_flop_ranges_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (4e16, get_benchmark(BenchmarkEnum.OPENCLIP.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
    BenchmarkEnum.LLM.value: (5.1e18, get_benchmark(BenchmarkEnum.LLM.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
    BenchmarkEnum.TABPFN.value: (9.9e13, get_benchmark(BenchmarkEnum.TABPFN.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
}

search_space_per_benchmark = {bench.value: convert_neps_to_classic_search_space(get_benchmark(bench.value, False, predictor_type=predictor_type, lazy=True).space) for bench in BenchmarkEnum}
cost_estimator_per_benchmark = {bench.value: get_benchmark(bench.value, False, predictor_type=predictor_type, lazy=True).flops_estimator for bench in BenchmarkEnum}
params_estimator_per_benchmark = {bench.value: get_benchmark(bench.value, False, predictor_type=predictor_type, lazy=True).params_estimator for bench in BenchmarkEnum}

acc_pred_flop_ranges_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (4e16, get_benchmark(BenchmarkEnum.OPENCLIP.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
    BenchmarkEnum.LLM.value: (5.1e18, get_benchmark(BenchmarkEnum.LLM.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
    BenchmarkEnum.TABPFN.value: (9.9e13, get_benchmark(BenchmarkEnum.TABPFN.value, False, predictor_type=predictor_type, lazy=True).max_evaluation_flops * 0.9),
}

selected_best_seed_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: 48,
    BenchmarkEnum.LLM.value: 47,
    BenchmarkEnum.TABPFN.value: 47,
}

flop_range_to_show_in_pareto_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: (
        get_benchmark(BenchmarkEnum.OPENCLIP.value, False, predictor_type=predictor_type, lazy=True).min_flops,
        get_benchmark(BenchmarkEnum.OPENCLIP.value, False, predictor_type=predictor_type, lazy=True).max_flops,
    ),
    BenchmarkEnum.LLM.value: (
        get_benchmark(BenchmarkEnum.LLM.value, False, predictor_type=predictor_type, lazy=True).min_flops,
        get_benchmark(BenchmarkEnum.LLM.value, False, predictor_type=predictor_type, lazy=True).max_flops,
    ),
    BenchmarkEnum.TABPFN.value: (
        get_benchmark(BenchmarkEnum.TABPFN.value, False, predictor_type=predictor_type, lazy=True).min_flops,
        get_benchmark(BenchmarkEnum.TABPFN.value, False, predictor_type=predictor_type, lazy=True).max_flops,
    ),
}

target_flop_multipliers = (10, 20)

target_flop_list_per_benchmark = {
    benchmark.value: [
        get_benchmark(benchmark.value, False, predictor_type=predictor_type, lazy=True).max_flops * multiplier
        for multiplier in target_flop_multipliers
        ] for benchmark in BenchmarkEnum
}

hp_target_flop_ratios = (0.05, 0.1, 0.5)
acc_pred_flop_ranges_per_benchmark_by_ratio = {
    benchmark_name: {
        target_flop: {
            ratio: (
                acc_pred_flop_ranges_per_benchmark[benchmark_name][0],
                ratio * target_flop,
            )
            for ratio in hp_target_flop_ratios
        }
        for target_flop in target_flops
    }
    for benchmark_name, target_flops in target_flop_list_per_benchmark.items()
    if benchmark_name in acc_pred_flop_ranges_per_benchmark
}
true_pareto_info_per_benchmark = {
    BenchmarkEnum.OPENCLIP.value: load_or_generate_pareto_info(BenchmarkEnum.OPENCLIP.value, target=target),
    BenchmarkEnum.LLM.value: load_or_generate_pareto_info(BenchmarkEnum.LLM.value, target=target),
    BenchmarkEnum.TABPFN.value: load_or_generate_pareto_info(BenchmarkEnum.TABPFN.value),
}

bench_confs = {
    # lazy=False for OPENCLIP/LLM (unlike the metadata-only dicts above): bench_confs
    # is the one place `.evaluate_pipeline` is actually invoked (table.py's
    # HP-prediction scoring), and evaluate_pipeline closes over `bench` -- with
    # lazy=True `bench` is None, so every call raised inside the wrapper's
    # try/except and silently returned NaN, making every "actual_loss" prediction
    # look diverged. TABPFN stays lazy=True: eagerly constructing it currently
    # crashes (installed scan_bench.tabpfn.api is missing an `import os`),
    # and table.py's benchmark loop excludes TABPFN anyway.
    BenchmarkEnum.OPENCLIP.value: get_benchmark(BenchmarkEnum.OPENCLIP.value, False, predictor_type=predictor_type, lazy=True),
    BenchmarkEnum.LLM.value: get_benchmark(BenchmarkEnum.LLM.value, False, predictor_type=predictor_type, lazy=True),
    BenchmarkEnum.TABPFN.value: get_benchmark(BenchmarkEnum.TABPFN.value, False, predictor_type=predictor_type, lazy=True),
}

results_dir = get_results_dir(predictor_type)
# results_dir = Path("/work/dlc2workfs2/alipourn-scan/test")
loss_key = "objective_to_minimize" if target is None else target_loss_key(target)


all_seeds = list(set(range(40, 50)))
# all_seeds = [40, 42, 43, 44, 45, 46, 47, 48]
# all_seeds = [50]
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
    # "bo_left_budget_ei": all_seeds,
    # "bo_left_budget_jes": all_seeds,
    # "bo_left_budget_mes": all_seeds,
    # "bo_left_budget_with_priors": all_seeds,
    # "bo_left_budget_zero_mean_ei": all_seeds,
    # "bo_left_budget_zero_mean_ei_with_priors": all_seeds,
    # "bo_space_expansion_modified_jes": all_seeds,
    # "bo_left_budget_modified_jes": all_seeds,
}
