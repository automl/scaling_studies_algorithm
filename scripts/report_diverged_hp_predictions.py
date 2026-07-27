"""Report the extrapolated HP configs that the LINEAR_REGRESSION predictor
produces (per benchmark/tier/ratio/optimizer/seed) which the benchmark's own
divergence surrogate flags as likely to diverge -- i.e. exactly the configs
responsible for the `nan` cells in the predicted-loss table (see
`src/plotting/table.py::plot_predicted_hp_table_over_flops`).

Usage:
    python -m scripts.report_diverged_hp_predictions [--pred tabpfn] [--out FILE]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scan_benchmark.base_performance_benchmark import PerformancePredictorType


def main() -> None:
    parser = argparse.ArgumentParser(description="Report HP-extrapolated configs flagged as diverging by the surrogate.")
    parser.add_argument(
        "--pred", type=str, default=PerformancePredictorType.TABPFN.value,
        choices=[p.value for p in PerformancePredictorType],
    )
    parser.add_argument("--out", type=str, default=None, help="Path to write the JSON report (default: <results_dir>/plots/diverged_configs_<pred>.json)")
    args = parser.parse_args()

    import os
    os.environ["SCAN_PREDICTOR_TYPE"] = args.pred

    from src.benchmarks.configs import BenchmarkEnum
    from src.plotting.standalone_plots import process_seed
    from src.hp_predictor import HPPredictorTypes, predict_hp_from_scaling_law
    from src.plotting.consts import (
        bench_confs, target_flop_list_per_benchmark, target_flop_multipliers,
        acc_pred_flop_ranges_per_benchmark, acc_pred_flop_ranges_per_benchmark_by_ratio,
        hp_target_flop_ratios, results_dir, all_seeds, optimizers_dict, loss_key,
    )
    from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space

    benchmarks = [BenchmarkEnum.LLM.value, BenchmarkEnum.OPENCLIP.value]
    table_optimizer_order = ("carbs", "random_search", "bo_left_budget_modified_jes")
    predictor_type = HPPredictorTypes.LINEAR_REGRESSION

    diverged: list[dict[str, Any]] = []
    total_checked = 0

    for benchmark_name in benchmarks:
        cfg = bench_confs[benchmark_name]
        space = convert_neps_to_classic_search_space(cfg.space)
        base_path = results_dir / benchmark_name
        flop_range = acc_pred_flop_ranges_per_benchmark[benchmark_name]

        # Mirrors the filtering plot_predicted_hp_table_over_flops applies internally:
        # only optimizers in table_optimizer_order that actually have seeds configured.
        optimizers = {
            opt: optimizers_dict[opt]
            for opt in table_optimizer_order
            if opt in optimizers_dict and optimizers_dict[opt]
        }

        for tier_idx, target_flop in enumerate(target_flop_list_per_benchmark[benchmark_name]):
            tier_label = f"{target_flop_multipliers[tier_idx]}x"
            flop_ranges_by_ratio = acc_pred_flop_ranges_per_benchmark_by_ratio[benchmark_name][target_flop]

            for ratio in hp_target_flop_ratios:
                ratio_target_flop = float(ratio * target_flop)

                for opt_name, seeds in optimizers.items():
                    for seed in seeds:
                        try:
                            losses, cumsum_costs, _max_cost, _rows, _lc_rows, configs = process_seed(
                                path=base_path / opt_name,
                                seed=seed,
                                key_to_extract="cost",
                                loss_key=loss_key,
                                return_incumbent=False,
                            )
                        except Exception:
                            continue
                        if not configs or not losses or not cumsum_costs:
                            continue

                        cumsum_arr_full = np.asarray(cumsum_costs, dtype=float)
                        ratio_flop_range = flop_ranges_by_ratio.get(float(ratio)) or flop_ranges_by_ratio.get(ratio)
                        if ratio_flop_range is None:
                            ratio_flop_range = (float(flop_range[0]), ratio_target_flop)
                        cutoff_lower = float(ratio_flop_range[0])
                        cutoff_upper = float(ratio_flop_range[1])

                        cutoff_mask = (cumsum_arr_full <= cutoff_upper) & (cumsum_arr_full >= cutoff_lower)
                        cutoff_indices = np.where(cutoff_mask)[0]
                        if len(cutoff_indices) == 0:
                            cutoff_indices = np.where(cumsum_arr_full <= cutoff_upper)[0]
                        if len(cutoff_indices) == 0:
                            continue

                        start = int(cutoff_indices[0])
                        stop = int(cutoff_indices[-1]) + 1
                        ratio_losses = losses[start:stop]
                        ratio_cumsum_costs = cumsum_costs[start:stop]
                        ratio_configs = configs[start:stop]
                        if not ratio_configs or not ratio_losses or not ratio_cumsum_costs:
                            continue

                        ratio_cumsum_arr = np.asarray(ratio_cumsum_costs, dtype=float)
                        previous_cumsum = float(cumsum_arr_full[start - 1]) if start > 0 else 0.0
                        actual_flops = np.diff(ratio_cumsum_arr, prepend=previous_cumsum).tolist()

                        try:
                            cache_dir = base_path / opt_name / f"seed_{seed}" / "prediction_cache"
                            pred_result = predict_hp_from_scaling_law(
                                space=space,
                                hp_predictor_type=predictor_type,
                                target_flops=ratio_target_flop,
                                configs=ratio_configs,
                                losses=[float(v) for v in ratio_losses],
                                flops=[float(v) for v in actual_flops],
                                cost_estimator=cfg.flops_estimator,
                                cache_dir=cache_dir,
                            )
                            predicted_cfg = dict(pred_result.get("extrapolated_config", {}))
                        except Exception as e:
                            continue

                        total_checked += 1
                        try:
                            eval_result = cfg.evaluate_pipeline(**predicted_cfg)
                        except Exception as e:
                            eval_result = None

                        predictions = (eval_result or {}).get("info_dict", {}).get("predictions") if eval_result else None
                        obj = (eval_result or {}).get("objective_to_minimize") if eval_result else None
                        did_diverge = eval_result is None or predictions is None or obj is None or not np.isfinite(obj)

                        if did_diverge:
                            diverged.append({
                                "benchmark": benchmark_name,
                                "optimizer": opt_name,
                                "seed": seed,
                                "tier": tier_label,
                                "target_flop": target_flop,
                                "ratio": ratio,
                                "ratio_target_flop": ratio_target_flop,
                                "n_trials_in_window": len(ratio_configs),
                                "predicted_config": predicted_cfg,
                            })

    print(f"Checked {total_checked} extrapolated configs; {len(diverged)} flagged as diverging by the surrogate.\n")
    for d in diverged:
        print(
            f"[{d['benchmark']}/{d['optimizer']}/seed_{d['seed']}/{d['tier']}/ratio={d['ratio']}] "
            f"target_flop={d['ratio_target_flop']:.2e} (n_trials_in_window={d['n_trials_in_window']}):"
        )
        print(f"    {d['predicted_config']}")

    out_path = Path(args.out) if args.out else (results_dir / "plots" / f"diverged_configs_{args.pred}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(diverged, fh, indent=2)
    print(f"\nWrote {len(diverged)} diverged-config records to {out_path}")


if __name__ == "__main__":
    main()
