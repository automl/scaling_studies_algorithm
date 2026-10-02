"""Side-by-side predicted loss / model size (all optimizers per panel) across scaling-law predictors.

Per benchmark, two 1x2 figures (SURROGATE runs at TIER x target FLOP):
  - predicted loss:   Chinchilla3 | Kaplan
  - predicted N:      Chinchilla3 | Power Law fit
Written to <results base>/plots/surrogate_comparison/.

    python -m src.plotting.surrogate_comparison_plots
"""
import matplotlib.pyplot as plt
import numpy as np

from scan_bench.base_performance_benchmark import PerformancePredictorType
from src.benchmarks.configs import BenchmarkEnum, RESULTS_BASE_DIR, get_results_dir
from src.loss_predictor import LossPredictorTypes
from src.scale_param_predictor import ScaleParamPredictorTypes
from src.plotting.consts import (
    target_flop_list_per_benchmark, target_flop_multipliers, true_pareto_info_per_benchmark,
    acc_pred_flop_ranges_per_benchmark, loss_key, target, optimizers_dict,
    search_space_per_benchmark, cost_estimator_per_benchmark, params_estimator_per_benchmark,
)
from src.plotting.prediction_plots import plot_predicted_loss_over_flops, plot_predicted_n_param_over_flops
from src.plotting.utils import merge_pareto_plots

SURROGATE = PerformancePredictorType.TABPFN
# Target tier (multiple of the base target FLOP) at which the predictors are compared.
TIER = 10

LOSS_PREDICTORS = [LossPredictorTypes.CHINCHILLA3, LossPredictorTypes.KAPLAN]
N_PARAM_PREDICTORS = [ScaleParamPredictorTypes.CHINCHILLA3, ScaleParamPredictorTypes.POWER_LAW_FIT]


def _loss_panel(benchmark_name, surrogate, tier_idx, predictor_type):
    true_pareto_info = true_pareto_info_per_benchmark[benchmark_name]
    fig, ax = plot_predicted_loss_over_flops(
        base_path=get_results_dir(surrogate) / benchmark_name,
        optimizers=optimizers_dict,
        target_flop=target_flop_list_per_benchmark[benchmark_name][tier_idx],
        predictor_type=predictor_type,
        key_to_extract="cost",
        loss_key=loss_key,
        flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
        true_pareto_info=true_pareto_info,
        return_split=False,
        normalized_flops=False,
        abs_loss_diff=true_pareto_info is not None,
        cost_estimator=cost_estimator_per_benchmark[benchmark_name],
        search_space=search_space_per_benchmark[benchmark_name],
    )
    return fig, ax


def _n_param_panel(benchmark_name, surrogate, tier_idx, predictor_type):
    true_pareto_info = true_pareto_info_per_benchmark[benchmark_name]
    fig, ax = plot_predicted_n_param_over_flops(
        base_path=get_results_dir(surrogate) / benchmark_name,
        optimizers=optimizers_dict,
        target_flop=target_flop_list_per_benchmark[benchmark_name][tier_idx],
        predictor_type=predictor_type,
        key_to_extract="cost",
        loss_key=loss_key,
        flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
        true_pareto_info=true_pareto_info,
        return_split=False,
        normalized_flops=False,
        abs_param_diff=true_pareto_info is not None,
        cost_estimator=cost_estimator_per_benchmark[benchmark_name],
        search_space=search_space_per_benchmark[benchmark_name],
        params_estimator=params_estimator_per_benchmark[benchmark_name],
    )
    return fig, ax


def _data_ylim(ax):
    """y-range spanned by the plotted curves and shaded bands (the copied source ylim can clip them)."""
    ys = [line.get_ydata() for line in ax.get_lines()]
    ys += [path.vertices[:, 1] for coll in ax.collections for path in coll.get_paths()]
    ys = np.concatenate([np.asarray(y, dtype=float).ravel() for y in ys])
    ys = ys[np.isfinite(ys) & ((ys > 0) if ax.get_yscale() == "log" else True)]
    return ys.min(), ys.max()


def _save_merged(panels, output_path):
    """Merge {title: (fig, ax)} into one row with a common y-range, so panels compare directly."""
    fig, axes = merge_pareto_plots(panels)
    ylims = [_data_ylim(ax) for ax in axes]
    lo, hi = min(lo for lo, _ in ylims), max(hi for _, hi in ylims)
    pad = (hi / lo) ** 0.05 if axes[0].get_yscale() == "log" else None
    for ax in axes:
        ax.set_ylim(*((lo / pad, hi * pad) if pad else (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    benchmarks = [
        BenchmarkEnum.TABPFN.value,
    ]

    plots_dir = RESULTS_BASE_DIR / "plots" / "surrogate_comparison"
    if target:
        plots_dir = plots_dir / target
    plots_dir.mkdir(parents=True, exist_ok=True)

    tier_idx = target_flop_multipliers.index(TIER)

    for benchmark_name in benchmarks:
        for panel_fn, predictors, quantity in [
            (_loss_panel, LOSS_PREDICTORS, "loss"),
            (_n_param_panel, N_PARAM_PREDICTORS, "n_param"),
        ]:
            _save_merged(
                {predictor_type.display_label: panel_fn(benchmark_name, SURROGATE, tier_idx, predictor_type)
                 for predictor_type in predictors},
                plots_dir / f"predicted_{quantity}_{benchmark_name}_{SURROGATE.value}_{TIER}x.pdf",
            )
