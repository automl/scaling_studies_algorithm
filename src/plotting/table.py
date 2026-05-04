from src.plotting.plot import process_seed, _extract_result_value, _format_hp_table_value
from src.loss_predictor import LossPredictorTypes, predict_loss
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Callable
import json
from src.plotting.plot_styles import BENCHMARK_TITLES 
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotting.utils import OPTIMIZER_LABELS, get_optimizer_label, _split_optimizers_by_priors, \
    _placeholder_plot


def write_true_pareto_prediction_result(
    *,
    out_file: str | Path,
    benchmark: str,
    target_flop: float,
    true_pareto_info: dict[str, list] | None,
) -> dict[str, Any] | None:
    """Append the true-Pareto loss closest to target_flop to prediction_result JSONL."""
    if true_pareto_info is None:
        return None

    true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
    true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)
    valid_true = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
    true_flops = true_flops[valid_true]
    true_losses = true_losses[valid_true]
    if len(true_flops) == 0:
        return None

    closest_idx = int(np.argmin(np.abs(true_flops - float(target_flop))))
    closest_flop = float(true_flops[closest_idx])
    closest_loss = float(true_losses[closest_idx])
    entry: dict[str, Any] = {
        "benchmark": benchmark,
        "true_pareto": {
            "target_flop": float(target_flop),
            "flop": closest_flop,
            "loss": {"value": closest_loss},
        },
    }

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


plt.rcParams.update({
    "font.family": "serif",        # Forces Times-style font
    "font.size": 15,               # Base font size
    "axes.labelsize": 15,          # X and Y labels
    "xtick.labelsize": 8,          # X-axis numbers
    "ytick.labelsize": 8,          # Y-axis numbers
    "legend.fontsize": 15,          # Legend labels
    "figure.figsize": (6, 4),      # Single-panel default used by composite plots
    "pdf.fonttype": 42,            # Ensures fonts render correctly in the final PDF
    "ps.fonttype": 42
})
def plot_predicted_hp_table_over_flops(
    *,
    base_path: str | Path,
    optimizers: dict[str, list[int]] | dict[str, list[int | str]],
    space: Any,
    target_flop: float,
    evaluate_pipeline: Callable,
    true_pareto_info: dict[str, list] | None = None,
    flop_range: tuple[float, float] | None = None,
    predictor_type: Any = None,
    key_to_extract: str | None = None,
    loss_key: str = "objective_to_minimize",
    figsize: tuple[int, int] = (6, 4),
    title: str = "Predicted Hyperparameters at Target FLOPs",
    return_split: bool = True,
    target_flop_ratios: tuple[float, ...] = (0.5, 1.0, 5.0),
    flop_ranges_by_ratio: dict[float, tuple[float, float]] | None = None,
    selected_seed: int | str | None = None,
) -> Any:
    """Create a NeurIPS-style table figure (no vertical lines) with predicted HP configs per optimizer.

    When an optimizer has multiple seeds, table cells are aggregated across
    seeds using the mean and standard error.
    """
    if target_flop <= 0:
        raise ValueError("target_flop must be positive")

    table_optimizer_order = ("carbs", "random_search")
    filtered_optimizers = {
        opt_name: optimizers[opt_name]
        for opt_name in table_optimizer_order
        if opt_name in optimizers
    }
    if filtered_optimizers:
        optimizers = filtered_optimizers #type: ignore

    if return_split:
        all_opts, with_priors, without_priors = _split_optimizers_by_priors(optimizers)

        fig_all, ax_all = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
            base_path=base_path,
            optimizers=all_opts,
            space=space,
            evaluate_pipeline=evaluate_pipeline,
            true_pareto_info=true_pareto_info,
            target_flop=target_flop,
            flop_range=flop_range,
            predictor_type=predictor_type,
            key_to_extract=key_to_extract,
            loss_key=loss_key,
            figsize=figsize,
            title=title,
            return_split=False,
            target_flop_ratios=target_flop_ratios,
            flop_ranges_by_ratio=flop_ranges_by_ratio,
            selected_seed=selected_seed,
        ))

        if with_priors:
            fig_with, ax_with = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
                base_path=base_path,
                optimizers=with_priors,
                space=space,
                evaluate_pipeline=evaluate_pipeline,
                true_pareto_info=true_pareto_info,
                target_flop=target_flop,
                flop_range=flop_range,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (with priors)",
                return_split=False,
                target_flop_ratios=target_flop_ratios,
                flop_ranges_by_ratio=flop_ranges_by_ratio,
                selected_seed=selected_seed,
            ))
        else:
            fig_with, ax_with = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (with priors)",
                xlabel="",
                ylabel="",
            )

        if without_priors:
            fig_without, ax_without = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
                base_path=base_path,
                optimizers=without_priors,
                space=space,
                evaluate_pipeline=evaluate_pipeline,
                true_pareto_info=true_pareto_info,
                target_flop=target_flop,
                flop_range=flop_range,
                predictor_type=predictor_type,
                key_to_extract=key_to_extract,
                loss_key=loss_key,
                figsize=figsize,
                title=f"{title} (without priors)",
                return_split=False,
                target_flop_ratios=target_flop_ratios,
                flop_ranges_by_ratio=flop_ranges_by_ratio,
                selected_seed=selected_seed,
            ))
        else:
            fig_without, ax_without = _placeholder_plot(
                figsize=figsize,
                title=f"{title} (without priors)",
                xlabel="",
                ylabel="",
            )

        return (fig_all, ax_all), (fig_with, ax_with), (fig_without, ax_without)

    from src.hp_predictor import HPPredictorTypes, predict_hp_from_scaling_law

    if predictor_type is None:
        predictor_type = HPPredictorTypes.LINEAR_REGRESSION
    if not target_flop_ratios:
        raise ValueError("target_flop_ratios must not be empty")
    for ratio in target_flop_ratios:
        if ratio <= 0:
            raise ValueError("target_flop_ratios must be positive")

    base_path = Path(base_path)

    def _aggregate_seed_values(values: list[Any]) -> tuple[float, float] | Any | None:
        numeric_values: list[float] = []
        fallback_values: list[Any] = []
        for value in values:
            if value is None:
                continue
            try:
                numeric_values.append(float(value))
            except Exception:
                fallback_values.append(value)

        if numeric_values:
            arr = np.asarray(numeric_values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return None
            mean = float(np.mean(arr))
            stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return mean, stderr

        if fallback_values:
            return fallback_values[0]

        return None

    def _format_seed_aggregate(value: Any) -> str:
        if isinstance(value, tuple) and len(value) == 2:
            mean, stderr = value
            if not np.isfinite(float(mean)):
                return "nan"
            return f"{_format_hp_table_value(mean)} ± {_format_hp_table_value(stderr)}"
        return _format_hp_table_value(value)


    # Single-benchmark mode: base_path is expected to point to the benchmark folder
    base_path = Path(base_path)
    bench_name = base_path.name

    optimizer_names = list(optimizers.keys())
    # Column labels: for each ratio, list each optimizer
    col_labels: list[str] = []
    for ratio in target_flop_ratios:
        for opt in optimizer_names:
            col_labels.append(f"{_ratio_label(ratio)} {get_optimizer_label(opt)}")

    cell_text: list[list[str]] = []

    # We'll also write per-ratio JSON lines to base_path.parent()/plots/prediction_result.json
    out_dir = base_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prediction_result.json"

    # Prepare a single row for this benchmark
    row_vals: list[str] = []
    for ratio in target_flop_ratios:
        ratio_target_flop = float(ratio * target_flop)
        entry: dict[str, Any] = {"benchmark": bench_name, "ratio": float(ratio), "loss": {}}
        for opt_name in optimizer_names:
            losses_for_opt: list[float] = []
            seeds = optimizers.get(opt_name, [])
            if not seeds:
                row_vals.append("nan")
                entry["loss"][opt_name] = None
                continue
            for seed in seeds:
                try:
                    losses, cumsum_costs, _max_cost, _rows, _learning_curve_rows, configs = process_seed(
                        path=base_path / opt_name,
                        seed=seed,
                        key_to_extract=key_to_extract,
                        loss_key=loss_key,
                        return_incumbent=False,
                    )
                except Exception:
                    continue
                if not configs or not losses or not cumsum_costs:
                    continue

                cumsum_arr_full = np.asarray(cumsum_costs, dtype=float)
                ratio_flop_range = None
                if flop_ranges_by_ratio is not None:
                    ratio_flop_range = flop_ranges_by_ratio.get(float(ratio)) or flop_ranges_by_ratio.get(ratio)
                if ratio_flop_range is None and flop_range is not None:
                    ratio_flop_range = (float(flop_range[0]), ratio_target_flop)

                cutoff_lower = float(ratio_flop_range[0]) if ratio_flop_range is not None else None
                cutoff_upper = float(ratio_flop_range[1]) if ratio_flop_range is not None else ratio_target_flop

                cutoff_mask = cumsum_arr_full <= cutoff_upper
                if cutoff_lower is not None:
                    cutoff_mask &= cumsum_arr_full >= cutoff_lower
                cutoff_indices = np.where(cutoff_mask)[0]
                if len(cutoff_indices) == 0 and cutoff_lower is not None:
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
                    pred_result = predict_hp_from_scaling_law(
                        space=space,
                        hp_predictor_type=predictor_type,
                        target_flops=ratio_target_flop,
                        configs=ratio_configs,
                        losses=[float(v) for v in ratio_losses],
                        flops=[float(v) for v in actual_flops],
                    )
                    predicted_cfg = cast(dict[str, Any], pred_result.get("extrapolated_config", {}))
                    # evaluate
                    actual_loss = float(
                        _extract_result_value(
                            (evaluate_pipeline[bench_name] if isinstance(evaluate_pipeline, dict) else evaluate_pipeline)(**predicted_cfg),
                            loss_key,
                        )
                    )
                    if np.isfinite(actual_loss) and actual_loss > 0:
                        losses_for_opt.append(actual_loss)
                except Exception:
                    continue

            agg = _aggregate_seed_values(losses_for_opt)
            row_vals.append(_format_seed_aggregate(agg) if agg is not None else "nan")
            if isinstance(agg, tuple) and len(agg) == 2:
                entry["loss"][opt_name] = {"mean": float(agg[0]), "stderr": float(agg[1])}
            elif agg is None:
                entry["loss"][opt_name] = None
            else:
                entry["loss"][opt_name] = {"value": agg}

        # append JSON line for this ratio
        try:
            import json

            with open(out_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as e:
            raise e

    write_true_pareto_prediction_result(
        out_file=out_file,
        benchmark=bench_name,
        target_flop=target_flop,
        true_pareto_info=true_pareto_info,
    )

    cell_text.append(row_vals)

    # Build and return a single-table figure for this benchmark
    fig, ax = plt.subplots(figsize=(max(figsize[0], 6.0 + 1.2 * len(col_labels)), max(figsize[1], 2.0 + 0.5 * len(cell_text))))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[bench_name],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    # table.scale(1.0, 1.2)
    ax.set_title(title, fontsize=16, pad=10)
    plt.tight_layout()
    return fig, ax


def plot_predicted_hp_table_from_json(
    *,
    json_path: str | Path,
    figsize: tuple[float, float] = (6.0, 4.0),
    title: str = "",
    ratio_order: tuple[float, ...] = (0.5, 1.0, 5.0),
) -> tuple[Any, Any]:
    """Read newline-delimited JSON at `json_path` and render a table.

    Expects each line to be a JSON object with keys: `benchmark`, `ratio`, `loss`.
    `loss` is a dict mapping optimizer name to {"mean": float, "stderr": float} or None.
    """
    from pathlib import Path
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    records = []
    with open(json_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    # Organize by benchmark and collect ratios/optimizers in stable display order.
    by_bench: dict[str, dict[float, dict[str, Any]]] = {}
    true_pareto_by_bench: dict[str, Any] = {}
    ratios_seen: list[float] = []
    optimizers_seen: list[str] = []
    for r in records:
        bench = r.get("benchmark", "")
        if "true_pareto" in r:
            true_pareto = r.get("true_pareto", {}) or {}
            true_pareto_by_bench[bench] = true_pareto.get("loss")
            continue

        ratio = float(r.get("ratio", 0.0))
        losses = r.get("loss", {}) or {}
        by_bench.setdefault(bench, {})[ratio] = losses
        if ratio not in ratios_seen:
            ratios_seen.append(ratio)
        for opt in losses.keys():
            if opt not in optimizers_seen:
                optimizers_seen.append(opt)

    if not by_bench and not true_pareto_by_bench:
        raise ValueError("No valid records found in JSON file")

    ordered_ratios = [float(r) for r in ratio_order]
    ratios = ordered_ratios + sorted(r for r in ratios_seen if r not in ordered_ratios)
    optimizers = optimizers_seen
    bench_names = sorted(set(by_bench.keys()) | set(true_pareto_by_bench.keys()))

    def _format_table_number(value: Any) -> str:
        try:
            v = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(v):
            return "nan"
        return f"{v:.2f}"

    def _format_table_entry(value: Any) -> str:
        if value is None:
            return "nan"
        if isinstance(value, dict) and "mean" in value and "stderr" in value:
            return f"{_format_table_number(value['mean'])} ± {_format_table_number(value['stderr'])}"
        if isinstance(value, dict) and "value" in value:
            return _format_table_number(value["value"])
        return _format_table_number(value)

    # Matplotlib tables do not support column spanning, so the first row is a
    # benchmark group header centered over the three ratio subcolumns.
    header_group = [""]
    header_ratio = ["Strategy"]
    for bench in bench_names:
        for ratio in ratios:
            header_group.append("")
            header_ratio.append(_ratio_label(ratio))
        if bench in true_pareto_by_bench:
            header_group.append("")
            header_ratio.append("Emp. Best")

    cell_text: list[list[str]] = []
    for opt in optimizers:
        row = [OPTIMIZER_LABELS.get(opt, get_optimizer_label(opt))]
        for bench in bench_names:
            for ratio in ratios:
                losses = by_bench.get(bench, {}).get(ratio, {}) or {}
                row.append(_format_table_entry(losses.get(opt)))
            if bench in true_pareto_by_bench:
                row.append(_format_table_entry(true_pareto_by_bench.get(bench)))
        cell_text.append(row)

    table_text = [header_group, header_ratio, *cell_text]
    n_cols = len(header_group)
    fig_width = max(figsize[0], 0.65 + 1.6 * n_cols)
    fig_height = max(figsize[1], 1.0 + 0.58 * len(table_text))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_text,
        cellLoc="center",
        bbox=[0.01, 0.08, 0.98, 0.84], #type: ignore
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(0.95, 1.65)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        if row == 0 and col == 0:
            cell.get_text().set_text("")
        if row == 0:
            cell.visible_edges = "T"
            cell.set_linewidth(2)
        elif row == 1:
            cell.visible_edges = "B"
            cell.set_linewidth(1)
        if col == 0:
            cell.set_text_props(ha="left")

    last_row = len(table_text) - 1
    for col in range(n_cols):
        cell = table[(last_row, col)]
        cell.visible_edges = "".join(sorted(set(cell.visible_edges + "B")))
        cell.set_linewidth(1.0)

    # Draw inset cmidrule-style underlines manually. Cell borders are stitched
    # by Matplotlib, so using visible_edges would create one full-width rule.
    fig.canvas.draw()
    for bench_idx, _bench in enumerate(bench_names):
        start_col = 1
        for previous_bench in bench_names[:bench_idx]:
            start_col += len(ratios) + (1 if previous_bench in true_pareto_by_bench else 0)
        end_col = start_col + len(ratios) + (1 if _bench in true_pareto_by_bench else 0) - 1
        start_cell = table[(0, start_col)]
        end_cell = table[(0, end_col)]
        x_start = start_cell.get_x()
        x_end = end_cell.get_x() + end_cell.get_width()
        y = start_cell.get_y()
        y_center = y + 0.5 * start_cell.get_height()
        pad = 0.06 * (x_end - x_start)
        ax.text(
            0.5 * (x_start + x_end),
            y_center,
            BENCHMARK_TITLES.get(_bench, _bench),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
        )
        ax.add_line(Line2D(
            [x_start + pad, x_end - pad],
            [y, y],
            transform=ax.transAxes,
            color="black",
            linewidth=1,
            solid_capstyle="butt",
        ))

    if title:
        ax.set_title(title, fontsize=15, pad=10)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04)
    return fig, ax


def _ratio_label(ratio: float) -> str:
    if abs(ratio - round(ratio)) < 1e-8:
        return f"{int(round(ratio))}x"
    return f"{ratio:g}x"

if __name__ == "__main__":
    from src.hp_predictor import HPPredictorTypes
    from src.benchmarks.configs import BenchmarkEnum
    from src.plotting.consts import results_dir, bench_confs
    from src.benchmarks.configs import BenchmarkEnum
    from src.benchmarks.llm_space import LLM_TARGETS
    from src.benchmarks.openclip_space import OPENCLIP_TARGETS
    from src.benchmarks.tabpfn_space import TABPFN_TARGETS
    from src.plotting.plot_styles import BENCHMARK_TITLES
    from src.extrapolate import load_or_generate_pareto_info
    from src.loss_predictor import LossPredictorTypes
    from src.scale_param_predictor import ScaleParamPredictorTypes
    from src.benchmarks.configs import get_benchmark
    from src.plotting.consts import target_flop_per_benchmark, \
    bench_confs, true_pareto_info_per_benchmark, acc_pred_flop_ranges_per_benchmark_by_ratio,\
        hp_target_flop_ratios, \
            selected_best_seed_per_benchmark, acc_pred_flop_ranges_per_benchmark, acc_flop_ranges_per_benchmark, \
            all_seeds, results_dir, optimizers_dict, loss_key
    benchmarks = [
        BenchmarkEnum.OPENCLIP.value, 
        BenchmarkEnum.LLM.value,
    ]
    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {benchmark_name}")
        print(f"{'='*60}")
        
        plot_output_dir = results_dir / benchmark_name / "plots"
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        
        benchmark_cfg = bench_confs[benchmark_name]
        target_flop = target_flop_per_benchmark[benchmark_name]


        # out_file = results_dir / "plots" / "prediction_result.json"
        # write_true_pareto_prediction_result(
        #     out_file=out_file,
        #     benchmark=benchmark_name,
        #     target_flop=target_flop,
        #     true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
        # )
        # for hp_predictor_type in [HPPredictorTypes.LINEAR_REGRESSION]:
            # hp_predictor_slug = hp_predictor_type.value.lower().replace(" ", "_")
            # print(f"  Building {hp_predictor_type.value} predicted HP table at FLOP: {target_flop:.2e}...")
            # fig_hp_all, _ = cast(tuple[Any, Any], plot_predicted_hp_table_over_flops(
            #     base_path=results_dir / benchmark_name,
            #     optimizers=optimizers_dict,
            #     space=convert_neps_to_classic_search_space(benchmark_cfg.space),
            #     evaluate_pipeline=benchmark_cfg.evaluate_pipeline,
            #     true_pareto_info=true_pareto_info_per_benchmark[benchmark_name],
            #     target_flop=target_flop,
            #     flop_range=acc_pred_flop_ranges_per_benchmark[benchmark_name],
            #     predictor_type=hp_predictor_type,
            #     key_to_extract="cost",
            #     loss_key=loss_key,
            #     title=f"{BENCHMARK_TITLES[benchmark_name]} ({hp_predictor_type.value})",
            #     return_split=False,
            #     target_flop_ratios=hp_target_flop_ratios,
            #     flop_ranges_by_ratio=acc_pred_flop_ranges_per_benchmark_by_ratio[benchmark_name],
            #     selected_seed=selected_best_seed_per_benchmark[benchmark_name],
            # ))
            # fig_hp_all.savefig(plot_output_dir / f"{hp_predictor_slug}_predicted_hp_table_all_{target_flop:.2e}.pdf", dpi=300, bbox_inches="tight")
            # plt.close(fig_hp_all)
            # print(f"    ✓ Saved: {plot_output_dir / f'{hp_predictor_slug}_predicted_hp_table_all_{target_flop:.2e}.pdf'}")

    from src.plotting.consts import results_dir
    plots_dir = results_dir / "plots"
    pred_hp_fig, ax = plot_predicted_hp_table_from_json(json_path=plots_dir / "prediction_result.json")
    pred_hp_fig.savefig(plots_dir / f"predicted_hp_table_composite_all.pdf", dpi=300, bbox_inches="tight")
    plt.close(pred_hp_fig)
    print(f"✓ Saved: {plots_dir / 'predicted_hp_table_composite_all.pdf'}")
