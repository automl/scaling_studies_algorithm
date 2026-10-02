import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter

from src.benchmarks.configs import BenchmarkEnum
from src.scaling_law_fit import fit_power_law
from src.plotting.plot_styles import BENCHMARK_TITLES
from src.plotting.utils import is_higher_better

logging.getLogger("fontTools").setLevel(logging.WARNING)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 20,
    "figure.figsize": (5.5, 3.5),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _sci_label(value: float, _pos: int) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    if np.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$"


PARETO_COLOR = "#D55E00"  # Okabe-Ito vermilion
DOMINATED_COLOR = "#B0B0B0"


def _log_padded(lo: float, hi: float, pad: float = 0.03) -> tuple[float, float]:
    span = np.log10(hi) - np.log10(lo) if hi > lo else 0.1
    return 10 ** (np.log10(lo) - pad * span), 10 ** (np.log10(hi) + pad * span)


def draw_empirical_pareto_front(
    ax: plt.Axes,
    *,
    true_pareto_info: dict[str, list],
    measured_points: tuple[np.ndarray, np.ndarray] | None = None,
    flop_range: tuple[float, float] | None = None,
    title: str = "Empirical Pareto Front",
) -> bool:
    """Draw the empirical Pareto front, its log-log power-law fit and, in gray, the
    other measured runs onto `ax`. Returns False if no front point is in range."""
    true_flops = np.asarray(true_pareto_info.get("flops", []), dtype=float)
    true_losses = np.asarray(true_pareto_info.get("loss", []), dtype=float)

    valid = np.isfinite(true_flops) & np.isfinite(true_losses) & (true_flops > 0) & (true_losses > 0)
    if flop_range is not None:
        valid &= (true_flops >= flop_range[0]) & (true_flops <= flop_range[1])
    true_flops = true_flops[valid]
    true_losses = true_losses[valid]
    if len(true_flops) == 0:
        return False

    order = np.argsort(true_flops)
    true_flops = true_flops[order]
    true_losses = true_losses[order]

    # Axes span only the plotted front (flop_range just selects the points), with a
    # small log-space pad so the end markers aren't clipped.
    x_lim = _log_padded(true_flops[0], true_flops[-1])
    y_extent = [true_losses.min(), true_losses.max()]

    if measured_points is not None:
        all_flops = np.asarray(measured_points[0], dtype=float)
        all_losses = np.asarray(measured_points[1], dtype=float)
        on_front = set(zip(true_flops.tolist(), true_losses.tolist()))
        dominated = (
            np.isfinite(all_flops) & np.isfinite(all_losses) & (all_losses > 0)
            & (all_flops >= x_lim[0]) & (all_flops <= x_lim[1])
            & np.array([(f, l) not in on_front for f, l in zip(all_flops.tolist(), all_losses.tolist())], dtype=bool)
        )
        if dominated.any():
            y_extent = [min(y_extent[0], all_losses[dominated].min()), max(y_extent[1], all_losses[dominated].max())]
            ax.scatter(
                all_flops[dominated],
                all_losses[dominated],
                color=DOMINATED_COLOR,
                s=10,
                alpha=0.5,
                linewidths=0,
                zorder=1,
                label="Dominated runs",
            )

    fit_result = fit_power_law(true_flops, true_losses)
    if fit_result is not None:
        fit_grid = np.logspace(np.log10(true_flops[0]), np.log10(true_flops[-1]), 100)
        fit_values = np.power(10.0, fit_result.intercept) * np.power(fit_grid, fit_result.slope)
        y_extent = [min(y_extent[0], fit_values.min()), max(y_extent[1], fit_values.max())]
        ax.plot(
            fit_grid,
            fit_values,
            color="black",
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            zorder=3,
            label="Power-law fit",
        )

    ax.scatter(
        true_flops,
        true_losses,
        color=PARETO_COLOR,
        edgecolors="black",
        linewidths=0.6,
        s=36,
        zorder=4,
        label="Empirical Pareto front",
    )

    y_lim = _log_padded(*y_extent)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("FLOPs", fontsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    # Plain decimal y-ticks: the losses often span < 1 decade, where matplotlib
    # falls back to cluttered "a x 10^b" minor labels.
    y_ticks = np.unique([float(f"{v:.3g}") for v in np.geomspace(*y_extent, 5)])
    ax.yaxis.set_major_locator(FixedLocator(y_ticks[(y_ticks >= y_lim[0]) & (y_ticks <= y_lim[1])]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Under one decade of FLOPs there'd be at most one power-of-ten label.
    if np.log10(x_lim[1] / x_lim[0]) < 1.5:
        x_ticks = np.unique([float(f"{v:.2g}") for v in np.geomspace(true_flops[0], true_flops[-1], 4)])
        ax.xaxis.set_major_locator(FixedLocator(x_ticks[(x_ticks >= x_lim[0]) & (x_ticks <= x_lim[1])]))
        ax.xaxis.set_major_formatter(FuncFormatter(_sci_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
    return True


def plot_empirical_pareto_fronts(
    *,
    true_pareto_info_per_benchmark: dict[str, dict[str, list] | None],
    measured_points_per_benchmark: dict[str, tuple[np.ndarray, np.ndarray] | None],
    flop_range_per_benchmark: dict[str, tuple[float, float]],
    benchmarks: list[str],
    loss_key: str = "objective_to_minimize",
    figsize: tuple[float, float] = (16.5, 4.0),
) -> tuple[plt.Figure, np.ndarray]:
    """One row of subplots, one empirical Pareto front per benchmark."""
    fig, axes = plt.subplots(1, len(benchmarks), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, benchmark_name in zip(axes, benchmarks):
        title = BENCHMARK_TITLES.get(benchmark_name, benchmark_name)
        true_pareto_info = true_pareto_info_per_benchmark.get(benchmark_name)
        drawn = true_pareto_info is not None and draw_empirical_pareto_front(
            ax,
            true_pareto_info=true_pareto_info,
            measured_points=measured_points_per_benchmark.get(benchmark_name),
            flop_range=flop_range_per_benchmark.get(benchmark_name),
            title=title,
        )
        if not drawn:
            ax.set_title(title, fontsize=20)
            ax.text(0.5, 0.5, "No empirical Pareto front", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()

    axes[0].set_ylabel("Error" if is_higher_better(loss_key) else "Loss", fontsize=20)
    plt.tight_layout()

    # Added after tight_layout so it doesn't shrink the axes; savefig's
    # bbox_inches="tight" grows the canvas to include it.
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), fontsize=17, frameon=True, markerscale=1.5)

    return fig, axes


if __name__ == '__main__':
    from src.extrapolate import measured_runs_from_csv
    from src.plotting.consts import results_dir, true_pareto_info_per_benchmark, \
        loss_key, flop_range_to_show_in_pareto_per_benchmark, target

    benchmarks = [
        BenchmarkEnum.OPENCLIP.value,
        BenchmarkEnum.LLM.value,
        BenchmarkEnum.TABPFN.value,
    ]
    # Same target per benchmark as the cached fronts in consts.py (TABPFN always
    # uses its own validation loss).
    target_per_benchmark = {b: (None if b == BenchmarkEnum.TABPFN.value else target) for b in benchmarks}

    measured_points_per_benchmark = {}
    for benchmark_name in benchmarks:
        try:
            _spec, _df, frame = measured_runs_from_csv(benchmark_name, target_per_benchmark[benchmark_name])
        except ValueError as e:
            print(f"  ✗ No measured runs for {benchmark_name}: {e}")
            measured_points_per_benchmark[benchmark_name] = None
            continue
        measured_points_per_benchmark[benchmark_name] = (frame["flops"].to_numpy(), frame["loss"].to_numpy())

    print("  Plotting empirical pareto fronts...")
    fig, _ = plot_empirical_pareto_fronts(
        true_pareto_info_per_benchmark=true_pareto_info_per_benchmark,
        measured_points_per_benchmark=measured_points_per_benchmark,
        flop_range_per_benchmark=flop_range_to_show_in_pareto_per_benchmark,
        benchmarks=benchmarks,
        loss_key=loss_key,
    )
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "empirical_pareto_fronts.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")
