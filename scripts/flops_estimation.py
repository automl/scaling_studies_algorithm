"""Plot FLOPs per cell vs model parameters for the tabpfn benchmark."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TABPFN_DIR = Path(
    "/home/alipourn/scaling_studies_algorithm/dep/scan_bench_suite/scan_benchmark/tabpfn"
)


def load_tabpfn_points(tabpfn_dir: Path):
    """Join n_params.json and flops.json on (embedding_size, num_layers)."""
    with open(tabpfn_dir / "n_params.json") as f:
        n_params_entries = json.load(f)
    with open(tabpfn_dir / "flops.json") as f:
        flops_entries = json.load(f)

    # A few configs are measured more than once; average their FLOPs.
    flops_by_config = defaultdict(list)
    for entry in flops_entries:
        cfg = entry["config"]
        flops_by_config[(cfg["embedding_size"], cfg["num_layers"])].append(
            float(entry["flops_per_cell"])
        )

    n_params_list = []
    flops_per_cell_list = []
    config_names = []

    for entry in n_params_entries:
        cfg = entry["model_config"]
        key = (cfg["embedding_size"], cfg["num_layers"])
        if key not in flops_by_config:
            print(f"Warning: no FLOPs entry for embedding_size={key[0]} num_layers={key[1]}")
            continue

        n_params_list.append(float(entry["n_parameters"]))
        flops_per_cell_list.append(float(np.mean(flops_by_config[key])))
        config_names.append(f"emb{key[0]}_L{key[1]}")

    return (
        np.array(n_params_list),
        np.array(flops_per_cell_list),
        config_names,
    )


def plot_flops_vs_params():
    n_params, flops_per_cell, config_names = load_tabpfn_points(TABPFN_DIR)

    if n_params.size == 0:
        print("Error: no matching (n_params, flops) pairs found")
        return

    print(f"Matched {len(n_params)} configurations")

    # Fit a power law in log-log space
    log_params = np.log(n_params)
    log_flops = np.log(flops_per_cell)
    slope, intercept = np.polyfit(log_params, log_flops, 1)

    predicted = intercept + slope * log_params
    ss_res = np.sum((log_flops - predicted) ** 2)
    ss_tot = np.sum((log_flops - np.mean(log_flops)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    params_range = np.array([n_params.min(), n_params.max()])
    fitted_line = np.exp(intercept) * (params_range**slope)
    chinchilla_line = 6 * params_range

    fig, ax = plt.subplots(figsize=(10, 7))

    embedding_sizes = np.array([int(name.split("_")[0][3:]) for name in config_names])
    scatter = ax.scatter(
        n_params,
        flops_per_cell,
        c=np.log2(embedding_sizes),
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
        label="Measured FLOPs/cell",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log2(embedding_size)", fontsize=11)

    ax.plot(
        params_range,
        fitted_line,
        "g-",
        linewidth=2.5,
        label=f"Fitted line (slope={slope:.3f})",
        alpha=0.8,
    )
    ax.plot(
        params_range,
        chinchilla_line,
        "r--",
        linewidth=2.5,
        label="Chinchilla rule (6 x N, slope=1)",
        alpha=0.8,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (N)", fontsize=12, fontweight="bold")
    ax.set_ylabel("FLOPs per Cell", fontsize=12, fontweight="bold")
    ax.set_title(
        "TabPFN: FLOPs per Cell vs Model Parameters\n(with Chinchilla Rule)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    output_path = Path(__file__).parent / "tabpfn_flops_vs_params.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")

    print(f"\nFitted Power Law: FLOPs/cell = {np.exp(intercept):.6f} x N^{slope:.4f}")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept (log-space): {intercept:.4f}")
    print(f"  Coefficient (linear-space): {np.exp(intercept):.6f}")
    print(f"  R^2 (log-log): {r_squared:.4f}")

    ratios = flops_per_cell / (6 * n_params)
    print("\nStatistics - Ratio of Actual to Chinchilla (6 x N):")
    print(f"  Mean: {np.mean(ratios):.4f}")
    print(f"  Std Dev: {np.std(ratios):.4f}")
    print(f"  Min: {np.min(ratios):.4f} (config: {config_names[int(np.argmin(ratios))]})")
    print(f"  Max: {np.max(ratios):.4f} (config: {config_names[int(np.argmax(ratios))]})")

    # The FLOPs grid is only keyed by (embedding_size, num_layers), so fit the
    # slope within each embedding size to see how much of the deviation from 1
    # comes from mixing widths and depths.
    print("\nPer-embedding-size slope (varying depth only):")
    for emb in sorted(set(embedding_sizes.tolist())):
        mask = embedding_sizes == emb
        if mask.sum() < 2:
            continue
        emb_slope, _ = np.polyfit(np.log(n_params[mask]), np.log(flops_per_cell[mask]), 1)
        print(f"  embedding_size={emb:>3}: slope={emb_slope:.4f}  ({int(mask.sum())} points)")


if __name__ == "__main__":
    plot_flops_vs_params()
