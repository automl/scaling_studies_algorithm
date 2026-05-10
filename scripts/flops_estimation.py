"""Plot FLOPs per sample vs Model parameters for gflops_mparams.json analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import inspect
import surrogate_benchmark


def plot_gflops_vs_mparams():
    """Read gflops_mparams.json and plot gflops_per_sample vs mparams with Chinchilla line."""
    
    # Get the JSON file path
    package_dir = Path(inspect.getfile(surrogate_benchmark)).parent
    json_file = package_dir / "gflops_mparams.json"
    
    if not json_file.exists():
        print(f"Error: {json_file} not found")
        return
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        gflops_mparams = json.load(f)
    
    print(f"Loaded {len(gflops_mparams)} configurations from {json_file}")
    
    # Extract data points
    mparams_list = []
    gflops_per_sample_list = []
    config_names = []
    
    for key, value in gflops_mparams.items():
        if isinstance(value, dict) and "gflops_per_sample" in value and "mparams" in value:
            mparams = float(value["mparams"])
            gflops_per_sample = float(value["gflops_per_sample"])
            
            mparams_list.append(mparams)
            gflops_per_sample_list.append(gflops_per_sample)
            config_names.append(key)
            
            print(f"{key}: mparams={mparams:.4e}, gflops_per_sample={gflops_per_sample:.4e}")
    
    if not mparams_list:
        print("Error: No valid data points found in JSON")
        return
    
    # Fit linear line to data in log-log space
    log_mparams = np.log(mparams_list)
    log_gflops = np.log(gflops_per_sample_list)
    coeffs = np.polyfit(log_mparams, log_gflops, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Create fitted line
    mparams_range_fit = np.array([min(mparams_list), max(mparams_list)])
    fitted_line = np.exp(intercept) * (mparams_range_fit ** slope)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot actual data points
    ax.scatter(mparams_list, gflops_per_sample_list, s=100, alpha=0.7, label='Actual GFLOPs/sample', color='blue', edgecolors='black', linewidth=1.5)
    
    # Plot fitted line
    ax.plot(mparams_range_fit, fitted_line, 'g-', linewidth=2.5, label=f'Fitted line (slope={slope:.3f})', alpha=0.8)
    
    # Plot Chinchilla line (6 * mparams, which has slope=1 in log-log)
    mparams_range = np.array([min(mparams_list), max(mparams_list)])
    chinchilla_line = 6 * mparams_range
    ax.plot(mparams_range, chinchilla_line, 'r--', linewidth=2.5, label='Chinchilla rule (6 × mparams, slope=1)', alpha=0.8)
    
    # Configure axes (log-log scale for better visualization)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Parameters (mparams)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPs per Sample', fontsize=12, fontweight='bold')
    ax.set_title('GFLOPs per Sample vs Model Parameters\n(with Chinchilla Rule)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # Add annotations for each point
    for i, (mparams, gflops, name) in enumerate(zip(mparams_list, gflops_per_sample_list, config_names)):
        # Only annotate if space allows
        if len(mparams_list) <= 10:
            ax.annotate(name, (mparams, gflops), fontsize=8, alpha=0.7, 
                       xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "gflops_vs_mparams.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Print fitted line information
    print(f"\nFitted Power Law: GFLOPs/sample = {np.exp(intercept):.6f} × mparams^{slope:.4f}")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept (log-space): {intercept:.4f}")
    print(f"  Coefficient (linear-space): {np.exp(intercept):.6f}")
    
    # Calculate statistics
    ratios = np.array(gflops_per_sample_list) / (6 * np.array(mparams_list))
    print(f"\nStatistics - Ratio of Actual to Chinchilla (6 × mparams):")
    print(f"  Mean: {np.mean(ratios):.4f}")
    print(f"  Std Dev: {np.std(ratios):.4f}")
    print(f"  Min: {np.min(ratios):.4f} (config: {config_names[np.argmin(ratios)]})")
    print(f"  Max: {np.max(ratios):.4f} (config: {config_names[np.argmax(ratios)]})")
    
    plt.show()


if __name__ == "__main__":
    plot_gflops_vs_mparams()
