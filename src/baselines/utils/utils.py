from neps.optimizers.scaling_law_guided import _fit_loglog_line
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
def _create_scaling_plot(xs: list, ys: list, xlabel: str, ylabel: str, title: str) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        
        # Try to fit a line
        fit_result = _fit_loglog_line(xs, ys)
        if fit_result is not None:
            slope, intercept, fitted_ys = fit_result
            ax.plot(xs, fitted_ys, 'r--', linewidth=2, alpha=0.8, label=f'Fit (slope={slope:.3f})')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        return fig