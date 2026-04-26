from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import linregress


@dataclass
class LogLogPowerLawFit:
    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    intercept_std_err: float
    num_points: int


def fit_loglog_power_law(flops: np.ndarray, losses: np.ndarray) -> LogLogPowerLawFit | None:
    """Fit log10(loss) = slope * log10(flops) + intercept with quality stats."""
    fit_x = np.asarray(flops, dtype=float)
    fit_y = np.asarray(losses, dtype=float)
    mask = (
        np.isfinite(fit_x)
        & np.isfinite(fit_y)
        & (fit_x > 0)
        & (fit_y > 0)
    )
    fit_x = fit_x[mask]
    fit_y = fit_y[mask]

    if len(fit_x) < 2 or len(np.unique(fit_x)) < 2:
        return None

    log_x = np.log10(fit_x)
    log_y = np.log10(fit_y)

    result: Any = linregress(log_x, log_y)

    return LogLogPowerLawFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue ** 2),
        slope_std_err=float(result.stderr),
        intercept_std_err=float(result.intercept_stderr),
        num_points=int(len(fit_x)),
    )


def predict_from_loglog_fit(
    fit: LogLogPowerLawFit,
    target_flop: float,
    z_score: float = 1.96,
) -> tuple[float, float, float]:
    """Predict loss at target FLOPs and return mean/upper/lower (95% CI default)."""
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for log-log prediction")

    log_target = float(np.log10(target_flop))

    predicted_log_loss = (fit.slope * log_target) + fit.intercept
    predicted_loss = float(np.power(10.0, predicted_log_loss))

    log_loss_upper = (
        (fit.slope + (z_score * fit.slope_std_err)) * log_target
        + (fit.intercept + (z_score * fit.intercept_std_err))
    )
    log_loss_lower = (
        (fit.slope - (z_score * fit.slope_std_err)) * log_target
        + (fit.intercept - (z_score * fit.intercept_std_err))
    )

    predicted_loss_upper = float(np.power(10.0, log_loss_upper))
    predicted_loss_lower = float(np.power(10.0, log_loss_lower))
    return predicted_loss, predicted_loss_upper, predicted_loss_lower
