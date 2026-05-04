from dataclasses import dataclass
from enum import Enum
from typing import cast
import numpy as np

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
# from src.baselines.chinchilla_guided_scaling import ScalingLawParameters
from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling
from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit

class ScaleParamPredictorTypes(Enum):
    def __new__(cls, label: str, color: str):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.color = color
        return obj

    POWER_LAW_FIT = ("Power Law Fit", "#1f77b4")
    CHINCHILLA3 = ("Chinchilla Approach3", "#ff7f0e")

@dataclass
class ScaleParamFitInfo:
    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    # num_pareto_points: int
    # predicted_loss: float
    # predicted_loss_upper: float
    # predicted_loss_lower: float


@dataclass
class ScaleParamChinchillaFitInfo:
    a: float
    b: float

def _flatten_rows(
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
) -> list[tuple[float, float, float, float]]:
    if not rows:
        return []

    first = rows[0]
    if isinstance(first, tuple):
        return list(rows)  # type: ignore[arg-type]

    flattened: list[tuple[float, float, float, float]] = []
    for curve in rows:  # type: ignore[assignment]
        if not isinstance(curve, (list, tuple)):
            continue
        for point in curve:
            if isinstance(point, tuple) and len(point) == 4:
                flattened.append(point)
    return flattened


def _last_point_in_curve(
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
) -> list[tuple[float, float, float, float]]:
    if not rows:
        return []

    first = rows[0]
    if isinstance(first, tuple):
        return list(rows)  # type: ignore[arg-type]

    flattened: list[tuple[float, float, float, float]] = []
    for curve in rows:  # type: ignore[assignment]
        if not isinstance(curve, (list, tuple)):
            continue
        if curve and isinstance(curve[-1], tuple) and len(curve[-1]) == 4:
            flattened.append(curve[-1])
        elif isinstance(curve, tuple) and len(curve) == 4:
            flattened.append(curve)
    return flattened


def prepare_rows_for_scale_param_prediction(
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Prepare rows for power law fit; interpolate only for grouped learning-curve rows."""
    all_rows = _flatten_rows(rows)

    if rows and isinstance(rows[0], tuple):
        unique_flops_min_loss: dict[float, tuple[float, float, float, float]] = {}
        for row in all_rows:
            n_param, n_data, loss, flop = row
            if not (np.isfinite(n_param) and np.isfinite(n_data) and np.isfinite(loss) and np.isfinite(flop)):
                continue
            if n_param <= 0 or n_data <= 0 or loss <= 0 or flop <= 0:
                continue

            existing = unique_flops_min_loss.get(float(flop))
            if existing is None or float(loss) < float(existing[2]):
                unique_flops_min_loss[float(flop)] = (float(n_param), float(n_data), float(loss), float(flop))

        selected_rows = [unique_flops_min_loss[f] for f in sorted(unique_flops_min_loss.keys())]
        return selected_rows, all_rows

    if not rows:
        return [], []

    grouped_rows = cast(list[list[tuple[float, float, float, float]]], rows)

    unique_flops_min_loss: dict[float, tuple[float, float, float, float]] = {}
    for curve in grouped_rows:
        if not curve:
            continue
        last_point = curve[-1]
        if len(last_point) != 4:
            continue
        flop_key = float(last_point[3])
        if np.isfinite(flop_key) and flop_key > 0 and flop_key not in unique_flops_min_loss:
            unique_flops_min_loss[flop_key] = (np.nan, np.nan, np.inf, flop_key)

    if not unique_flops_min_loss:
        return [], all_rows

    target_flops = sorted(unique_flops_min_loss.keys())

    for curve in grouped_rows:
        if not curve:
            continue

        curve_arr = np.asarray(curve, dtype=float)
        if curve_arr.ndim != 2 or curve_arr.shape[1] != 4:
            continue

        n_params = curve_arr[:, 0]
        losses = curve_arr[:, 2]
        flops = curve_arr[:, 3]

        mask = np.isfinite(n_params) & np.isfinite(losses) & np.isfinite(flops) & (n_params > 0) & (losses > 0) & (flops > 0)
        if np.sum(mask) < 2:
            continue

        curve_n_params = n_params[mask]
        curve_losses = losses[mask]
        curve_flops = flops[mask]

        min_flop = float(np.min(curve_flops))
        max_flop = float(np.max(curve_flops))
        in_range_targets = [f for f in target_flops if min_flop <= f <= max_flop]
        if not in_range_targets:
            continue

        fit_result = fit_power_law(curve_flops, curve_losses)
        if fit_result is None:
            continue

        n_param_ref = float(np.median(curve_n_params))
        if not np.isfinite(n_param_ref) or n_param_ref <= 0:
            continue

        for target_flop in in_range_targets:
            pred_loss, _pred_upper, _pred_lower = predict_from_loglog_fit(
                fit=fit_result,
                target_flop=float(target_flop),
            )
            if not np.isfinite(pred_loss):
                continue

            n_data = float(target_flop / (6.0 * n_param_ref))
            if n_data <= 0 or not np.isfinite(n_data):
                continue

            current_best = unique_flops_min_loss[target_flop]
            if pred_loss < current_best[2]:
                unique_flops_min_loss[target_flop] = (n_param_ref, n_data, float(pred_loss), float(target_flop))

    selected_rows = [v for v in unique_flops_min_loss.values() if np.isfinite(v[0]) and np.isfinite(v[1]) and np.isfinite(v[2])]
    return selected_rows, all_rows


def predict_scale_parameters(
    predictor_type: ScaleParamPredictorTypes,
    target_flop: float,
    *,
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
    best_loss_guess: float,
) -> tuple[tuple[float, float], ScaleParamFitInfo | ScaleParamChinchillaFitInfo]:
    """Predict loss for a single rows set and return the scalar prediction plus fit info."""

    if predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT:
        rows_for_fit, _all_rows = prepare_rows_for_scale_param_prediction(rows)
        if target_flop <= 0:
            raise ValueError("target_flop must be positive for power-law log-log scaling")

        rows_arr = np.asarray(rows_for_fit, dtype=float)
        if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
            raise ValueError("rows must be a 2D array-like with shape (n, 4)")

        params_arr = rows_arr[:, 0]
        data_arr = rows_arr[:, 1]
        flops_arr = rows_arr[:, 3]
        mask = (
            np.isfinite(params_arr)
            & np.isfinite(data_arr)
            & np.isfinite(flops_arr)
            & (params_arr > 0)
            & (data_arr > 0)
            & (flops_arr > 0)
        )
        if not np.any(mask):
            raise ValueError("rows do not contain any finite positive loss/flop values")

        fit_result = fit_power_law(
            np.asarray(flops_arr[mask], dtype=float),
            np.asarray(params_arr[mask], dtype=float),
        )
        if fit_result is None:
            raise ValueError("Could not fit power-law predictor from the provided rows")

        predicted_param, _predicted_param_upper, _predicted_param_lower = predict_from_loglog_fit(
            fit=fit_result,
            target_flop=target_flop,
            z_score=1.96,
        )

        info = ScaleParamFitInfo(
            slope=fit_result.slope,
            intercept=fit_result.intercept,
            r_squared=fit_result.r_squared,
            slope_std_err=fit_result.slope_std_err,
        )
        predicted_n_param = float(predicted_param)
        if not np.isfinite(predicted_n_param) or predicted_n_param <= 0:
            raise ValueError("Could not predict a positive n_param from power-law fit")
        predicted_n_data = float(target_flop / (6.0 * predicted_n_param))
        if not np.isfinite(predicted_n_data) or predicted_n_data <= 0:
            raise ValueError("Could not derive a positive n_data from predicted n_param")

        return (predicted_n_param, predicted_n_data), info

    if predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
        if target_flop <= 0:
            raise ValueError("target_flop must be positive for Chinchilla prediction")
        if rows[0] and isinstance(rows[0], list):
            rows = _last_point_in_curve(rows)
        rows_arr = np.asarray(rows, dtype=float)
        if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
            raise ValueError("rows must be a 2D array-like with shape (n, 4)")

        n_params_arr = rows_arr[:, 0]
        n_tokens_arr = rows_arr[:, 1]
        losses_arr = rows_arr[:, 2]
        flops_arr = rows_arr[:, 3]

        mask = (
            np.isfinite(n_params_arr)
            & np.isfinite(n_tokens_arr)
            & np.isfinite(losses_arr)
            & np.isfinite(flops_arr)
            & (n_params_arr > 0)
            & (n_tokens_arr > 0)
            & (losses_arr > 0)
            & (flops_arr > 0)
        )
        if np.sum(mask) < 3:
            raise ValueError("Need at least 3 valid rows for Chinchilla curvature fitting")

        n_params = n_params_arr[mask]
        n_tokens = n_tokens_arr[mask]
        losses = losses_arr[mask]
        flops = flops_arr[mask]

        scaling_params = Chinchilla_Guided_Scaling.get_power_law_curvature(
            N=n_params,
            D=n_tokens,
            L=losses,
            best_loss_guess=best_loss_guess,
        )

        alpha = float(scaling_params.alpha)
        beta = float(scaling_params.beta)
        A = float(scaling_params.A)
        B = float(scaling_params.B)

        alpha_plus_beta = alpha + beta
        if alpha_plus_beta <= 0:
            raise ValueError("Invalid Chinchilla parameters: alpha + beta must be positive")
        if alpha <= 0 or beta <= 0:
            raise ValueError("Invalid Chinchilla parameters: alpha and beta must be positive")
        if A <= 0 or B <= 0:
            raise ValueError("Invalid Chinchilla parameters: A and B must be positive")

        # Closed-form optimum under compute constraint C = 6 * N * D.
        a = beta / alpha_plus_beta
        b = alpha / alpha_plus_beta
        G = ((alpha * A) / (beta * B)) ** (1.0 / alpha_plus_beta)

        c_div_6 = target_flop / 6.0
        if c_div_6 <= 0:
            raise ValueError("target_flop must be positive")

        n_target = float(G * (c_div_6 ** a))
        d_target = float((1.0 / G) * (c_div_6 ** b))

        if n_target <= 0 or d_target <= 0:
            raise ValueError("Could not derive positive optimal N and D from Chinchilla fit")

        return (n_target, d_target), ScaleParamChinchillaFitInfo(a=a, b=b)
        

    raise ValueError(f"Unknown predictor type: {predictor_type}")

