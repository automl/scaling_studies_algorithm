from dataclasses import dataclass
from enum import Enum
import numpy as np

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from src.baselines.chinchilla_guided_scaling import ScalingLawParameters
from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling
from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit

class LossPredictorTypes(Enum):
    KAPLAN = "Kaplan"
    CHINCHILLA3 = "Chinchilla Approach3"

@dataclass
class KaplanScalingLawInfo:
    slope: float
    intercept: float
    r_squared: float
    slope_std_err: float
    num_pareto_points: int
    predicted_loss: float
    predicted_loss_upper: float
    predicted_loss_lower: float


def predict_loss(
    predictor_type: LossPredictorTypes,
    target_flop: float,
    *,
    rows: list[tuple[float, float, float, float]],
    best_loss_guess: float,
) -> tuple[float, KaplanScalingLawInfo | ScalingLawParameters]:
    """Predict loss for a single rows set and return the scalar prediction plus fit info."""

    if predictor_type == LossPredictorTypes.KAPLAN:
        if target_flop <= 0:
            raise ValueError("target_flop must be positive for Kaplan log-log scaling")

        rows_arr = np.asarray(rows, dtype=float)
        if rows_arr.ndim != 2 or rows_arr.shape[1] != 4:
            raise ValueError("rows must be a 2D array-like with shape (n, 4)")

        losses_arr = rows_arr[:, 2]
        flops_arr = rows_arr[:, 3]
        mask = (
            np.isfinite(losses_arr)
            & np.isfinite(flops_arr)
            & (losses_arr > 0)
            & (flops_arr > 0)
        )
        if not np.any(mask):
            raise ValueError("rows do not contain any finite positive loss/flop values")

        res = ScalingLawGuidedOptimizer.compute_search_pareto_front(
            losses_arr[mask].tolist(),
            flops_arr[mask].tolist(),
        )
        if not res:
            raise ValueError("Could not compute a Pareto front from the provided rows")

        pareto_flops, pareto_losses, pareto_indices = res
        fit_result = fit_power_law(
            np.asarray(pareto_flops, dtype=float),
            np.asarray(pareto_losses, dtype=float),
        )
        if fit_result is None:
            raise ValueError("Could not fit Kaplan predictor from the provided rows")

        predicted_loss, predicted_loss_upper, predicted_loss_lower = predict_from_loglog_fit(
            fit=fit_result,
            target_flop=target_flop,
            z_score=1.96,
        )

        info = KaplanScalingLawInfo(
            slope=fit_result.slope,
            intercept=fit_result.intercept,
            r_squared=fit_result.r_squared,
            slope_std_err=fit_result.slope_std_err,
            num_pareto_points=fit_result.num_points,
            predicted_loss=predicted_loss,
            predicted_loss_upper=predicted_loss_upper,
            predicted_loss_lower=predicted_loss_lower,
        )
        return predicted_loss, info

    if predictor_type == LossPredictorTypes.CHINCHILLA3:
        if target_flop <= 0:
            raise ValueError("target_flop must be positive for Chinchilla prediction")

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

        res = ScalingLawGuidedOptimizer.compute_search_pareto_front(
            losses.tolist(),
            flops.tolist(),
        )
        if not res:
            raise ValueError("Could not compute a Pareto front from the provided rows")

        pareto_flops, pareto_losses, pareto_indices = res

        # pareto_flops = flops[pareto_indices]
        pareto_losses = losses[pareto_indices]
        pareto_n_params = n_params[pareto_indices]
        pareto_n_tokens = n_tokens[pareto_indices]
        

        scaling_params = Chinchilla_Guided_Scaling.get_power_law_curvature(
            N=pareto_n_params,
            D=pareto_n_tokens,
            L=pareto_losses,
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

        predicted_loss = float(
            scaling_params.E
            + scaling_params.A / (n_target ** scaling_params.alpha)
            + scaling_params.B / (d_target ** scaling_params.beta)
        )

        return predicted_loss, scaling_params
        

    raise ValueError(f"Unknown predictor type: {predictor_type}")

