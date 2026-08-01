from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from src.baselines.chinchilla_guided_scaling import ScalingLawParameters
from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling
from src.scaling_law_fit import LogLogPowerLawFit, fit_power_law, predict_from_loglog_fit
import src.predictor_cache as predictor_cache

class LossPredictorTypes(Enum):
    """`value` is the on-disk slug (cache filenames, result dirs) -- never change
    it. `display_label` is what shows up in figures and can be renamed freely."""

    def __new__(cls, label: str, color: str, display_label: str | None = None):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.color = color
        obj.display_label = display_label if display_label is not None else label
        return obj

    KAPLAN = ("Indep. PL", "#1f77b4", "Kaplan")
    CHINCHILLA3 = ("Joint Param.", "#ff7f0e", "Chinchilla3")
    GP_POSTERIOR = ("GP Posterior", "#2ca02c")

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


@dataclass
class LossFit:
    """A target-flop-independent fit for KAPLAN/CHINCHILLA3/GP_POSTERIOR.

    Call `.predict(target_flop)` to cheaply extrapolate to any compute budget
    without refitting -- for GP_POSTERIOR this loads the trained model (already
    held on `gp_fit`) and draws a fresh acquisition point for the target.
    """

    predictor_type: LossPredictorTypes
    kaplan_fit: LogLogPowerLawFit | None = None
    chinchilla_params: ScalingLawParameters | None = None
    gp_fit: Any = None  # GPPosteriorFit, GP_POSTERIOR only

    def predict(self, target_flop: float) -> tuple[float, "KaplanScalingLawInfo | ScalingLawParameters | Any"]:
        if self.predictor_type == LossPredictorTypes.KAPLAN:
            assert self.kaplan_fit is not None
            return _predict_kaplan(self.kaplan_fit, target_flop)
        if self.predictor_type == LossPredictorTypes.CHINCHILLA3:
            assert self.chinchilla_params is not None
            return _predict_chinchilla_loss(self.chinchilla_params, target_flop)
        if self.predictor_type == LossPredictorTypes.GP_POSTERIOR:
            assert self.gp_fit is not None
            _best_config, predicted_loss, info = self.gp_fit.predict(target_flop)
            return predicted_loss, info
        raise ValueError(f"LossFit.predict does not support predictor type: {self.predictor_type}")


@dataclass
class LossFitTrajectory:
    """One `LossFit` per trial-prefix n=1..len(rows)."""

    predictor_type: LossPredictorTypes
    fits: list[LossFit]

    def predict(self, target_flop: float) -> np.ndarray:
        preds = np.full(len(self.fits), np.nan, dtype=float)
        for i, fit in enumerate(self.fits):
            try:
                loss, _info = fit.predict(target_flop)
                preds[i] = float(loss)
            except Exception:
                pass
        return preds


def _fit_loss(
    predictor_type: LossPredictorTypes,
    *,
    rows: list[tuple[float, float, float, float]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None = None,
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> LossFit:
    """Fit a scaling law from observed rows, independent of any target_flop."""

    if predictor_type == LossPredictorTypes.KAPLAN:
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

        return LossFit(predictor_type=predictor_type, kaplan_fit=fit_result)

    if predictor_type == LossPredictorTypes.CHINCHILLA3:
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
            exponent_bounds=exponent_bounds,
        )
        return LossFit(predictor_type=predictor_type, chinchilla_params=scaling_params)

    if predictor_type == LossPredictorTypes.GP_POSTERIOR:
        if space is None or configs is None or cost_estimator is None:
            raise ValueError(
                "GP_POSTERIOR requires space, configs, and cost_estimator kwargs"
            )
        from src.gp_posterior_predictor import fit_gp_posterior, gp_cache_path

        n = len(rows)
        cache_path = gp_cache_path(cache_dir, n) if cache_dir is not None else None
        losses_from_rows = [float(r[2]) for r in rows]
        gp_fit = fit_gp_posterior(
            space=space,
            configs=configs[:n],
            losses=losses_from_rows,
            cost_estimator=cost_estimator,
            cache_path=cache_path,
            override=override,
        )
        return LossFit(predictor_type=predictor_type, gp_fit=gp_fit)

    raise ValueError(f"Unknown predictor type: {predictor_type}")


def _predict_kaplan(fit_result: LogLogPowerLawFit, target_flop: float) -> tuple[float, KaplanScalingLawInfo]:
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for Kaplan log-log scaling")

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


def _predict_chinchilla_loss(
    scaling_params: ScalingLawParameters, target_flop: float
) -> tuple[float, ScalingLawParameters]:
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for Chinchilla prediction")

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


def _fit_to_cache_row(fit: LossFit) -> list[float]:
    """Serialize a fit to a fixed-width row. A failed fit (both fields None,
    e.g. too few points at an early trial-prefix) becomes an all-NaN row."""
    if fit.predictor_type == LossPredictorTypes.KAPLAN:
        f = fit.kaplan_fit
        if f is None:
            return [float("nan")] * 6
        return [f.slope, f.intercept, f.r_squared, f.slope_std_err, f.intercept_std_err, float(f.num_points)]
    if fit.predictor_type == LossPredictorTypes.CHINCHILLA3:
        p = fit.chinchilla_params
        if p is None:
            return [float("nan")] * 5
        return [p.E, p.A, p.B, p.alpha, p.beta]
    raise ValueError(f"Cannot serialize fit for predictor type: {fit.predictor_type}")


def _fit_from_cache_row(predictor_type: LossPredictorTypes, row: np.ndarray) -> LossFit:
    """Deserialize a fit row. A NaN in the fields needed for a point prediction
    (slope/intercept for Kaplan; all of E/A/B/alpha/beta for Chinchilla, since
    the closed-form solve uses every one) means the original fit failed at this
    trial-prefix -- e.g. `linregress` legitimately returns NaN r_squared/stderr
    (division by zero) when an HP's value is constant across the Pareto front,
    even though slope/intercept are still perfectly valid; that case must NOT
    be treated as a failed fit. Reconstruct an empty sentinel so `.predict()`
    raises only for genuine failures, matching the pre-fit failure."""
    row = np.asarray(row, dtype=float)
    if predictor_type == LossPredictorTypes.KAPLAN:
        if np.any(np.isnan(row[:2])):
            return LossFit(predictor_type=predictor_type)
        slope, intercept, r_squared, slope_std_err, intercept_std_err, num_points = row
        fit_result = LogLogPowerLawFit(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            slope_std_err=float(slope_std_err),
            intercept_std_err=float(intercept_std_err),
            num_points=int(num_points) if np.isfinite(num_points) else 0,
        )
        return LossFit(predictor_type=predictor_type, kaplan_fit=fit_result)
    if predictor_type == LossPredictorTypes.CHINCHILLA3:
        if np.any(np.isnan(row)):
            return LossFit(predictor_type=predictor_type)
        E, A, B, alpha, beta = row
        params = ScalingLawParameters(E=float(E), A=float(A), B=float(B), alpha=float(alpha), beta=float(beta))
        return LossFit(predictor_type=predictor_type, chinchilla_params=params)
    raise ValueError(f"Cannot deserialize fit for predictor type: {predictor_type}")


def get_loss_fit_trajectory(
    predictor_type: LossPredictorTypes,
    *,
    rows: list[tuple[float, float, float, float]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float],
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> LossFitTrajectory:
    """Fit (or load from cache) one model per trial-prefix n=1..len(rows).

    Caching, when `cache_dir` is given, is keyed by `(predictor_type, n_trials)`
    -- never by target_flop, since the fit itself doesn't depend on it. Call
    `.predict(target_flop)` on the result to extrapolate cheaply to any target
    without refitting. Pass `override=True` to ignore any cached fit, refit
    from scratch, and overwrite the cache with the new result.
    """
    n_trials = len(rows)

    if predictor_type == LossPredictorTypes.GP_POSTERIOR:
        # A GP fit is expensive enough that each trial-prefix's model is cached
        # independently (see gp_posterior_predictor.gp_cache_path), rather than
        # bundled into one array like the cheap closed-form fits below.
        fits = []
        for n in range(1, n_trials + 1):
            try:
                fit = _fit_loss(
                    predictor_type,
                    rows=rows[:n],
                    best_loss_guess=best_loss_guess,
                    exponent_bounds=exponent_bounds,
                    space=space,
                    configs=configs,
                    cost_estimator=cost_estimator,
                    cache_dir=cache_dir,
                    override=override,
                )
            except Exception:
                fit = LossFit(predictor_type=predictor_type)
            fits.append(fit)
        return LossFitTrajectory(predictor_type=predictor_type, fits=fits)

    if predictor_type not in (LossPredictorTypes.KAPLAN, LossPredictorTypes.CHINCHILLA3):
        raise ValueError(f"Unknown predictor type: {predictor_type}")

    predictor_slug = str(getattr(predictor_type, "value", None) or predictor_type)

    cache_file = None
    if cache_dir is not None:
        cache_file = predictor_cache.cache_path(cache_dir, predictor_slug, "loss")
        cached = None if override else predictor_cache.load_fit_array(cache_file, n_trials)
        if cached is not None:
            fits = [_fit_from_cache_row(predictor_type, cached["params"][i]) for i in range(n_trials)]
            return LossFitTrajectory(predictor_type=predictor_type, fits=fits)

    fits = []
    params_rows: list[list[float]] = []
    for n in range(1, n_trials + 1):
        try:
            fit = _fit_loss(
                predictor_type,
                rows=rows[:n],
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
            )
        except Exception:
            # Too few points to fit at this trial-prefix (e.g. n=1); leave a
            # failed-fit sentinel so `.predict()` yields NaN, not a hard error.
            fit = LossFit(predictor_type=predictor_type)
        fits.append(fit)
        params_rows.append(_fit_to_cache_row(fit))

    if cache_file is not None:
        predictor_cache.save_fit_array(cache_file, n_trials, params=np.asarray(params_rows, dtype=float))

    return LossFitTrajectory(predictor_type=predictor_type, fits=fits)


def predict_loss(
    predictor_type: LossPredictorTypes,
    target_flop: float,
    *,
    rows: list[tuple[float, float, float, float]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float],
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
) -> tuple[float, KaplanScalingLawInfo | ScalingLawParameters | Any]:
    """Predict loss for a single rows set and return the scalar prediction plus fit info."""
    fit = _fit_loss(
        predictor_type,
        rows=rows,
        best_loss_guess=best_loss_guess,
        exponent_bounds=exponent_bounds,
        space=space,
        configs=configs,
        cost_estimator=cost_estimator,
        cache_dir=cache_dir,
    )
    return fit.predict(target_flop)
