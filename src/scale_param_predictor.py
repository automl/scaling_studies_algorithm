from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast
import numpy as np

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
# from src.baselines.chinchilla_guided_scaling import ScalingLawParameters
from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling, ScalingLawParameters
from src.scaling_law_fit import LogLogPowerLawFit, fit_power_law, predict_from_loglog_fit
import src.predictor_cache as predictor_cache

class ScaleParamPredictorTypes(Enum):
    """`value` is the on-disk slug (cache filenames, result dirs) -- never change
    it. `display_label` is what shows up in figures and can be renamed freely."""

    def __new__(cls, label: str, color: str, display_label: str | None = None):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.color = color
        obj.display_label = display_label if display_label is not None else label
        return obj

    POWER_LAW_FIT = ("Direct $N$", "#1f77b4", "Power Law fit")
    CHINCHILLA3 = ("Derived $N$", "#ff7f0e", "Chinchilla3")
    GP_POSTERIOR = ("GP Posterior", "#2ca02c")

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


@dataclass
class ScaleParamFit:
    """A target-flop-independent fit for POWER_LAW_FIT/CHINCHILLA3/GP_POSTERIOR.

    Call `.predict(target_flop)` to cheaply extrapolate (n_param, n_data) for
    any compute budget without refitting -- for GP_POSTERIOR this loads the
    trained model (already held on `gp_fit`) and draws a fresh acquisition
    point for the target.
    """

    predictor_type: ScaleParamPredictorTypes
    power_law_fit: LogLogPowerLawFit | None = None
    chinchilla_params: ScalingLawParameters | None = None
    gp_fit: Any = None  # GPPosteriorFit, GP_POSTERIOR only
    params_estimator: Any = None  # needed at predict time for GP_POSTERIOR

    def predict(self, target_flop: float) -> tuple[tuple[float, float], "ScaleParamFitInfo | ScaleParamChinchillaFitInfo | Any"]:
        if self.predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT:
            assert self.power_law_fit is not None
            return _predict_power_law_scale_param(self.power_law_fit, target_flop)
        if self.predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
            assert self.chinchilla_params is not None
            return _predict_chinchilla_scale_param(self.chinchilla_params, target_flop)
        if self.predictor_type == ScaleParamPredictorTypes.GP_POSTERIOR:
            assert self.gp_fit is not None
            assert self.params_estimator is not None
            best_config, _, info = self.gp_fit.predict(target_flop)
            n_param = float(self.params_estimator(**best_config))
            if not np.isfinite(n_param) or n_param <= 0:
                raise ValueError("params_estimator returned a non-positive value for the GP-predicted config")
            n_data = float(target_flop / (6.0 * n_param))
            if not np.isfinite(n_data) or n_data <= 0:
                raise ValueError("Could not derive a positive n_data from GP-predicted n_param")
            return (n_param, n_data), info
        raise ValueError(f"ScaleParamFit.predict does not support predictor type: {self.predictor_type}")


@dataclass
class ScaleParamFitTrajectory:
    """One `ScaleParamFit` per trial-prefix n=1..len(rows)."""

    predictor_type: ScaleParamPredictorTypes
    fits: list[ScaleParamFit]

    def predict(self, target_flop: float) -> np.ndarray:
        """Return an (n_trials, 2) array of (n_param, n_data) predictions."""
        preds = np.full((len(self.fits), 2), np.nan, dtype=float)
        for i, fit in enumerate(self.fits):
            try:
                (n_param, n_data), _info = fit.predict(target_flop)
                preds[i, 0] = float(n_param)
                preds[i, 1] = float(n_data)
            except Exception:
                pass
        return preds


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


def _fit_scale_param(
    predictor_type: ScaleParamPredictorTypes,
    *,
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float] | None = None,
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    params_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> ScaleParamFit:
    """Fit a scaling law from observed rows, independent of any target_flop."""

    if predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT:
        rows_for_fit, _all_rows = prepare_rows_for_scale_param_prediction(rows)

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

        return ScaleParamFit(predictor_type=predictor_type, power_law_fit=fit_result)

    if predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
        rows_in = rows
        if rows_in[0] and isinstance(rows_in[0], list):
            rows_in = _last_point_in_curve(rows_in)
        rows_arr = np.asarray(rows_in, dtype=float)
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

        scaling_params = Chinchilla_Guided_Scaling.get_power_law_curvature(
            N=n_params,
            D=n_tokens,
            L=losses,
            best_loss_guess=best_loss_guess,
            exponent_bounds=exponent_bounds,
        )
        return ScaleParamFit(predictor_type=predictor_type, chinchilla_params=scaling_params)

    if predictor_type == ScaleParamPredictorTypes.GP_POSTERIOR:
        if space is None or configs is None or cost_estimator is None or params_estimator is None:
            raise ValueError(
                "GP_POSTERIOR requires space, configs, cost_estimator, and params_estimator kwargs"
            )
        from src.gp_posterior_predictor import fit_gp_posterior, gp_cache_path

        flat_rows = _flatten_rows(rows) if rows else []
        losses_from_rows = [float(r[2]) for r in flat_rows]
        n_rows = len(flat_rows)
        cache_path = gp_cache_path(cache_dir, n_rows) if cache_dir is not None else None
        gp_fit = fit_gp_posterior(
            space=space,
            configs=configs[:n_rows],
            losses=losses_from_rows,
            cost_estimator=cost_estimator,
            cache_path=cache_path,
            override=override,
        )
        return ScaleParamFit(predictor_type=predictor_type, gp_fit=gp_fit, params_estimator=params_estimator)

    raise ValueError(f"Unknown predictor type: {predictor_type}")


def _predict_power_law_scale_param(
    fit_result: LogLogPowerLawFit, target_flop: float
) -> tuple[tuple[float, float], ScaleParamFitInfo]:
    if target_flop <= 0:
        raise ValueError("target_flop must be positive for power-law log-log scaling")

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


def _predict_chinchilla_scale_param(
    scaling_params: ScalingLawParameters, target_flop: float
) -> tuple[tuple[float, float], ScaleParamChinchillaFitInfo]:
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

    return (n_target, d_target), ScaleParamChinchillaFitInfo(a=a, b=b)


def _fit_to_cache_row(fit: ScaleParamFit) -> list[float]:
    """Serialize a fit to a fixed-width row. A failed fit (both fields None,
    e.g. too few points at an early trial-prefix) becomes an all-NaN row."""
    if fit.predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT:
        f = fit.power_law_fit
        if f is None:
            return [float("nan")] * 6
        return [f.slope, f.intercept, f.r_squared, f.slope_std_err, f.intercept_std_err, float(f.num_points)]
    if fit.predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
        p = fit.chinchilla_params
        if p is None:
            return [float("nan")] * 5
        return [p.E, p.A, p.B, p.alpha, p.beta]
    raise ValueError(f"Cannot serialize fit for predictor type: {fit.predictor_type}")


def _fit_from_cache_row(predictor_type: ScaleParamPredictorTypes, row: np.ndarray) -> ScaleParamFit:
    """Deserialize a fit row. A NaN in the fields needed for a point prediction
    (slope/intercept for the power-law fit; all of E/A/B/alpha/beta for
    Chinchilla, since the closed-form solve uses every one) means the original
    fit failed at this trial-prefix -- e.g. `linregress` legitimately returns
    NaN r_squared/stderr (division by zero) when n_param is constant across
    the Pareto front, even though slope/intercept are still perfectly valid;
    that case must NOT be treated as a failed fit. Reconstruct an empty
    sentinel so `.predict()` raises only for genuine failures, matching the
    pre-fit failure."""
    row = np.asarray(row, dtype=float)
    if predictor_type == ScaleParamPredictorTypes.POWER_LAW_FIT:
        if np.any(np.isnan(row[:2])):
            return ScaleParamFit(predictor_type=predictor_type)
        slope, intercept, r_squared, slope_std_err, intercept_std_err, num_points = row
        fit_result = LogLogPowerLawFit(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            slope_std_err=float(slope_std_err),
            intercept_std_err=float(intercept_std_err),
            num_points=int(num_points) if np.isfinite(num_points) else 0,
        )
        return ScaleParamFit(predictor_type=predictor_type, power_law_fit=fit_result)
    if predictor_type == ScaleParamPredictorTypes.CHINCHILLA3:
        if np.any(np.isnan(row)):
            return ScaleParamFit(predictor_type=predictor_type)
        E, A, B, alpha, beta = row
        params = ScalingLawParameters(E=float(E), A=float(A), B=float(B), alpha=float(alpha), beta=float(beta))
        return ScaleParamFit(predictor_type=predictor_type, chinchilla_params=params)
    raise ValueError(f"Cannot deserialize fit for predictor type: {predictor_type}")


def get_scale_param_fit_trajectory(
    predictor_type: ScaleParamPredictorTypes,
    *,
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float],
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    params_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> ScaleParamFitTrajectory:
    """Fit (or load from cache) one model per trial-prefix n=1..len(rows).

    Caching, when `cache_dir` is given, is keyed by `(predictor_type, n_trials)`
    -- never by target_flop. Call `.predict(target_flop)` on the result to
    extrapolate cheaply to any target without refitting. Pass `override=True`
    to ignore any cached fit, refit from scratch, and overwrite the cache.
    """
    n_trials = len(rows)

    if predictor_type == ScaleParamPredictorTypes.GP_POSTERIOR:
        # A GP fit is expensive enough that each trial-prefix's model is cached
        # independently (see gp_posterior_predictor.gp_cache_path), rather than
        # bundled into one array like the cheap closed-form fits below.
        fits = []
        for n in range(1, n_trials + 1):
            try:
                fit = _fit_scale_param(
                    predictor_type,
                    rows=rows[:n],
                    best_loss_guess=best_loss_guess,
                    exponent_bounds=exponent_bounds,
                    space=space,
                    configs=configs,
                    cost_estimator=cost_estimator,
                    params_estimator=params_estimator,
                    cache_dir=cache_dir,
                    override=override,
                )
            except Exception:
                fit = ScaleParamFit(predictor_type=predictor_type)
            fits.append(fit)
        return ScaleParamFitTrajectory(predictor_type=predictor_type, fits=fits)

    if predictor_type not in (ScaleParamPredictorTypes.POWER_LAW_FIT, ScaleParamPredictorTypes.CHINCHILLA3):
        raise ValueError(f"Unknown predictor type: {predictor_type}")

    predictor_slug = str(getattr(predictor_type, "value", None) or predictor_type)

    cache_file = None
    if cache_dir is not None:
        cache_file = predictor_cache.cache_path(cache_dir, predictor_slug, "nparam")
        cached = None if override else predictor_cache.load_fit_array(cache_file, n_trials)
        if cached is not None:
            fits = [_fit_from_cache_row(predictor_type, cached["params"][i]) for i in range(n_trials)]
            return ScaleParamFitTrajectory(predictor_type=predictor_type, fits=fits)

    fits = []
    params_rows: list[list[float]] = []
    for n in range(1, n_trials + 1):
        try:
            fit = _fit_scale_param(
                predictor_type,
                rows=rows[:n],
                best_loss_guess=best_loss_guess,
                exponent_bounds=exponent_bounds,
            )
        except Exception:
            # Too few points to fit at this trial-prefix (e.g. n=1); leave a
            # failed-fit sentinel so `.predict()` yields NaN, not a hard error.
            fit = ScaleParamFit(predictor_type=predictor_type)
        fits.append(fit)
        params_rows.append(_fit_to_cache_row(fit))

    if cache_file is not None:
        predictor_cache.save_fit_array(cache_file, n_trials, params=np.asarray(params_rows, dtype=float))

    return ScaleParamFitTrajectory(predictor_type=predictor_type, fits=fits)


def predict_scale_parameters(
    predictor_type: ScaleParamPredictorTypes,
    target_flop: float,
    *,
    rows: list[tuple[float, float, float, float]] | list[list[tuple[float, float, float, float]]],
    best_loss_guess: float,
    exponent_bounds: tuple[float, float],
    space: Any = None,
    configs: list[dict] | None = None,
    cost_estimator: Any = None,
    params_estimator: Any = None,
    cache_dir: Path | None = None,
) -> tuple[tuple[float, float], ScaleParamFitInfo | ScaleParamChinchillaFitInfo | Any]:
    """Predict scale parameters for a single rows set and return the scalar prediction plus fit info."""
    fit = _fit_scale_param(
        predictor_type,
        rows=rows,
        best_loss_guess=best_loss_guess,
        exponent_bounds=exponent_bounds,
        space=space,
        configs=configs,
        cost_estimator=cost_estimator,
        params_estimator=params_estimator,
        cache_dir=cache_dir,
    )
    return fit.predict(target_flop)
