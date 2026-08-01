from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from typing import Any

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.space.search_space import SearchSpace, HPOInteger, HPOFloat
from src.scaling_law_fit import LogLogPowerLawFit, fit_power_law, predict_from_loglog_fit
import src.predictor_cache as predictor_cache

class HPPredictorTypes(Enum):
    LINEAR_REGRESSION = "Linear Regression"
    GP_POSTERIOR = "GP Posterior"


@dataclass
class HPFit:
    """A target-flop-independent fit for LINEAR_REGRESSION/GP_POSTERIOR.

    Call `.predict(target_flops)` to cheaply extrapolate the full HP config for
    any compute budget without refitting -- for GP_POSTERIOR this loads the
    trained model (already held on `gp_fit`) and draws a fresh acquisition
    point for the target.
    """

    predictor_type: HPPredictorTypes
    space: SearchSpace
    linear_fits: dict[str, LogLogPowerLawFit] | None = None
    pareto_ids: list[int] | None = None
    pareto_configs: list[dict[str, Any]] | None = None
    pareto_flops: list[float] | None = None
    pareto_losses: list[float] | None = None
    gp_fit: Any = None  # GPPosteriorFit, GP_POSTERIOR only

    def predict(self, target_flops: float) -> dict[str, Any]:
        if self.predictor_type == HPPredictorTypes.LINEAR_REGRESSION:
            assert self.linear_fits is not None
            return _predict_linear_regression_hp(
                space=self.space,
                linear_fits=self.linear_fits,
                pareto_ids=self.pareto_ids or [],
                pareto_configs=self.pareto_configs or [],
                pareto_flops=self.pareto_flops or [],
                pareto_losses=self.pareto_losses or [],
                target_flops=target_flops,
            )
        if self.predictor_type == HPPredictorTypes.GP_POSTERIOR:
            assert self.gp_fit is not None
            best_config, predicted_loss, info = self.gp_fit.predict(target_flops)
            return {
                "extrapolated_config": best_config,
                "predicted_loss": predicted_loss,
                "gp_posterior_info": info,
                "pareto_ids": [],
                "pareto_configs": [],
                "pareto_flops": [],
                "pareto_losses": [],
            }
        raise ValueError(f"HPFit.predict does not support predictor type: {self.predictor_type}")


@dataclass
class HPFitTrajectory:
    """One `HPFit` per trial-prefix n=1..len(configs)."""

    predictor_type: HPPredictorTypes
    fits: list[HPFit]

    def predict(self, target_flops: float) -> list[dict[str, Any]]:
        """Return one `HPFit.predict` result dict per trial-prefix n."""
        results: list[dict[str, Any]] = []
        for fit in self.fits:
            try:
                results.append(fit.predict(target_flops))
            except Exception:
                results.append({
                    "extrapolated_config": {},
                    "pareto_ids": [],
                    "pareto_configs": [],
                    "pareto_flops": [],
                    "pareto_losses": [],
                })
        return results


def _build_pareto_indices(
    *,
    losses: list[float],
    flops: list[float],
    pareto_losses: list[float],
    pareto_flops: list[float],
    tol: float = 1e-12,
) -> list[int]:
    """Map Pareto front points back to source row indices using value matching."""
    used_indices: set[int] = set()
    pareto_indices: list[int] = []

    for p_flop, p_loss in zip(pareto_flops, pareto_losses):
        chosen_idx: int | None = None
        for i, (f, l) in enumerate(zip(flops, losses)):
            if i in used_indices:
                continue
            if np.isclose(float(f), float(p_flop), rtol=0.0, atol=tol) and np.isclose(float(l), float(p_loss), rtol=0.0, atol=tol):
                chosen_idx = i
                break
        if chosen_idx is None:
            raise ValueError(
                "Could not map Pareto point back to source inputs. "
                "Check for inconsistent losses/flops arrays."
            )
        used_indices.add(chosen_idx)
        pareto_indices.append(chosen_idx)

    return pareto_indices


def _compute_hp_pareto(
    *,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
) -> tuple[list[int], list[dict[str, Any]], list[float], list[float]]:
    pareto_result = ScalingLawGuidedOptimizer.compute_search_pareto_front(
        losses=[float(v) for v in losses],
        flops_list=[float(v) for v in flops],
    )
    if pareto_result is None:
        raise ValueError("Could not compute Pareto front from losses/flops")

    pareto_flops, pareto_losses, _pareto_indices = pareto_result
    pareto_ids = _build_pareto_indices(
        losses=[float(v) for v in losses],
        flops=[float(v) for v in flops],
        pareto_losses=pareto_losses,
        pareto_flops=pareto_flops,
    )
    pareto_configs = [configs[i] for i in pareto_ids]
    return pareto_ids, pareto_configs, pareto_flops, pareto_losses


def _fit_linear_regression_hp(
    *,
    space: SearchSpace,
    pareto_configs: list[dict[str, Any]],
    pareto_flops: list[float],
) -> dict[str, LogLogPowerLawFit]:
    """Fit one power law per searchable HP, independent of any target_flops."""
    parameters = space.searchables
    x_flops = np.asarray(pareto_flops, dtype=float)

    linear_fits: dict[str, LogLogPowerLawFit] = {}
    for param_name, param in parameters.items():
        hp_values_param: list[float] = []
        for config in pareto_configs:
            if param_name not in config:
                raise ValueError(f"HP '{param_name}' is missing from config: {config}")
            hp_values_param.append(float(config[param_name]))

        # Only the parameter's own declared log flag decides the fit space.
        # `is_scaling` used to force it too, which put ordinal-index HPs (e.g.
        # OpenCLIP's vision_width/text_width, indices into WIDTH_CHOICES) in log
        # space, where a power law in the index is not a power law in the value.
        log = isinstance(param, HPOInteger | HPOFloat) and param.log

        fit_result = fit_power_law(
            x_flops,
            np.asarray(hp_values_param, dtype=float),
            log_scale=log,
        )
        if fit_result is None:
            raise ValueError(f"Not enough valid data points to fit scaling law for HP '{param_name}'")
        linear_fits[param_name] = fit_result

    return linear_fits


def _fit_hp(
    hp_predictor_type: HPPredictorTypes,
    *,
    space: SearchSpace,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> HPFit:
    """Fit an HP model from observed configs/losses/flops, independent of any target_flops."""
    if len(configs) != len(losses) or len(configs) != len(flops):
        raise ValueError("configs, losses, and flops must have the same length")
    if len(configs) == 0:
        raise ValueError("At least one config/loss/flops row is required")

    if hp_predictor_type == HPPredictorTypes.LINEAR_REGRESSION:
        pareto_ids, pareto_configs, pareto_flops, pareto_losses = _compute_hp_pareto(
            configs=configs, losses=losses, flops=flops,
        )
        linear_fits = _fit_linear_regression_hp(
            space=space, pareto_configs=pareto_configs, pareto_flops=pareto_flops,
        )
        return HPFit(
            predictor_type=hp_predictor_type,
            space=space,
            linear_fits=linear_fits,
            pareto_ids=pareto_ids,
            pareto_configs=pareto_configs,
            pareto_flops=pareto_flops,
            pareto_losses=pareto_losses,
        )

    if hp_predictor_type == HPPredictorTypes.GP_POSTERIOR:
        if cost_estimator is None:
            raise ValueError("cost_estimator is required for GP_POSTERIOR predictor")
        from src.gp_posterior_predictor import fit_gp_posterior, gp_cache_path

        n = len(configs)
        cache_path = gp_cache_path(cache_dir, n) if cache_dir is not None else None
        gp_fit = fit_gp_posterior(
            space=space, configs=configs, losses=losses, cost_estimator=cost_estimator, cache_path=cache_path,
            override=override,
        )
        return HPFit(predictor_type=hp_predictor_type, space=space, gp_fit=gp_fit)

    raise ValueError(f"Unknown predictor type: {hp_predictor_type}")


def _predict_linear_regression_hp(
    *,
    space: SearchSpace,
    linear_fits: dict[str, LogLogPowerLawFit],
    pareto_ids: list[int],
    pareto_configs: list[dict[str, Any]],
    pareto_flops: list[float],
    pareto_losses: list[float],
    target_flops: float,
) -> dict[str, Any]:
    parameters = space.searchables
    extrapolated_config: dict[str, Any] = {}
    for param_name, param in parameters.items():
        fit_result = linear_fits[param_name]
        predicted_hp, _, _ = predict_from_loglog_fit(fit_result, target_flops)
        predicted_hp = np.clip(predicted_hp, param.domain.lower, param.domain.upper)
        casted_hp = param.domain.cast_one(float(predicted_hp), frm=param.domain)
        extrapolated_config[param_name] = casted_hp

    return {
        "extrapolated_config": extrapolated_config,
        "pareto_ids": pareto_ids,
        "pareto_configs": pareto_configs,
        "pareto_flops": pareto_flops,
        "pareto_losses": pareto_losses,
    }


def _loglog_to_row(fit: LogLogPowerLawFit) -> list[float]:
    return [fit.slope, fit.intercept, fit.r_squared, fit.slope_std_err, fit.intercept_std_err, float(fit.num_points)]


def _row_to_loglog(row: np.ndarray) -> LogLogPowerLawFit:
    slope, intercept, r_squared, slope_std_err, intercept_std_err, num_points = row
    return LogLogPowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        slope_std_err=float(slope_std_err),
        intercept_std_err=float(intercept_std_err),
        num_points=int(num_points) if np.isfinite(num_points) else 0,
    )


def get_hp_fit_trajectory(
    space: SearchSpace,
    hp_predictor_type: HPPredictorTypes,
    *,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
    override: bool = False,
) -> HPFitTrajectory:
    """Fit (or load from cache) one HP model per trial-prefix n=1..len(configs).

    Caching, when `cache_dir` is given, is keyed by `(hp_predictor_type,
    n_trials)` -- never by target_flops. Call `.predict(target_flops)` on the
    result to extrapolate cheaply to any target without refitting. Pass
    `override=True` to ignore any cached fit, refit from scratch, and
    overwrite the cache with the new result.
    """
    if len(configs) != len(losses) or len(configs) != len(flops):
        raise ValueError("configs, losses, and flops must have the same length")
    if len(configs) == 0:
        raise ValueError("At least one config/loss/flops row is required")

    n_trials = len(configs)

    if hp_predictor_type == HPPredictorTypes.GP_POSTERIOR:
        # Shares the same per-trial-prefix GP cache files as loss_predictor's
        # and scale_param_predictor's GP_POSTERIOR trajectories (same cache_dir,
        # same n) -- one GP fit serves loss, n_param, and HP predictions alike.
        fits: list[HPFit] = []
        for n in range(1, n_trials + 1):
            try:
                fit = _fit_hp(
                    hp_predictor_type,
                    space=space, configs=configs[:n], losses=losses[:n], flops=flops[:n],
                    cost_estimator=cost_estimator, cache_dir=cache_dir,
                    override=override,
                )
            except Exception:
                fit = HPFit(predictor_type=hp_predictor_type, space=space)
            fits.append(fit)
        return HPFitTrajectory(predictor_type=hp_predictor_type, fits=fits)

    if hp_predictor_type != HPPredictorTypes.LINEAR_REGRESSION:
        raise ValueError(f"Unknown predictor type: {hp_predictor_type}")

    hp_names = list(space.searchables.keys())
    n_hps = len(hp_names)
    predictor_slug = str(getattr(hp_predictor_type, "value", None) or hp_predictor_type)

    cache_file = None
    cached_params = None
    if cache_dir is not None:
        cache_file = predictor_cache.cache_path(cache_dir, predictor_slug, "hp")
        cached = None if override else predictor_cache.load_fit_array(cache_file, n_trials)
        if cached is not None and list(cached.get("hp_names", [])) == hp_names:
            cached_params = cached["params"]  # shape (n_trials, n_hps, 6)

    fits: list[HPFit] = []
    params_rows: list[list[list[float]]] = []
    for n in range(1, n_trials + 1):
        linear_fits: dict[str, LogLogPowerLawFit] | None = None
        try:
            pareto_ids, pareto_configs, pareto_flops, pareto_losses = _compute_hp_pareto(
                configs=configs[:n], losses=losses[:n], flops=flops[:n],
            )

            if cached_params is not None:
                row_n = np.asarray(cached_params[n - 1], dtype=float)  # shape (n_hps, 6)
                # Only slope/intercept (cols 0:2) are needed for the point
                # prediction. r_squared/stderr (cols 2:5) can legitimately be
                # NaN -- e.g. linregress divides by zero when an HP's value is
                # constant across the Pareto front -- without the fit itself
                # having failed, matching the original (fit_power_law raises
                # only when it can't produce slope/intercept at all).
                if np.any(np.isnan(row_n[:, :2])):
                    raise ValueError("cached fit is invalid (NaN slope/intercept)")
                linear_fits = {name: _row_to_loglog(row_n[j]) for j, name in enumerate(hp_names)}
            else:
                linear_fits = _fit_linear_regression_hp(
                    space=space, pareto_configs=pareto_configs, pareto_flops=pareto_flops,
                )

            fit = HPFit(
                predictor_type=hp_predictor_type,
                space=space,
                linear_fits=linear_fits,
                pareto_ids=pareto_ids,
                pareto_configs=pareto_configs,
                pareto_flops=pareto_flops,
                pareto_losses=pareto_losses,
            )
        except Exception:
            # Too few points to fit at this trial-prefix (e.g. n=1); leave a
            # failed-fit sentinel so `.predict()` yields an empty result, not
            # a hard error.
            fit = HPFit(predictor_type=hp_predictor_type, space=space)
            linear_fits = None

        fits.append(fit)
        if cached_params is None:
            if linear_fits is not None:
                params_rows.append([_loglog_to_row(linear_fits[name]) for name in hp_names])
            else:
                params_rows.append([[float("nan")] * 6 for _ in range(n_hps)])

    if cache_file is not None and cached_params is None:
        predictor_cache.save_fit_array(
            cache_file,
            n_trials,
            params=np.asarray(params_rows, dtype=float),
            hp_names=np.asarray(hp_names, dtype=object),
        )

    return HPFitTrajectory(predictor_type=hp_predictor_type, fits=fits)


def predict_hp_from_scaling_law(
    space: SearchSpace,
    hp_predictor_type: HPPredictorTypes,
    target_flops: float,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
    *,
    cost_estimator: Any = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    fit = _fit_hp(
        hp_predictor_type,
        space=space, configs=configs, losses=losses, flops=flops,
        cost_estimator=cost_estimator, cache_dir=cache_dir,
    )
    return fit.predict(target_flops)
