from enum import Enum
import numpy as np
from typing import Any

from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer
from neps.space.search_space import SearchSpace, HPOInteger, HPOFloat
from src.scaling_law_fit import fit_power_law, predict_from_loglog_fit

class HPPredictorTypes(Enum):
    LINEAR_REGRESSION = "Linear Regression"
    GP_POSTERIOR = "GP Posterior"


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


def predict_hp_from_scaling_law(
    space: SearchSpace,
    hp_predictor_type: HPPredictorTypes,
    target_flops: float,
    configs: list[dict[str, Any]],
    losses: list[float],
    flops: list[float],
    *,
    cost_estimator: Any = None,
) -> dict[str, Any]:
    if len(configs) != len(losses) or len(configs) != len(flops):
        raise ValueError("configs, losses, and flops must have the same length")
    if len(configs) == 0:
        raise ValueError("At least one config/loss/flops row is required")

    pareto_result = ScalingLawGuidedOptimizer.compute_search_pareto_front(
        losses=[float(v) for v in losses],
        flops_list=[float(v) for v in flops],
    )
    if pareto_result is None:
        raise ValueError("Could not compute Pareto front from losses/flops")

    pareto_flops, pareto_losses, pareto_indices = pareto_result
    pareto_ids = _build_pareto_indices(
        losses=[float(v) for v in losses],
        flops=[float(v) for v in flops],
        pareto_losses=pareto_losses,
        pareto_flops=pareto_flops,
    )
    pareto_configs = [configs[i] for i in pareto_ids]

    extrapolated_config: dict[str, Any] = {}
    parameters = space.searchables
            
    if hp_predictor_type == HPPredictorTypes.LINEAR_REGRESSION:
        x_flops = np.asarray(pareto_flops, dtype=float)
        for param_name, param in parameters.items():
            hp_values_param: list[float] = []
            for config in pareto_configs:
                if param_name not in config:
                    raise ValueError(f"HP '{param_name}' is missing from config: {config}")
                hp_values_param.append(float(config[param_name]))

            if (
                isinstance(param, HPOInteger) and param.is_scaling
                ) or (
                isinstance(param, HPOInteger | HPOFloat) and param.log
            ):
                log = True
            else:
                log = False

            fit_result = fit_power_law(
                x_flops,
                np.asarray(hp_values_param, dtype=float),
                log_scale=log,
            )
            if fit_result is None:
                raise ValueError(f"Not enough valid data points to fit scaling law for HP '{param_name}'")
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

    if hp_predictor_type == HPPredictorTypes.GP_POSTERIOR:
        if cost_estimator is None:
            raise ValueError("cost_estimator is required for GP_POSTERIOR predictor")
        from src.gp_posterior_predictor import predict_from_gp_posterior
        best_config, predicted_loss, info = predict_from_gp_posterior(
            space=space,
            configs=configs,
            losses=losses,
            cost_estimator=cost_estimator,
            target_flop=target_flops,
        )
        return {
            "extrapolated_config": best_config,
            "predicted_loss": predicted_loss,
            "gp_posterior_info": info,
            "pareto_ids": [],
            "pareto_configs": [],
            "pareto_flops": [],
            "pareto_losses": [],
        }

    raise ValueError(f"Unsupported HP predictor type: {hp_predictor_type}")
