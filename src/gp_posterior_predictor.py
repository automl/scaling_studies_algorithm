from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

@dataclass
class GPPosteriorPredictionInfo:
    num_trials_used: int
    target_flop: float
    predicted_loss: float
    constrained: bool  # True if a cost-feasible candidate was found


def predict_from_gp_posterior(
    *,
    space: Any,
    configs: list[dict[str, Any]],
    losses: list[float],
    cost_estimator: Callable,
    target_flop: float,
    margin: float = 0.1,
    num_optima: int = 100,
) -> tuple[dict[str, Any], float, GPPosteriorPredictionInfo]:
    """Fit a GP to configs/losses and find the optimal config near target_flop.

    The search is constrained to candidates whose estimated cost lies within
    ``target_flop * [1 - margin, 1 + margin]`` (default ±10 %, matching the
    ``around_max_flops=True`` behaviour of BO_Guided_Scaling).  If no candidate
    satisfies the constraint the unconstrained minimum is returned instead.

    Args:
        space: neps SearchSpace with a ``searchables`` attribute.
        configs: List of evaluated hyperparameter dicts.
        losses: Corresponding objective values (lower is better).
        cost_estimator: Callable(**config) -> float returning FLOPs.
        target_flop: Target compute budget.
        margin: Fractional tolerance around target_flop (default 0.1 = ±10 %).
        num_optima: Number of Thompson-sampling candidates (default 100).

    Returns:
        (best_config, predicted_loss, info)
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition.utils import get_optimal_samples
    from neps.space import ConfigEncoder

    if target_flop <= 0:
        raise ValueError("target_flop must be positive")
    if cost_estimator is None:
        raise ValueError("cost_estimator is required for GP posterior prediction")

    parameters = space.searchables
    encoder = ConfigEncoder.from_parameters(parameters)

    valid = [(c, float(l)) for c, l in zip(configs, losses) if np.isfinite(float(l))]
    if len(valid) < 2:
        raise ValueError(f"Need at least 2 finite loss values, got {len(valid)}")

    valid_configs, valid_losses = zip(*valid)

    x_train = encoder.encode(list(valid_configs))
    y_train = torch.tensor(valid_losses, dtype=torch.float64).unsqueeze(-1)

    gp = SingleTaskGP(x_train, y_train, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    lower = [domain.lower for domain in encoder.domains]
    upper = [domain.upper for domain in encoder.domains]
    bounds = torch.tensor([lower, upper], dtype=torch.float64)

    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=gp, bounds=bounds, num_optima=num_optima, maximize=False
    )

    decoded_configs = encoder.decode(optimal_inputs)

    lo = target_flop * (1.0 - margin)
    hi = target_flop * (1.0 + margin)

    best_config: dict[str, Any] | None = None
    best_output = float("inf")
    constrained = False

    for i, config in enumerate(decoded_configs):
        try:
            flops = cost_estimator(**config)
        except Exception:
            continue
        if lo <= flops <= hi:
            val = float(optimal_outputs[i].item())
            if val < best_output:
                best_output = val
                best_config = config
                constrained = True

    if best_config is None:
        print(
            "WARNING: No GP posterior sample satisfied the cost constraint. "
            "Using unconstrained best."
        )
        best_idx = int(torch.argmin(optimal_outputs).item())
        best_config = decoded_configs[best_idx]
        best_output = float(optimal_outputs[best_idx].item())

    info = GPPosteriorPredictionInfo(
        num_trials_used=len(valid),
        target_flop=target_flop,
        predicted_loss=best_output,
        constrained=constrained,
    )
    return best_config, best_output, info
