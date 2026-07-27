from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np


@dataclass
class GPPosteriorPredictionInfo:
    num_trials_used: int
    target_flop: float
    predicted_loss: float
    constrained: bool  # True if a cost-feasible candidate was found


def gp_cache_path(cache_dir: Path, n_trials: int) -> Path:
    """One file per trial-prefix -- unlike the small closed-form fits, a GP
    fit is expensive enough that each n's model is worth caching independently
    (adding a trial shouldn't invalidate every earlier n's fit)."""
    return cache_dir / f"fit_gp_n{n_trials}.pt"


@dataclass
class GPPosteriorFit:
    """A target-flop-independent trained GP posterior over observed configs.

    Training (the expensive part) doesn't depend on target_flop at all. Call
    `.predict(target_flop)` to draw a fresh Sobol candidate pool, evaluate the
    already-trained posterior over it, and select the best cost-feasible
    candidate for that budget -- without retraining the GP.
    """

    space: Any
    encoder: Any
    gp: Any
    cost_estimator: Callable
    num_trials_used: int
    x_train: Any
    y_train: Any
    mean_slope: float
    mean_bias: float
    min_flops: int

    def predict(
        self,
        target_flop: float,
        margin: float = 0.1,
        num_candidates: int = 2048,
    ) -> tuple[dict[str, Any], float, GPPosteriorPredictionInfo]:
        """Draw Sobol candidates, evaluate the trained posterior over them, and
        pick the optimal config at or under target_flop.

        Candidates are drawn via Sobol sampling over the search space and filtered by
        estimated cost so the result never exceeds ``target_flop`` (matching the
        cost-constrained candidate construction used by ``IT-modified-JES`` rather than
        optimizing the GP posterior unconstrained and hoping a sample happens to land
        near ``target_flop``). The search first tries candidates within
        ``target_flop * [1 - margin, 1]`` and picks whichever minimizes predicted loss;
        if that band is empty it falls back to the closest-to-budget feasible candidate
        (``flops <= target_flop``); if no candidate is feasible at all, it falls back to
        the cheapest candidate found.
        """
        import random
        import torch
        from botorch.utils.sampling import draw_sobol_samples

        if target_flop <= 0:
            raise ValueError("target_flop must be positive")

        # Seed every RNG from target_flop itself, so predicting for the same
        # target_flop always draws the same Sobol candidate pool and yields a
        # reproducible acquisition -- rather than a fresh random draw each call.
        seed = int(target_flop) % (2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        lower = [domain.lower for domain in self.encoder.domains]
        upper = [domain.upper for domain in self.encoder.domains]
        bounds = torch.tensor([lower, upper], dtype=torch.float64)

        candidate_set = (
            draw_sobol_samples(bounds=bounds, n=num_candidates, q=1, seed=seed)
            .squeeze(1)
            .to(dtype=torch.float64)
        )
        decoded_candidates = self.encoder.decode(candidate_set)

        flops = torch.full((len(decoded_candidates),), float("inf"), dtype=torch.float64)
        for i, config in enumerate(decoded_candidates):
            try:
                flops[i] = self.cost_estimator(**config)
            except Exception:
                continue
        if torch.isinf(flops).all():
            raise ValueError("cost_estimator failed for every Sobol candidate")

        with torch.no_grad():
            predicted_loss = self.gp.posterior(candidate_set.unsqueeze(-2)).mean.squeeze(-1).squeeze(-1)

        lo = target_flop * (1.0 - margin)

        def _best_in(mask: "torch.Tensor") -> int | None:
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                return None
            best_local = int(torch.argmin(predicted_loss[idxs]).item())
            return int(idxs[best_local].item())

        # 1. Prefer candidates in the target band, closest to budget: [lo, target_flop].
        best_idx = _best_in((flops >= lo) & (flops <= target_flop))
        # 2. Fall back to any feasible candidate (flops <= target_flop).
        if best_idx is None:
            best_idx = _best_in(flops <= target_flop)
        constrained = best_idx is not None

        # 3. No feasible candidate at all -- use the cheapest one found.
        if best_idx is None:
            print(
                "WARNING: No Sobol candidate had estimated cost <= target_flop. "
                "Falling back to the cheapest candidate found."
            )
            best_idx = int(torch.argmin(flops).item())

        best_config = decoded_candidates[best_idx]
        best_output = float(predicted_loss[best_idx].item())

        info = GPPosteriorPredictionInfo(
            num_trials_used=self.num_trials_used,
            target_flop=target_flop,
            predicted_loss=best_output,
            constrained=constrained,
        )
        return best_config, best_output, info


def _train_gp(*, space: Any, configs: list[dict[str, Any]], losses: list[float], cost_estimator: Callable) -> GPPosteriorFit:
    """Train a GP on observed configs/losses. Independent of any target_flop."""
    import torch
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from neps.space import ConfigEncoder
    from neps.optimizers.models.gp import make_default_single_obj_gp
    from src.scaling_law_fit import fit_power_law_mean_params

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

    # Fit a power law to determine the GP mean prior, matching the scaling-law
    # kernel used by BayesianOptimization.sample_candidates.
    mean_params = fit_power_law_mean_params(valid_configs, valid_losses, cost_estimator)

    gp = make_default_single_obj_gp(
        x=x_train, y=y_train, encoder=encoder,
        flop_estimator=cost_estimator,
        min_flops=mean_params.min_flops,
        mean_slope=mean_params.slope,
        mean_bias=mean_params.intercept,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    return GPPosteriorFit(
        space=space,
        encoder=encoder,
        gp=gp,
        cost_estimator=cost_estimator,
        num_trials_used=len(valid),
        x_train=x_train,
        y_train=y_train,
        mean_slope=mean_params.slope,
        mean_bias=mean_params.intercept,
        min_flops=mean_params.min_flops,
    )


def _save_gp_fit(path: Path, n_trials: int, fit: GPPosteriorFit) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "n_trials": n_trials,
                "gp_state_dict": fit.gp.state_dict(),
                "x_train": fit.x_train,
                "y_train": fit.y_train,
                "mean_slope": fit.mean_slope,
                "mean_bias": fit.mean_bias,
                "min_flops": fit.min_flops,
            },
            path,
        )
    except Exception as e:
        print(f"Warning: could not save GP fit cache to {path}: {e}")


def _load_gp_fit(path: Path, n_trials: int, *, space: Any, cost_estimator: Callable) -> GPPosteriorFit | None:
    import torch
    from neps.space import ConfigEncoder
    from neps.optimizers.models.gp import make_default_single_obj_gp

    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, weights_only=False)
        if int(checkpoint["n_trials"]) != n_trials:
            return None

        encoder = ConfigEncoder.from_parameters(space.searchables)
        x_train = checkpoint["x_train"]
        y_train = checkpoint["y_train"]
        gp = make_default_single_obj_gp(
            x=x_train, y=y_train, encoder=encoder,
            flop_estimator=cost_estimator,
            min_flops=checkpoint["min_flops"],
            mean_slope=checkpoint["mean_slope"],
            mean_bias=checkpoint["mean_bias"],
        )
        gp.load_state_dict(checkpoint["gp_state_dict"])
        gp.eval()

        return GPPosteriorFit(
            space=space,
            encoder=encoder,
            gp=gp,
            cost_estimator=cost_estimator,
            num_trials_used=n_trials,
            x_train=x_train,
            y_train=y_train,
            mean_slope=checkpoint["mean_slope"],
            mean_bias=checkpoint["mean_bias"],
            min_flops=checkpoint["min_flops"],
        )
    except Exception:
        return None


def fit_gp_posterior(
    *,
    space: Any,
    configs: list[dict[str, Any]],
    losses: list[float],
    cost_estimator: Callable,
    cache_path: Path | None = None,
    override: bool = False,
) -> GPPosteriorFit:
    """Fit (or load from cache) a GP posterior over `configs`/`losses`.

    Caching, when `cache_path` is given, is keyed by `n_trials = len(configs)`
    only -- never by target_flop, since training doesn't depend on it. Call
    `.predict(target_flop)` on the result to acquire a point for any target
    without retraining. Pass `override=True` to ignore any cached fit and
    retrain from scratch, overwriting the cache file with the new result.
    """
    n_trials = len(configs)

    if cache_path is not None and not override:
        cached = _load_gp_fit(cache_path, n_trials, space=space, cost_estimator=cost_estimator)
        if cached is not None:
            return cached

    fit = _train_gp(space=space, configs=configs, losses=losses, cost_estimator=cost_estimator)

    if cache_path is not None:
        _save_gp_fit(cache_path, n_trials, fit)

    return fit
