import json
import logging
import time
import random
from pathlib import Path
from typing import Any

import numpy as np

from neps.status.status import post_run_csv
from neps.state import NePSState
from neps.optimizers.scaling_law_guided import fit_pareto_front_loglog


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_neps_state(root_directory: Path | str) -> tuple[Any, Any]:
    """Load NePSState and trials from a directory."""
    root_directory = Path(root_directory)

    if not root_directory.exists():
        raise FileNotFoundError(
            f"No neps state found at: {root_directory}\n"
            "Please provide a valid neps results directory."
        )
    try:
        state = NePSState.create_or_load(path=root_directory, load_only=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load neps state from: {root_directory}"
        ) from e

    space = state.lock_and_get_search_space()
    trials = state._trial_repo.latest(refresh_cache=True)

    if space is None:
        raise RuntimeError("Could not load search space from neps state.")

    return state, trials


def _load_neps_state_with_retry(root_directory: Path | str, max_retries: int = 10) -> tuple[Any, Any]:
    """Load NePSState with retry logic to handle concurrent file access."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return _load_neps_state(root_directory)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Failed to load neps state (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
            continue

    raise RuntimeError(
        f"Failed to load neps state after {max_retries} attempts. Last error: {last_error}"
    )


def get_true_pareto_front(root_directory: Path | str) -> dict[str, list]:
    """Extract and return Pareto front data from a neps study."""
    _, trials = _load_neps_state_with_retry(root_directory)

    result = fit_pareto_front_loglog(trials, lc_aware=False)
    pareto_rows = result[5]
    pareto_trial_ids = result[7]
    pareto_configs = [trial.config for trial in trials.values() if trial.id in pareto_trial_ids]

    return {
        "loss": [r[2] for r in pareto_rows],
        "flops": [r[3] for r in pareto_rows],
        "n_params": [float(r[0]) for r in pareto_rows],
        "n_data": [float(r[1]) for r in pareto_rows],
        "configs": pareto_configs,
    }


def save_pareto_info(pareto_info: dict, output_dir: Path | str) -> None:
    """Save Pareto front information to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pareto_info.json"
    with open(output_file, 'w') as f:
        json.dump(pareto_info, f, indent=2)
    logger.info(f"Saved pareto_info to {output_file}")


def load_pareto_info(output_dir: Path | str) -> dict[str, list] | None:
    """Load Pareto front information from JSON file if it exists."""
    output_dir = Path(output_dir)
    output_file = output_dir / "pareto_info.json"

    if output_file.exists():
        with open(output_file, 'r') as f:
            logger.info(f"Loaded pareto_info from {output_file}")
            return json.load(f)
    return None


def load_or_generate_pareto_info(benchmark_name: str) -> dict[str, list]:
    from src.baselines.configs import OptimizerEnum
    from src.benchmarks.configs import get_results_dir
    comprehensive_study_root_dir = (
        # get_results_dir() / benchmark_name / f"{OptimizerEnum.GRID_SEARCH}" / "seed_40"
        Path("/work/dlc2workfs2/alipourn-scan/test") / benchmark_name / f"{OptimizerEnum.GRID_SEARCH}" / "seed_40"
    )
    summary_dir = Path(comprehensive_study_root_dir) / "summary"

    pareto_info = load_pareto_info(summary_dir)
    if pareto_info is None:
        pareto_info = get_true_pareto_front(comprehensive_study_root_dir)
        save_pareto_info(pareto_info, summary_dir)
    return pareto_info
