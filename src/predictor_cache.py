from pathlib import Path
import numpy as np


def cache_path(cache_dir: Path, predictor_slug: str, kind: str) -> Path:
    safe_slug = predictor_slug.replace(" ", "_").replace("/", "_")
    return cache_dir / f"fit_{kind}_{safe_slug}.npz"


def load_fit_array(path: Path, n_trials: int) -> dict[str, np.ndarray] | None:
    """Return the cached fit arrays if the cache is valid for n_trials, else None."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        if int(data["n_trials"]) != n_trials:
            return None
        return {k: data[k] for k in data.files if k != "n_trials"}
    except Exception:
        return None


def save_fit_array(path: Path, n_trials: int, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(path, n_trials=np.array(n_trials), **arrays)
    except Exception as e:
        print(f"Warning: could not save fit cache to {path}: {e}")
