from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.plotting.labels import (
    ABLATION_INIT_LABELS,
    ABLATION_LABELS,
    GENERAL_HPOSUITE,
    HP_LABELS,
    INTRO_LABELS,
    LABELS_1,
    SIG_LABELS,
    SO_LABELS,
)
from src.plotting.plot_styles import (
    COLORS_HPS,
    COLORS_MAIN,
    COLORS_SO,
    HPOSUITE_COLORS,
    MARKERS,
)

map_labels = {
    "1": LABELS_1,
    "intro": INTRO_LABELS,
    "ablation": ABLATION_LABELS,
    "ablation_init": ABLATION_INIT_LABELS,
    "sig": SIG_LABELS,
    "SO": SO_LABELS,
    "hposuite": GENERAL_HPOSUITE,
}

bench_alts = {
    "pd1-cifar100-wide_resnet-2048": "pd1-cifar100",
    "pd1-imagenet-resnet-512": "pd1-imagenet",
    "pd1-lm1b-transformer-2048": "pd1-lm1b",
    "pd1-translate_wmt-xformer_translate-64": "pd1-translate_wmt",
    "yahpo-lcbench-126026": "lcbench-126026",
    "yahpo-lcbench-146212": "lcbench-146212",
    "yahpo-lcbench-168330": "lcbench-168330",
    "yahpo-lcbench-168868": "lcbench-168868",

}


def dynamic_reference_point(
    loss_vals: pd.Series | list[Mapping[str, float]]
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if isinstance(loss_vals, pd.Series):
        loss_vals = loss_vals.to_numpy()
    loss_vals = np.array(
        [
            list(cost.values())
            for cost in loss_vals
        ]
    )
    worst_point = np.max(loss_vals, axis=0)
    reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
    reference_point[reference_point == 0] = 1e-12
    return reference_point


def pareto_front(
    costs: pd.Series | list[Mapping[str, float]] | np.array,
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if not isinstance(costs, np.ndarray):
        costs: np.array = np.array([list(cost.values()) for cost in costs])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto


def get_style(instance: str) -> tuple[str, str, str, str | None]:
    """Function to get the plotting style for a given instance."""
    prior_annot = None
    hps = None
    opt_splits = instance.split(";")
    if isinstance(opt_splits, str):
        opt_splits = [opt_splits]
    match len(opt_splits):
        case 1:
            pass
        case 2:
            if "priors=" in opt_splits[1]:
                prior_annot = opt_splits[1].split("priors=")[-1]
            else:
                hps = opt_splits[1]
        case 3:
            hps = opt_splits[1]
            prior_annot = opt_splits[2].split("priors=")[-1]
        case _:
            raise ValueError(
                "Multiple HPs not yet supported"
            )
    opt = opt_splits[0]
    color = (
        COLORS_HPS.get((opt, hps)) if hps 
        else COLORS_MAIN.get(opt, COLORS_SO.get(opt, HPOSUITE_COLORS.get(opt)))
    )
    marker = MARKERS.get(prior_annot, "s")
    if color is None:
        print(f"No color found for {opt}")
        breakpoint()  # noqa: T100
    return marker, color, opt, hps, prior_annot


def change_opt_names(optimizer: str) -> str:
    """Function to change the optimizer names."""
    if optimizer == "NepsMOASHAPiBORW":
        optimizer = "NepsPriMO"
    return optimizer


def edit_bench_labels(
    name: str,
) -> str:
    """Function to edit the benchmark names."""
    return bench_alts.get(name, name)


def edit_axis_labels(
    x_or_y: str,
) -> str:
    """Function to edit the axis labels."""
    return "Training Cost" if x_or_y == "y" else "Validation Error"


def edit_legend_labels(  # noqa: C901, PLR0912
    labels: list[str],
    which_labels: int | str = "1",
    prior_annotations: str | None = "default",
) -> list[str]:
    """Edit the legend labels to be more readable."""
    if isinstance(which_labels, int):
        which_labels = str(which_labels)
    new_labels = []
    for label in labels:
        prior_annot = ""
        hps = None
        _label = label
        if "_w_continuations" in _label:
            _label = _label.replace("_w_continuations", "")
        opt_splits = _label.split(";")
        if isinstance(opt_splits, str):
            opt_splits = [opt_splits]
        match len(opt_splits):
            case 1:
                pass
            case 2:
                if "priors=" in opt_splits[1]:
                    prior_annot = opt_splits[1].split("priors=")[-1]
                else:
                    hps = opt_splits[1]
            case 3:
                hps = opt_splits[1]
                prior_annot = opt_splits[2].split("priors=")[-1]
            case _:
                raise ValueError(
                    "Multiple HPs not yet supported"
                )
        _label = opt_splits[0]
        _label = map_labels[which_labels].get(
            _label, _label)
        if hps is not None:
            hps, value = hps.split("=")
            hps = HP_LABELS.get(
                hps, hps)
            hps = f"{hps}={value}"
            _label = f"{_label} ({hps})"
        elif prior_annot:
            prior_annot = prior_annotations if prior_annotations != "default" else prior_annot
            if prior_annot == "good-good":
                prior_annot = "all priors good"
            elif prior_annot == "bad-bad":
                prior_annot = "all priors bad"
            elif prior_annot == "mixed":
                prior_annot = "mixed priors"
            _label = _label if prior_annot is None else f"{_label} ({prior_annot})"
        new_labels.append(_label)
    return new_labels


def avg_seed_dfs_for_ranking(
    seed_means_dict: dict[str | int, dict[str, pd.Series]],
) -> dict[int, dict[str, pd.Series]]:
    """Function to average prior seeds and move them into their
    respective seed dicts.
    """
    final_dict = {}
    seed_dfs = {}
    for seed, instances in seed_means_dict.items():
        if isinstance(seed, str) and "_" in seed:
            assert len(instances) == 1
            _seed = int(seed.split("_")[0])
            if _seed not in seed_dfs:
                seed_dfs[_seed] = {}
            instance = next(iter(instances))
            if instance not in seed_dfs[_seed]:
                seed_dfs[_seed][instance] = []
            seed_dfs[_seed][instance].append(instances[instance])
        elif seed not in final_dict:
            final_dict[seed] = instances
    for seed, instances in seed_dfs.items():
        for instance, df_list in instances.items():
            if seed not in final_dict:
                final_dict[seed] = {}
            else:
                assert instance not in final_dict[seed]
            final_dict[seed][instance] = pd.DataFrame(df_list).mean()
    return final_dict


regret_bounds = {
    "mfh3_good": (-3.8, 1000.0),
    "mfh6_good": (-3.3, 1000.0),
    "pd1-cifar100-wide_resnet-2048": (0.0, 1.0),
    "pd1-imagenet-resnet-512": (0.0, 1.0),
    "pd1-lm1b-transformer-2048": (0.0, 1.0),
    "pd1-translate_wmt-xformer_translate-64": (0.0, 1.0),
    "yahpo-lcbench-126026": (0.0, 2.0),
    "yahpo-lcbench-167190": (0.0, 2.0),
    "yahpo-lcbench-168330": (0.0, 2.0),
    "yahpo-lcbench-189906": (0.0, 2.0),
    "jahs-CIFAR10": (-100.0, 0),
    "jahs-ColorectalHistology": (-100.0, 0.0),
    "jahs-FashionMNIST": (-100.0, 0.0),
}


reference_points_dict = {

    # HW-GPT-Bench
    "hwgptbench-s": {"perplexity": 100, "params": 130000000, "flops": 9e12},
    "hwgptbench-m": {"perplexity": 100, "params": 400000000, "flops": 9e12},
    "hwgptbench-l": {"perplexity": 100, "params": 800000000, "flops": 9e12},

    # MOMFPark
    "MOMFPark": {"value1": 1, "value2": 1},

    # MOMFBraninCurrin
    "MOMFBraninCurrin": {"value1": 1, "value2": 1},

    # PD1
    "pd1-cifar100-wide_resnet-2048": {"valid_error_rate": 1, "train_cost": 30},
    "pd1-imagenet-resnet-512": {"valid_error_rate": 1, "train_cost": 5000},
    "pd1-lm1b-transformer-2048": {"valid_error_rate": 1, "train_cost": 1000},
    "pd1-translate_wmt-xformer_translate-64": {"valid_error_rate": 1, "train_cost": 20000},

    # MF-ZDT Benchmarks
    "MFZDT1": {"f1": 2, "f2": 2},
    "MFZDT6": {"f1": 2, "f2": 2},

    # JAHSBench
    "jahs-CIFAR10": {"runtime": 200000, "valid_acc": 0},
    "jahs-ColorectalHistology": {"runtime": 200000, "valid_acc": 0},
    "jahs-FashionMNIST": {"runtime": 200000, "valid_acc": 0},

    # YAHPO-LCBench
    "yahpo-lcbench-126026": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 150
    },
    "yahpo-lcbench-167190": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-168330": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 5000
    },
    "yahpo-lcbench-168910": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189906": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 1000
    },
    "yahpo-lcbench-3945": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-7593": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-34539": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126025": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126029": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-146212": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 150
    },
    "yahpo-lcbench-167104": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167149": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167152": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167161": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167168": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167181": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167184": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167185": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167200": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167201": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168329": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168331": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168335": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168868": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-168908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189354": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189862": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189865": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189866": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189873": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189905": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189909": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
}

hv_low_cutoffs = {
    "MOMFPark": 1.2,
    "pd1-cifar100-wide_resnet-2048": 10,
    "pd1-imagenet-resnet-512": 2600,
    "pd1-lm1b-transformer-2048": 275,
    "pd1-translate_wmt-xformer_translate-64": 7200,
    "yahpo-lcbench-126026": 45,
    "yahpo-lcbench-146212": 15,
    "yahpo-lcbench-168330": 150,
    "yahpo-lcbench-168868": 45,
    "jahs-CIFAR10": 0,
    "jahs-ColorectalHistology": 0,
    "jahs-FashionMNIST": 0,
    "hwgptbench-s": 6.4e14,
    "hwgptbench-m": 6.6e14,
    "hwgptbench-l": 6.6e14,
}

fid_perc_momfbo = {
    "MOMFPark": 0.10,
    "pd1-cifar100-wide_resnet-2048": 0.10,
    "pd1-imagenet-resnet-512": 0.10,
    "pd1-lm1b-transformer-2048": 0.10,
    "pd1-translate_wmt-xformer_translate-64": 0.10,
    "yahpo-lcbench-126026": 0.10,
    "yahpo-lcbench-146212": 0.10,
    "yahpo-lcbench-168330": 0.10,
    "yahpo-lcbench-168868": 0.10,
    "jahs-CIFAR10": 0.9,
    "jahs-ColorectalHistology": 0.9,
    "jahs-FashionMNIST": 0.9,
    "yahpo-lcbench-167190": 0.9,
    "yahpo-lcbench-168910": 0.9,
    "yahpo-lcbench-189906": 0.9,
    "yahpo-lcbench-3945": 0.9,
    "yahpo-lcbench-7593": 0.9,
    "yahpo-lcbench-34539": 0.9,
    "yahpo-lcbench-126025": 0.9,
    "yahpo-lcbench-126029": 0.9,
    "yahpo-lcbench-167104": 0.9,
    "yahpo-lcbench-167149": 0.9,
    "yahpo-lcbench-167152": 0.9,
    "yahpo-lcbench-167161": 0.9,
    "yahpo-lcbench-167168": 0.9,
    "yahpo-lcbench-167181": 0.9,
    "yahpo-lcbench-167184": 0.9,
    "yahpo-lcbench-167185": 0.9,
    "yahpo-lcbench-167200": 0.9,
    "yahpo-lcbench-167201": 0.9,
    "yahpo-lcbench-168329": 0.9,
    "yahpo-lcbench-168331": 0.9,
    "yahpo-lcbench-168335": 0.9,
    "yahpo-lcbench-168908": 0.9,
    "yahpo-lcbench-189354": 0.9,
    "yahpo-lcbench-189862": 0.9,
    "yahpo-lcbench-189865": 0.9,
    "yahpo-lcbench-189866": 0.9,
    "yahpo-lcbench-189873": 0.9,
    "yahpo-lcbench-189905": 0.9,
    "yahpo-lcbench-189908": 0.9,
    "yahpo-lcbench-189909": 0.9,
}