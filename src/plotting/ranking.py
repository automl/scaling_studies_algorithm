from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pymoo.indicators.hv import Hypervolume

from baselines.configs import OPTIMIZERS
# from src.constants import DEFAULT_RESULTS_DIR
from src.plotting.plot_styles import (
    RC_PARAMS,
    XTICKS,
    other_fig_params,
)
from src.plotting.plot_utils import (
    edit_legend_labels,
    fid_perc_momfbo,
    get_style,
    pareto_front,
    reference_points_dict,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


sns.set_theme(style="whitegrid")
sns.set_context("paper")

SEED_COL = "seed"
OPTIMIZER_COL = "optimizer"
BENCHMARK_COL = "benchmark"
HP_COL = "optimizer_hyperparameters"
OBJECTIVES_COL = "objectives"
OBJECTIVES_MINIMIZE_COL = "minimize"
BUDGET_USED_COL = "budget_used_total"
BUDGET_TOTAL_COL = "budget_total"
FIDELITY_COL = "fidelity"
BENCHMARK_COUNT_FIDS = "benchmark.fidelity.count"
BENCH_FIDELITY_NAME = "benchmark.fidelity.1.name"
BENCH_FIDELITY_MIN_COL = "benchmark.fidelity.1.min"
BENCH_FIDELITY_MAX_COL = "benchmark.fidelity.1.max"
CONTINUATIONS_COL = "continuations_cost"
CONTINUATIONS_BUDGET_USED = "continuations_budget_used_total"


def plot_average_rank(
    ax: plt.Axes,
    ranks: dict,
    budget: int,
) -> None:
    """Plots the average rank of optimizers over iterations with standard error."""
    def _mean(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(_dfs).reset_index().groupby("index").mean()

    def _sem(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(_dfs).reset_index().groupby("index").sem()

    # Get mean across all benchmarks, for each seed
    ranks_per_seed_averaged_over_benchmarks = {
        seed: _mean(ranks[seed].values()) for seed in ranks
    }

    # Average over all seeds
    mean_ranks = _mean(ranks_per_seed_averaged_over_benchmarks.values())
    sem_ranks = _sem(ranks_per_seed_averaged_over_benchmarks.values())

    # Sort by the last iteration
    mean_ranks = mean_ranks.sort_values(by=budget, axis=1, ascending=False)
    sem_ranks = sem_ranks[mean_ranks.columns]

    # Plot settings
    plt.figure(figsize=(12, 6))

    # Plotting
    for instance in mean_ranks.columns:
        means = mean_ranks[instance]
        sems = sem_ranks[instance]

        marker, color, _, _ = get_style(instance)
        ax.plot(
            means,
            linestyle="-",
            label=instance,
            color=color,
        )
        ax.fill_between(
            means.index,
            means - sems,
            means + sems,
            alpha=0.1,
            color=color,
            edgecolor=None,
        )


def create_plots(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    ax_hv: plt.Axes,
    ax_pareto: plt.Axes,
    agg_data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
    objectives: list[str],
    seed_for_pareto: int,
    budget: int,
    fidelity: str | None = None,
    plot_opt: str | None = None,
    is_single_opt: bool = False,
    no_save: bool = False,
    save_individual: bool = False,
) -> dict[str, pd.DataFrame]:
    """Function to plot the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Plots for runs on benchmark: {benchmark}")

    seed_means_dict = {}
    rank_means_dict = {}

    reference_point = np.array([
        reference_points_dict[benchmark][obj]
        for obj in objectives
    ])
    plt.figure(5, figsize=(10, 10))
    if plot_opt:
        plt.figure(6, figsize=(10, 10))
    for instance, instance_data in agg_data.items():

        # Get the marker, color, optimizer name and prior annotations for the optimizer instance
        marker, color, opt, prior_annot = get_style(instance)

        assert opt in OPTIMIZERS

        continuations = False
        agg_pareto_costs = []
        agg_pareto_front = []
        logger.info(f"Plots for instance: {instance}")
        seed_hv_dict = {}
        seed_cont_dict = {}
        for seed, data in instance_data.items():
            results: list[dict[str, Any]] = data["results"]
            _df = data["_df"]
            keys = list(results[0].keys())
            acc_costs = []
            pareto = None
            hv_vals = []
            budget_type = "TrialBudget" if fidelity is None else "FidelityBudget"
            match budget_type:
                case "FidelityBudget":
                    assert FIDELITY_COL in _df.columns
                    if _df[FIDELITY_COL].iloc[0] is None:
                        budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)
                case "TrialBudget":
                    budget_list = _df[BUDGET_USED_COL].values.astype(np.float64)
                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")


            if "single" in OPTIMIZERS[opt].support.fidelities:
                continuations = True
                continuations_list = _df[CONTINUATIONS_COL].values.astype(np.float64)
                continuations_list = np.cumsum(continuations_list)

            num_full_evals = 0
            for i, costs in enumerate(results, start=1):
                if budget_type == "FidelityBudget" and _df[FIDELITY_COL][0] is not None:
                    fidelity_queried = _df[FIDELITY_COL].iloc[i-1]
                    bench_max_fid = _df[BENCH_FIDELITY_MAX_COL].iloc[0]
                    continuations_budget_used = _df[CONTINUATIONS_BUDGET_USED].iloc[i-1]
                    max_fid_flag = True
                    match instance:
                        case "MOMFBO":
                            max_fid_flag = (
                                float(fidelity_queried) / float(bench_max_fid)
                                > float(fid_perc_momfbo[benchmark])
                            )
                        case str():
                            max_fid_flag = float(fidelity_queried) == float(bench_max_fid)
                        case _:
                            print("Huh?")

                    if not max_fid_flag:
                        if int(continuations_budget_used) > int(num_full_evals):
                            num_full_evals += 1
                            if len(hv_vals) > 0:
                                hv_vals.append(hv_vals[-1])
                            else:
                                hv_vals.append(np.nan)
                        continue
                # Compute hypervolume
                num_full_evals += 1
                acc_costs.append(costs)
                pareto = pareto_front(acc_costs)
                pareto = np.array([list(ac.values()) for ac in acc_costs])[pareto]
                pareto = pareto[pareto[:, 0].argsort()]
                agg_pareto_costs.extend(pareto)
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)

                if num_full_evals == budget:
                    break

            budget_list = np.arange(1, num_full_evals + 1, 1)

            if continuations:
                seed_cont_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_incumbents = seed_cont_dict[seed].cummax()
                instance_name = f"{instance}_w_continuations"
            else:
                seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)
                seed_incumbents = seed_hv_dict[seed].cummax()
                instance_name = instance

            seed_incumbents[budget] = seed_incumbents.iloc[-1]

            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance_name] = seed_incumbents

            plot_dir = exp_dir / "plots"
            pareto_save_dir = plot_dir / "pareto"

            match plot_opt:
                case None:
                    pass
                case "all":
                # Plotting Pareto fronts for all seeds
                    plt.figure(6)
                    plt.step(
                        pareto[:, 0],
                        pareto[:, 1],
                        where="post",
                        marker=marker,
                        label=f"{instance}_{seed}",
                        markersize=7,
                        linewidth=1,
                    )
                    plt.xlabel(keys[0])
                    plt.ylabel(keys[1])
                    plt.grid(visible=True)
                    plt.title(f"Multiple seeds pareto plot for \n{instance} on {benchmark}")
                case str():
                    if plot_opt in instance:
                        plt.figure(6)
                        plt.step(
                            pareto[:, 0],
                            pareto[:, 1],
                            where="post",
                            marker=marker,
                            label=f"{instance}_{seed}",
                            markersize=7,
                            linewidth=1,
                        )
                        plt.xlabel(keys[0])
                        plt.ylabel(keys[1])
                        plt.grid(visible=True)
                        plt.title(f"Multiple seeds pareto plot for \n{instance} on {benchmark}")
                case _:
                    raise ValueError(f"Invalid plot_opt: {plot_opt}.")

            # Plotting pareto front for a given seed
            if seed == seed_for_pareto:
                plt.figure(5)
                plt.step(
                    pareto[:, 0],
                    pareto[:, 1],
                    where="post",
                    marker=marker,
                    color=color if not is_single_opt else None,
                    label=f"{instance}_{seed}",
                    markersize=7,
                    linewidth=1,
                )

        if plot_opt and (plot_opt == "all" or plot_opt in instance):
            pareto_save_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(6)
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.05),
                fontsize=12
            )
            plt.tight_layout()
            if not no_save and save_individual:
                plt.savefig(
                    pareto_save_dir / "multiple" /
                    f"Multiple seeds pareto plot for {instance} on {benchmark}.png"
                )
            plt.clf()

        plt.figure(5)

        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            fontsize=12
        )
        plt.tight_layout()
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.grid(visible=True)
        plt.title(benchmark)

        # Aggregating Hypervolumes - calculating means, cumulative max and std_error

        _color = None

        if not continuations:
            seed_hv_df = pd.DataFrame(seed_hv_dict)
            seed_hv_df = seed_hv_df.ffill(axis=0)
            means = pd.Series(seed_hv_df.mean(axis=1), name=f"means_{instance}")
            sem = pd.Series(seed_hv_df.sem(axis=1), name=f"ste_{instance}")
            means = means.cummax()
            means = means.drop_duplicates()
            sem = sem.loc[means.index]
            means[budget] = means.iloc[-1]
            sem[budget] = sem.iloc[-1]

            # Plotting Hypervolumes

            ax_hv.plot(
                means,
                label=instance,
                color=color if not is_single_opt else None,
            )
            ax_hv.fill_between(
                means.index,
                means - sem,
                means + sem,
                alpha=0.1,
                color=color if not is_single_opt else None,
                edgecolor=None,
            )

        # For plotting continuations
        else:

            _, _color, _, _ = get_style(f"{instance}_w_continuations")
            seed_cont_df = pd.DataFrame(seed_cont_dict)
            seed_cont_df = seed_cont_df.ffill(axis=0)
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance}")
            sem_cont = pd.Series(seed_cont_df.sem(axis=1), name=f"sem_{instance}")
            means_cont = means_cont.cummax()
            means_cont = means_cont.drop_duplicates()
            sem_cont = sem_cont.loc[means_cont.index]
            means_cont[budget] = means_cont.iloc[-1]
            sem_cont[budget] = sem_cont.iloc[-1]

            ax_hv.plot(
                means_cont,
                label=f"{instance}_w_continuations",
                color=_color if not is_single_opt else None,
            )
            ax_hv.fill_between(
                means_cont.index,
                means_cont - sem_cont,
                means_cont + sem_cont,
                alpha=0.1,
                color=_color if not is_single_opt else None,
                edgecolor=None,
            )


        # Aggregated Pareto front over all seeds

        agg_pareto_costs = np.array(agg_pareto_costs)
        agg_pareto_front = pareto_front(agg_pareto_costs)
        agg_pareto_front = np.array(agg_pareto_costs)[agg_pareto_front]
        agg_pareto_front = agg_pareto_front[agg_pareto_front[:, 0].argsort()]
        ax_pareto.step(
            agg_pareto_front[:, 0],
            agg_pareto_front[:, 1],
            where="post",
            marker=marker,
            color=color if not continuations else _color,
            label=instance if not continuations else f"{instance}_w_continuations",
        )

    if not no_save and save_individual:
        plt.figure(5)
        pareto_save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pareto_save_dir / "seeds" / f"Pareto_plot_{benchmark}_single_seed.png")

    plt.close("all")

    # Ranking every optimizer instance per seed per benchmark
    for seed, instances in seed_means_dict.items():
        _rankdf = pd.DataFrame(instances)
        _rankdf = _rankdf.ffill(axis=0)
        _rankdf = _rankdf.rank(axis=1, method="average", ascending=False)
        rank_means_dict[seed] = _rankdf
    instances = list(agg_data.keys())
    return rank_means_dict


def agg_data(
    exp_dir: Path,
    skip_opt: list[str] | None = None,
) -> tuple[
    Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]],
    int,
    int
]:
    """Function to aggregate data from all runs in the experiment directory
    into a dictionary mapping benchmark names to a dictionary of run configs and dataframes.
    """
    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    benchmarks_in_dir.sort()
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    # Infer total_budget and seed_for_pareto from result files instead of exp.yaml
    total_budget = 0
    seed_for_pareto = None
    benchmarks_dict: Mapping[str, Mapping[tuple[str, str], list[pd.DataFrame]]] = {}

    for benchmark in benchmarks_in_dir:
        objectives = []
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            opt_name = file.name.split("optimizer=")[-1].split(".")[0]
            if skip_opt and opt_name in skip_opt:
                continue
            _df = pd.read_parquet(file)

            # Infer total_budget from the max budget used across all results
            if BUDGET_USED_COL in _df.columns:
                max_budget_in_df = _df[BUDGET_USED_COL].max()
                total_budget = max(total_budget, max_budget_in_df)

            # Set seed_for_pareto from first seed encountered
            if seed_for_pareto is None and SEED_COL in _df.columns:
                seed_for_pareto = int(_df[SEED_COL].iloc[0])

            with (file.parent / "run_config.yaml").open("r") as f:
                run_config = yaml.safe_load(f)
            objectives = run_config["problem"]["objectives"]
            fidelities = run_config["problem"]["fidelities"]
            if fidelities and not isinstance(fidelities, str) and len(fidelities) > 1:
                raise NotImplementedError("Plotting not yet implemented for many-fidelity runs.")

            all_plots_dict = benchmarks_dict.setdefault(benchmark, {})
            conf_tuple = (tuple(objectives), fidelities)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)

    return benchmarks_dict, seed_for_pareto, int(total_budget)


def gen_plots_per_bench(  # noqa: C901, PLR0913
    ax_hv: plt.Axes,
    ax_pareto: plt.Axes,
    exp_dir: Path,
    seed_for_pareto: int,
    total_budget: int,
    *,
    save_individual: bool = False,
    benchmark: str,
    conf_tuple: tuple[str, str],
    _all_dfs: list[pd.DataFrame],
    plot_opt: str | None = None,
    no_save: bool = False,
    priors_to_avg: list[str] | None = None,
) -> dict[int, pd.DataFrame]:
    """Function to generate plots for a given benchmark and its config dict."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    df_agg = {}
    objectives = conf_tuple[0]
    fidelity = conf_tuple[1]
    for _df in _all_dfs:
        if _df.empty:
            continue
        instance = _df[OPTIMIZER_COL].iloc[0]
        if _df[HP_COL].iloc[0] is not None:
            instance = f"{instance}_{_df[HP_COL].iloc[0]}"
        seed = _df[SEED_COL].iloc[0]
        if instance not in df_agg:
            df_agg[instance] = {}
        if int(seed) not in df_agg[instance]:
            df_agg[instance][int(seed)] = {"results": _df}
        assert objectives is not None


        assert len(objectives) == 2, ( # noqa: PLR2004
            "More than 2 objectives found in results file: "
            f"{objectives}. "
            "Can only plot pareto front for 2D cost space."
        )

        minimize: dict[str, bool] = _df["minimize"][0]

        _results = _df["results"].apply(
            lambda x, objectives=objectives, minimize=minimize: {
                k: x[k] if minimize[k] else -x[k] for k in objectives
            }
        )

        annotations = None

        if _df["prior_annotations"][0] is not None:
            annotations = "-".join(
                [a.split("=")[-1] for a in _df["prior_annotations"][0].split(",")]
            )

        instance = (
            _df["optimizer"][0] +
            (
                ";" + _df[HP_COL][0]
                if "default" not in _df[HP_COL][0]
                else ""
            ) +
            (f";priors={annotations}" if annotations else "")
        )

        seed = int(_df[SEED_COL][0])

        if priors_to_avg and annotations in priors_to_avg:
            instance = (
                _df["optimizer"][0] +
                (
                    ";" + _df[HP_COL][0]
                    if "default" not in _df[HP_COL][0]
                    else ""
                ) + ";priors=all"
            )
            if instance not in agg_dict:
                agg_dict[instance] = {}
            seed = f"{seed}_{_df['optimizer'][0]}_{annotations}"
            agg_dict[instance][seed] = {
                "_df": _df,
                "results": _results,
            }
            continue
        if instance not in agg_dict:
            agg_dict[instance] = {}
        agg_dict[instance][seed] = {
            "_df": _df,
            "results": _results,
        }
    is_single_opt = False
    if len(agg_dict) == 1:
        is_single_opt = True
    assert len(objectives) > 0, "Objectives not found in results file."


    return create_plots(
        ax_hv=ax_hv,
        ax_pareto=ax_pareto,
        agg_data=agg_dict,
        exp_dir=exp_dir,
        benchmark=benchmark,
        budget=total_budget,
        fidelity=fidelity,
        is_single_opt=is_single_opt,
        plot_opt=plot_opt,
        seed_for_pareto=seed_for_pareto,
        objectives=objectives,
        no_save=no_save,
        save_individual=save_individual,
    )



def make_subplots(  # noqa: C901, PLR0912, PLR0915
    exp_dir: Path,
    plot_opt: str | None = None,
    *,
    cut_off_iteration: int | None = None,
    no_save: bool = False,
    save_individual: bool = False,
    save_suffix: str = "",
    skip_opt: list[str] | None = None,
    priors_to_avg: list[str] | None = None,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    fig_size = other_fig_params["fig_size"]
    benchmarks_dict, seed_for_pareto, total_budget = agg_data(exp_dir, skip_opt=skip_opt)
    if cut_off_iteration:
        total_budget = cut_off_iteration
    num_plots = len(benchmarks_dict)
    nrows, ncols = other_fig_params["n_rows_cols"][num_plots]

    # Hypervolume plot
    fig_hv, axs_hv = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Pareto plot
    fig_pareto, axs_pareto = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Relative Ranking per benchmark plot
    fig_rank, axs_rank = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize=fig_size,
    )

    # Overall Relative Ranking plot
    fig_ov_rank, axs_ov_rank = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(7, 7),
    )

    plt.rcParams.update(RC_PARAMS)

    axs_hv: list[plt.Axes] = axs_hv.flatten() if num_plots > 1 else [axs_hv]
    axs_pareto: list[plt.Axes] = axs_pareto.flatten() if num_plots > 1 else [axs_pareto]
    axs_rank: list[plt.Axes] = axs_rank.flatten() if num_plots > 1 else [axs_rank]
    axs_ov_rank: list[plt.Axes] = [axs_ov_rank]

    xylabel_fontsize = other_fig_params["xylabel_fontsize"]
    xlabel_i = other_fig_params["xlabel_start_i"][num_plots]
    ylabel_i_inc = other_fig_params["ylabel_i_inc"][num_plots]
    ylabel_i_counter = 0

    means_dict = {}
    bench_dict = {}
    for i, (benchmark, conf_dict) in enumerate(benchmarks_dict.items()):
        assert len(conf_dict) == 1, (
            "Plotting more than 1 config for a benchmark not yet implemented."
            f" Found {len(conf_dict)} configs for benchmark {benchmark}."
        )
        for conf_tuple, _all_dfs in conf_dict.items():
            seed_dict_per_bench = gen_plots_per_bench(
                ax_hv=axs_hv[i],
                ax_pareto=axs_pareto[i],
                exp_dir=exp_dir,
                benchmark=benchmark,
                conf_tuple=conf_tuple,
                _all_dfs=_all_dfs,
                plot_opt=plot_opt,
                no_save=no_save,
                seed_for_pareto=seed_for_pareto,
                save_individual=save_individual,
                total_budget=total_budget,
                priors_to_avg=priors_to_avg,
            )

            axs_hv[i].set_xticks(XTICKS[(1, total_budget)])
            axs_hv[i].grid(visible=True)
            axs_hv[i].set_title(benchmark)
            axs_hv[i].set_xlim(1, total_budget)


            axs_pareto[i].set_xlabel(conf_tuple[0][0], fontsize=xylabel_fontsize)
            axs_pareto[i].set_ylabel(conf_tuple[0][1], fontsize=xylabel_fontsize)
            axs_pareto[i].grid(visible=True)
            axs_pareto[i].set_title(benchmark)

            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
                if _seed not in bench_dict:
                    bench_dict[_seed] = {}
                bench_dict[_seed][benchmark] = rank_df

            # Plotting average ranks per benchmark
            plot_average_rank(
                ax=axs_rank[i],
                ranks=bench_dict,
                budget=total_budget,
            )
            axs_rank[i].set_xticks(XTICKS[(1, total_budget)])
            axs_rank[i].set_title(benchmark)
            axs_rank[i].grid(visible=True)
            axs_rank[i].set_xlim(1, total_budget)

            if i >= xlabel_i:
                axs_hv[i].set_xlabel("Full Evaluations", fontsize=xylabel_fontsize)
                axs_rank[i].set_xlabel("Full Evaluations", fontsize=xylabel_fontsize)

            if i == ylabel_i_counter:
                axs_hv[i].set_ylabel("Hypervolume", fontsize=xylabel_fontsize)
                axs_rank[i].set_ylabel("Relative Rank", fontsize=xylabel_fontsize)
                ylabel_i_counter += ylabel_i_inc


            bench_dict = {}

    # Plotting average ranks over all benchmarks and seeds
    plot_average_rank(
        ax=axs_ov_rank[0],
        ranks=means_dict,
        budget=total_budget,
    )
    axs_ov_rank[0].set_xlabel("Full Evaluations")
    axs_ov_rank[0].set_xticks(XTICKS[(1, total_budget)])
    axs_ov_rank[0].set_ylabel("Relative Rank")
    axs_ov_rank[0].grid(visible=True)
    axs_ov_rank[0].set_xlim(1, total_budget)

    hv_handles, hv_labels = axs_hv[0].get_legend_handles_labels()
    pareto_handles, pareto_labels = axs_pareto[0].get_legend_handles_labels()
    rank_handles, rank_labels = axs_rank[0].get_legend_handles_labels()
    ov_rank_handles, ov_rank_labels = axs_ov_rank[0].get_legend_handles_labels()

    hv_labels = edit_legend_labels(hv_labels)
    pareto_labels = edit_legend_labels(pareto_labels)
    rank_labels = edit_legend_labels(rank_labels)
    ov_rank_labels = edit_legend_labels(ov_rank_labels)

    num_opts = len(hv_labels)

    bbox_to_anchor = other_fig_params["bbox_to_anchor"]
    tight_layout_pads = other_fig_params["tight_layout_pads"]
    legend_fontsize = other_fig_params["legend_fontsize"]

    # Remove empty subplots for hypervolume plot
    for j in range(i + 1, len(axs_hv)):
        fig_hv.delaxes(axs_hv[j])

    # Legend for hypervolume plot
    hv_leg = fig_hv.legend(
        hv_handles,
        hv_labels,
        fontsize=legend_fontsize,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=num_opts // 2 if num_opts > 2 else num_opts,  # noqa: PLR2004
        frameon=True,
        markerscale=2,
    )
    for item in hv_leg.legend_handles:
        item.set_linewidth(2)

    fig_hv.tight_layout(**tight_layout_pads)

    # Remove empty subplots for pareto plot
    for j in range(i + 1, len(axs_pareto)):
        fig_pareto.delaxes(axs_pareto[j])

    # Legend for pareto plot
    pareto_leg = fig_pareto.legend(
        pareto_handles,
        pareto_labels,
        fontsize=legend_fontsize,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=num_opts // 2 if num_opts > 2 else num_opts,  # noqa: PLR2004
        frameon=True,
        markerscale=2,
    )
    for item in pareto_leg.legend_handles:
        item.set_linewidth(2)

    fig_pareto.tight_layout(**tight_layout_pads)

    # Remove empty subplots for rank plot
    for j in range(i + 1, len(axs_rank)):
        fig_rank.delaxes(axs_rank[j])

    # Legend for rank plot
    rank_leg = fig_rank.legend(
        rank_handles,
        rank_labels,
        fontsize=legend_fontsize,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=num_opts // 2 if num_opts > 2 else num_opts,  # noqa: PLR2004
        frameon=True,
        markerscale=2,
    )
    for item in rank_leg.legend_handles:
        item.set_linewidth(2)

    fig_rank.tight_layout(**tight_layout_pads)

    # Legend for overall rank plot
    ov_rank_leg = fig_ov_rank.legend(
        ov_rank_handles,
        ov_rank_labels,
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=num_opts // 2 if num_opts > 2 else num_opts,  # noqa: PLR2004
        frameon=True,
        markerscale=2,
    )
    for item in ov_rank_leg.legend_handles:
        item.set_linewidth(2)

    fig_ov_rank.tight_layout(**tight_layout_pads)

    save_dir = exp_dir/ "plots" / "subplots"
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_suffix:
        save_suffix = f"_{save_suffix}"
    if not no_save:
        fig_hv.savefig(
            save_dir / f"hypervolume_subplots{save_suffix}.png", dpi=300, bbox_inches="tight"
        )
        fig_pareto.savefig(
            save_dir / f"pareto_subplots{save_suffix}.png", dpi=300, bbox_inches="tight"
        )
        fig_rank.savefig(
            save_dir / f"rank_subplots{save_suffix}.png", dpi=300, bbox_inches="tight"
        )
        fig_ov_rank.savefig(
            save_dir / f"rank_overall_subplots{save_suffix}.png", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        required=True,
        help="Main experiment directory containing the runs to plot."
    )
    parser.add_argument(
        "--make_subplots", "-s",
        action="store_true",
        help="make subplot for all plots in the same experiment directory."
    )
    parser.add_argument(
        "--plot_opt", "-o",
        type=str,
        default=None,
        help="Plot all pareto fronts only for the given optimizer."
    )
    parser.add_argument(
        "--cut_off_iteration", "-i",
        type=int,
        default=None,
        help="Plot pareto fronts and Hypervolumes for the given iterations."
    )
    parser.add_argument(
        "--no_save", "-ns",
        action="store_true",
        help="Do not save the plots."
    )
    parser.add_argument(
        "--save_individual", "-sv_ind",
        action="store_true",
        help="Save individual plots for each optimizer instance."
    )
    parser.add_argument(
        "--save_suffix", "-suffix",
        type=str,
        default="",
        help="Suffix to add to the saved plots."
    )
    parser.add_argument(
        "--skip_opt", "-skip",
        nargs="+",
        type=str,
        default=None,
        help="Skip the given optimizers."
    )
    parser.add_argument(
        "--priors_to_avg", "-avg",
        nargs="+",
        type=str,
        default=None,
        help="Priors to average over."
    )
    args = parser.parse_args()
    exp_dir: Path = Path("results") / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}

    make_subplots(
        exp_dir=exp_dir,
        plot_opt=args.plot_opt,
        cut_off_iteration=args.cut_off_iteration,
        no_save=args.no_save,
        save_individual=args.save_individual,
        save_suffix=args.save_suffix,
        skip_opt=args.skip_opt,
        priors_to_avg=args.priors_to_avg,
    )