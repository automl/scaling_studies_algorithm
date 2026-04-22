import math
import sys

import numpy as np
from loguru import logger
from pathlib import Path
import torch
import neps
import copy

from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom, CustomOptimizer
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer, fit_pareto_front_loglog
from neps.state import Trial
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
import neps

from src.baselines.chinchilla_guided_scaling import Chinchilla_Guided_Scaling

import numpy as np
import copy
from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom
from neps.state import Trial
from neps.space.domain import Domain as NepsDomain
from neps.optimizers.optimizer import Artifact, ArtifactType

from typing import TYPE_CHECKING, Mapping, Any, Callable
from matplotlib.figure import Figure
from collections import OrderedDict
from neps.state.optimizer import BudgetInfo
from carbs import CARBS, CARBSParams, LogSpace, Param, LinearSpace, LogitSpace, ObservationInParam, Param

logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{message}")

import numpy as np
import copy
from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom

    
import logging

from neps.state.optimizer import BudgetInfo

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def convert_space_to_carbs_space(space: dict[str, Any]):
    for attr_name, attr_value in space.items():
        if isinstance(attr_value, (neps.HPOInteger, neps.HPOFloat)):
            search_center = (attr_value.domain.lower + attr_value.domain.upper) / 2 if not attr_value.prior else attr_value.prior
            # search_center = None
            if attr_value.log:
                yield Param(
                    name=attr_name,
                    space=LogSpace(
                        # scale=- 1/ math.log(attr_value.upper / attr_value.lower),
                        scale=math.log(math.e, 10),
                        min=attr_value.domain.lower,
                        max=attr_value.domain.upper,
                        is_integer=isinstance(attr_value, (neps.HPOInteger)),
                    ),
                    search_center=search_center,
                )
            elif 0 <= attr_value.domain.lower and attr_value.domain.upper <= 1:
                yield Param(
                    name=attr_name,
                    space=LogitSpace(
                        scale=1.0,
                        min=attr_value.domain.lower,
                        max=attr_value.domain.upper,
                        is_integer=isinstance(attr_value, (neps.HPOInteger)),
                    ),
                    search_center=search_center,
                )
            else:
                yield Param(
                    name=attr_name,
                    space=LinearSpace(
                        scale=3 * (attr_value.domain.upper - attr_value.domain.lower),
                        min=attr_value.domain.lower,
                        max=attr_value.domain.upper,
                        is_integer=isinstance(attr_value, (neps.HPOInteger)),
                    ),
                    search_center=search_center,
                )
        elif isinstance(attr_value, (int, float, str, neps.HPOConstant)):
            continue
        else:
            raise NotImplementedError(f"Space type {type(attr_value)} is not supported.")

def evaluate_pipeline_wrapper(carbs_opt, evaluate_pipeline):
    def eval_pipeline(**config):
        res = evaluate_pipeline(**config)
        obs_out = carbs_opt.observe(ObservationInParam(input=config, output=res["objective_to_minimize"], cost=res["cost"]))
        return res
    return eval_pipeline

class CARBSSuggester(ScalingLawGuidedOptimizer, AskFunction):
    
    target_over_spent_flops_ratios = np.array([0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 3.0])
    spent_over_max_flops_ratios = [(0.015, 0.005), (0.09, 0.11), (0.45, 0.55), (0.95, 1.05), (1.45, 1.55), (1.95, 2.05)]
    
    def __init__(self, space, *, carbs_opt, use_priors,
                 params_estimator: Callable, 
                 seen_datapoints_estimator: Callable, 
                 flops_estimator: Callable,
                 evaluate_pipeline: Callable,
                 max_evaluation_flops: float,
                 data_parameter_name
                ):
        self.carbs = carbs_opt
        self.space: neps.SearchSpace = space
        self.params_estimator = params_estimator
        self.evaluate_pipeline = evaluate_pipeline
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.flops_estimator = flops_estimator
        self.min_flops, self.max_flops = self.find_flops_bounds()
        self.use_priors = use_priors
        self.max_evaluation_flops = max_evaluation_flops
        self.data_parameter_name = data_parameter_name
        self._artifacts = []

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n = None,
    ) -> SampledConfig | list[SampledConfig]:
        suggestion = self.carbs.suggest().suggestion
        sug = copy.deepcopy(suggestion)
        sug.pop("suggestion_uuid")
        if self.use_priors:
            for name, param in self.space.searchables.items():
                if param.prior is not None:
                    print(f"Using prior for param {name}: {param.prior}")
                    sug.update({name: param.prior})
        print("CARBS suggestion: ", sug)
        return SampledConfig(id=str(len(trials)), config=sug)
    
    def find_flops_bounds(self) -> tuple[float, float]:
        attrs = self.space.searchables
        conf_max = {param_name: param.domain.upper for param_name, param in attrs.items()}
        conf_min = {param_name: param.domain.lower for param_name, param in attrs.items()}
        return self.flops_estimator(**conf_min), self.flops_estimator(**conf_max)

    @staticmethod
    def _fit_linear_line(xs: list, ys: list) -> tuple[list, float, float] | None:
        """Fit line in log-log space and return fitted values, slope, and intercept.
        
        Uses the same approach as Chinchilla_Guided_Scaling._fit_loglog_line()
        """
        try:
            
            if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                return None
            
            coeffs = np.polyfit(xs, ys, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            if not np.isfinite(slope):
                return None
            
            fit_line = np.poly1d(coeffs)
            fitted_ys = fit_line(xs)
            
            return fitted_ys.tolist(), slope, intercept
        except Exception as e:
            logger.debug(f"Could not fit loglog line: {e}")
            return None

    def _find_optimal_data_parameter_value_for_target_flops(
        self,
        conf: Mapping[str, Any],
        target_flops: float,
        data_values,

    ) -> float | None:
        """Use binary search to find data_value where estimated_flops is closest to target_flops."""
        if not data_values:
            return None
        
        conf_copy = copy.deepcopy(conf)
        data_values_sorted = sorted(data_values)
        
        left, right = 0, len(data_values_sorted) - 1
        closest_idx = 0
        closest_diff = float('inf')
        
        # Binary search to narrow down the range
        while left <= right:
            mid = (left + right) // 2
            conf_copy[self.data_parameter_name] = data_values_sorted[mid]
            estimated_flops = self.flops_estimator(**conf_copy)
            diff = abs(estimated_flops - target_flops)
            
            # Track the closest value found so far
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = mid
            
            # Continue search based on whether we're above or below target
            if estimated_flops < target_flops:
                left = mid + 1
            else:
                right = mid - 1
        
        # Check neighbors to find true closest
        for idx in [max(0, closest_idx - 1), closest_idx, min(len(data_values_sorted) - 1, closest_idx + 1)]:
            conf_copy[self.data_parameter_name] = data_values_sorted[idx]
            estimated_flops = self.flops_estimator(**conf_copy)
            diff = abs(estimated_flops - target_flops)
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = idx
        
        return data_values_sorted[closest_idx]
        
        
    def _transform_hp_to_basic_space(self, hp_values: list[float], param_name: str) -> list[float]:
        """Transform hyperparameter values from param space to basic space.
        
        Args:
            hp_values: List of values in param space
            param_name: Name of the parameter
        
        Returns:
            List of values in basic space (transformed domain)
        """
        # real_number_space_by_name = self.get_real_number_space_by_name()
        # transform_fn_per_param = {k: v.basic_from_param for k, v in real_number_space_by_name.items()}
        transform_fn_per_param = {param_name: math.log10}

        if param_name not in transform_fn_per_param:
            raise ValueError(f"No transformation function found for parameter '{param_name}'")
        return [transform_fn_per_param[param_name](val) for val in hp_values]
        
    def _transform_hp_from_basic_space(self, hp_value: float, param_name: str) -> float:
        """Transform hyperparameter value from basic space back to param space.
        
        Args:
            hp_value: Value in basic space
            param_name: Name of the parameter
        
        Returns:
            Value in param space (original domain)
        """
        # real_number_space_by_name = self.get_real_number_space_by_name()
        # transform_fn_per_param = {k: v.param_from_basic for k, v in real_number_space_by_name.items()}
        transform_fn_per_param = {param_name: lambda x: 10 ** x}
        if param_name not in transform_fn_per_param:
            raise ValueError(f"No transformation function found for parameter '{param_name}'")
        return transform_fn_per_param[param_name](hp_value)

    def _find_optimal_hp_value_for_target_flops(
        self,
        trials: Mapping[str, Trial],
        target_flops: float
    ) -> dict[str, float] | None:
        """Find optimal hyperparameter values for a given target FLOP budget using Pareto front.
        
        Follows the Chinchilla approach of fitting FLOPS vs hyperparameters in log-log space
        (in basic/transformed space for consistency) and then extrapolating to the target FLOPS.
        
        Returns:
            Dict of {param_name: extrapolated_value} or None if extrapolation fails
        """
        data_param = self.space.searchables[self.data_parameter_name]
        data_grid_values = torch.linspace(data_param.domain.lower, data_param.domain.upper, steps=10000).tolist()
        # Get Pareto front from trials
        result = fit_pareto_front_loglog(trials=trials, lc_aware=False)
        pareto_trial_ids = result[7]  # Trial IDs on Pareto front
        
        if not pareto_trial_ids:
            logger.warning("No Pareto front found")
            return None
        
        # Extract pareto trials
        pareto_trials = [trials[trial_id] for trial_id in pareto_trial_ids]
        
        # Collect FLOPS and hyperparameter values from pareto trials (in param space initially)
        # Only include trials that have all hyperparameters to keep arrays aligned
        flops_list = []
        hp_values_by_name = {name: [] for name in trials[next(iter(trials))].config.keys()}
        
        for trial in pareto_trials:
            # Estimate FLOPS for this trial
            # For CARBS, we need to extract from trial config
            if trial.report is None or trial.report.cost is None:
                continue
            
            # Check that trial has all parameters - skip if any are missing
            if not all(param_name in trial.config for param_name in hp_values_by_name.keys()):
                continue
            
            flops_list.append(float(trial.report.cost))  # Use cost as FLOPS proxy
            
            # Extract hyperparameter values (in param space) only for valid trials
            for param_name in hp_values_by_name.keys():
                hp_values_by_name[param_name].append(trial.config[param_name])
        
        if len(flops_list) < 2:
            logger.warning(f"Not enough Pareto points ({len(flops_list)}) for fitting")
            return None
        
        # Fit lines in log-log space for each hyperparameter vs FLOPS
        # Transform to basic space before fitting for consistency with CARBS
        extrapolated_hp = {}
        for param_name, hp_values in hp_values_by_name.items():
            if not hp_values or len(set(hp_values)) < 2:
                if len(set(hp_values)) == 1:
                    extrapolated_hp[param_name] = hp_values[0]
                    continue
                raise Exception(f"not enough points for the hp {param_name} for extrapolate")
            
            # Transform hp_values to basic space before fitting
            hp_values_basic = self._transform_hp_to_basic_space(hp_values, param_name)
            assert len(hp_values_basic) == len(flops_list), \
            f"Length mismatch after transformation for {param_name}"
           
            valid_flops = flops_list
            valid_flops = [math.log10(f) for f in valid_flops]
            valid_hp_basic = hp_values_basic
            
            fit_result = self._fit_linear_line(valid_flops, valid_hp_basic)
            if fit_result is None:
                logger.debug(f"Could not fit line for {param_name}")
                raise ValueError(f"Could not fit line for {param_name}: polyfit returned None")
            
            fitted_ys, slope, intercept = fit_result
            
            logger.debug(f"Parameter '{param_name}' (in basic space): slope={slope:.4f}, intercept={intercept:.4f}, "
                        f"valid_points={len(valid_flops)}/{len(hp_values_basic)}")
            
            target_hp_basic = slope * math.log10(target_flops) + intercept
            
            target_hp_param = self._transform_hp_from_basic_space(float(target_hp_basic), param_name)
            extrapolated_hp[param_name] = target_hp_param
        
        n_tokens = self._find_optimal_data_parameter_value_for_target_flops(extrapolated_hp, target_flops, data_grid_values)
        extrapolated_hp[self.data_parameter_name] = n_tokens

        if not extrapolated_hp:
            logger.warning("Could not extrapolate any hyperparameters")
            return None
        
        return extrapolated_hp
            
    def extrapolate_for_target_flops(
        self,
        trials: Mapping[str, Trial],
        target_flops: list[float],
        optimal_points: dict[float, tuple]
    ) -> dict | None:
        """Find best configuration closest to target FLOPS by searching hyperparameter space.
        
        Uses the extrapolated hyperparameters as guidance and performs a grid search
        to find the configuration that gets closest to the target FLOPS without exceeding it.
        
        Returns:
            Tuple of (best_config, actual_flops) or None if no valid config found
        """

        spent_flops = sum([self.flops_estimator(**trial.config) for trial in trials.values()])
        extrapolation_per_target_flop = {}
        for target in target_flops:
            # Get extrapolated hyperparameter values as initial guidance
            try:
                extrapolated_hp = self._find_optimal_hp_value_for_target_flops(trials, target)
            except Exception as e:
                logger.warning(f"Extrapolation failed for target FLOPS {target}: {e}")
                extrapolated_hp = None
            if extrapolated_hp is None:
                logger.warning("Could not extrapolate hyperparameters")
                return None
            flops = self.flops_estimator(**extrapolated_hp)
            params = self.params_estimator(**extrapolated_hp)
            loss = self.evaluate_pipeline(**extrapolated_hp)["objective_to_minimize"]
            extrapolation_per_target_flop[target] = {
                'target_flop': float(target),
                'flops': flops,
                'target_spent_flops_ratio': float(flops/spent_flops),
                'params': params,
                'loss': loss,
                'optimal_flops': optimal_points.get(target, (None, None, None))[0],
                'optimal_params': optimal_points.get(target, (None, None, None))[1],
                'optimal_loss': optimal_points.get(target, (None, None, None))[2],
                'config': extrapolated_hp,
            }
        return {
            'spent_flops': float(spent_flops),
            'max_evaluation_flops': float(self.max_evaluation_flops),
            'spent_max_flops_ratio': float(spent_flops/self.max_evaluation_flops),
            'configs': extrapolation_per_target_flop,
            'fits': None,
        }
    def _plot_extrapolation_horizon(self, trials: Mapping[str, Trial], extrapolation_data: dict) -> Figure| None:

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        
        # Extract previously evaluated points from Pareto front
        result = fit_pareto_front_loglog(trials, lc_aware=False)
        if result is None:
            logger.warning("Could not extract Pareto front from trials")
            return None
        
        pareto_rows = result[5]
        all_rows = result[6 ]
        
        actual_flops = [r[3] for r in pareto_rows]      # flops
        actual_params = [float(r[0]) for r in pareto_rows]  # n_params
        actual_loss = [r[2] for r in pareto_rows]       # loss

        all_flops = [r[3] for r in all_rows] if all_rows else []
        all_params = [r[0] for r in all_rows] if all_rows else []
        all_loss = [r[2] for r in all_rows] if all_rows else []
        
        # Extract predicted and optimal points from extrapolation_data
        configs = extrapolation_data.get('configs', {})
        
        predicted_flops = []
        predicted_params = []
        predicted_loss = []
        
        optimal_flops = []
        optimal_params = []
        optimal_loss = []
        
        for target, result_dict in configs.items():
            # Predicted points
            if result_dict.get('flops') is not None:
                predicted_flops.append(result_dict['flops'])
                predicted_params.append(result_dict.get('params', 0))
                predicted_loss.append(result_dict.get('loss', 0))
            
            # Optimal points (from ground truth)
            if result_dict.get('optimal_flops') is not None:
                optimal_flops.append(result_dict['optimal_flops'])
                optimal_params.append(result_dict.get('optimal_params', 0))
                optimal_loss.append(result_dict.get('optimal_loss', 0))
        
        # Create Figure 1: FLOPs vs Params
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1 = axes[0]
        ax2 = axes[1]

        if all_flops:
            ax1.scatter(all_flops, all_params, c='gray', marker='o', s=30, alpha=0.3, label='All evaluated trials', zorder=1)
    
        # Plot evaluated points
        ax1.scatter(actual_flops, actual_params, c='blue', marker='o', s=80, 
                    alpha=0.6, label='Evaluated trials', edgecolors='darkblue', linewidth=1)
        
        # Plot predicted points
        if predicted_flops:
            ax1.scatter(predicted_flops, predicted_params, c='green', marker='^', s=80, 
                    alpha=0.8, edgecolors='darkgreen', linewidth=1.5, label='Predicted', zorder=3)
        
        # Plot optimal points
        if optimal_flops:
            ax1.scatter(optimal_flops, optimal_params, c='gold', marker='*', s=100, 
                edgecolors='orange', linewidth=2, label='Optimal (true Pareto)', zorder=5)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('FLOPs', fontsize=12)
        ax1.set_ylabel('Parameters', fontsize=12)
        ax1.set_title('Extrapolation Horizon: FLOPs vs Parameters', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        plt.tight_layout()
        

        # Plot all evaluated trials in gray (background)
        if all_flops:
            ax2.scatter(all_flops, all_loss, c='gray', marker='o', s=30, alpha=0.3, label='All evaluated trials', zorder=1)
    
        
        # Plot evaluated points
        ax2.scatter(actual_flops, actual_loss, c='blue', marker='o', s=70, 
                    alpha=0.8, label='Pareto optimal', edgecolors='darkblue', linewidth=1)
        
        # Plot predicted points
        if predicted_flops:
            ax2.scatter(predicted_flops, predicted_loss, c='green', marker='^', s=100, 
                        alpha=0.8, label='Predicted points', edgecolors='darkgreen', linewidth=1.5)
        
        # Plot optimal points
        if optimal_flops:
            ax2.scatter(optimal_flops, optimal_loss, c='gold', marker='*', s=100, 
                        alpha=0.9, label='Optimal points (ground truth)', 
                        edgecolors='orange', linewidth=2)
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('FLOPs', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Extrapolation Horizon: FLOPs vs Loss', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        plt.tight_layout()
        
        return fig
            
        
    def get_real_number_space_by_name(self):
        checkpoint_dir = Path(self.carbs.config.checkpoint_dir) / "carbs_experiment"
        # Find the latest checkpoint file (with biggest number in carbs_{i}obs.pt)
        checkpoint_files = list(checkpoint_dir.glob("carbs_*obs.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.stem.split('_')[1].replace('obs', '')))
        carbs = CARBS.load_from_file(str(latest_checkpoint))
        
        return carbs._real_number_space_by_name
    
    def extrapolate(self, trials: Mapping[str, Trial], true_pareto_info: dict) -> dict[str, Any] | None:
        
        trials_sorted = OrderedDict(sorted(trials.items(), key=lambda item: item[1].metadata.time_sampled))
        
        extrapolation_results_by_spent_ratio = {}  # Accumulate results for heatmap
        extrapolation_data = {}  # Initialize to empty dict
        
        for i in range(len(trials_sorted)):
            consecutive_trials = dict(list(trials_sorted.items())[:i])
            spent_flops = sum([self.flops_estimator(**trial.config) for trial in consecutive_trials.values()])
            jitter = 0.1
            chosen_ratio = None
            for ratio in self.spent_over_max_flops_ratios:
                if ratio[0] <= spent_flops / self.max_flops <= ratio[1] + jitter:
                    chosen_ratio = ratio
                    break
            if chosen_ratio is None:
                continue
            ratio_center = float((chosen_ratio[0] + chosen_ratio[1]) / 2)
            if ratio_center in extrapolation_results_by_spent_ratio:
                continue
            
            ratio_str = f"{ratio_center:.3f}".replace(".", "_")
            logger.info(f"Trial {i+1}/{len(trials_sorted)}: Spent FLOPs so far: {spent_flops:.2e} ratio {ratio_center:.2f}")

            selected_target_flops = np.unique(
                np.clip(
                    spent_flops * self.target_over_spent_flops_ratios, 
                    self.min_flops,
                    self.max_flops,
                )
            ).tolist()
            optimal_points = {}
            used_indices = set()  # Track already-selected indices
            
            for target in selected_target_flops:
                distances = np.abs(np.array(true_pareto_info["flops"]) - target)
                
                # Find closest index not yet used
                sorted_indices = np.argsort(distances)
                closest_idx = None
                for idx in sorted_indices:
                    if idx not in used_indices:
                        closest_idx = idx
                        break
                
                if closest_idx is None:
                    logger.warning(f"No available points left for target {target:.2e}")
                    continue
                
                used_indices.add(closest_idx)
                closest_flop, optimal_param, optimal_loss = true_pareto_info["flops"][closest_idx], true_pareto_info["n_params"][closest_idx], true_pareto_info["loss"][closest_idx]
                optimal_points[target] = (closest_flop, optimal_param, optimal_loss)

        
            extrapolation_data = None
            
            extrapolation_data = self.extrapolate_for_target_flops(consecutive_trials, selected_target_flops, optimal_points)
            if extrapolation_data is not None:
                extrapolation_results_by_spent_ratio[ratio_center] = extrapolation_data
                # Add extrapolation plots
                try:
                    fig_extrapolation = self._plot_extrapolation_horizon(consecutive_trials, extrapolation_data)
                    if fig_extrapolation is not None:
                        self._artifacts.append(
                            Artifact(f"extrapolation_horizon_plot_{ratio_str}", fig_extrapolation, ArtifactType.FIGURE)
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate extrapolation horizon plot: {e}")
                
                # Add extrapolation JSON artifact
                self._artifacts.append(
                    Artifact(f"extrapolation_results_{ratio_str}", extrapolation_data, ArtifactType.JSON)
                )
        
        # create heatmaps
        # import json
        # with open(Path("/work/dlclarge1/alipourn-benchmark_res/llm/carbs/seed_40/summary/extrapolation_results_all_ratios.json"), "r") as f:
        #     extrapolation_results_by_spent_ratio = json.load(f)
        if extrapolation_results_by_spent_ratio:
            heatmap_result = Chinchilla_Guided_Scaling._plot_interpolation_extrapolation_heatmap(extrapolation_results_by_spent_ratio)
            if heatmap_result is not None:
                fig_loss, fig_params = heatmap_result
                self._artifacts.append(
                    Artifact("loss_prediction_quality_heatmap", fig_loss, ArtifactType.FIGURE)
                )
                self._artifacts.append(
                    Artifact("params_prediction_quality_heatmap", fig_params, ArtifactType.FIGURE)
                )
            
            try:
                extrapolation_results_json = {}
                for spent_ratio, result_data in extrapolation_results_by_spent_ratio.items():
                    extrapolation_results_json[float(spent_ratio)] = result_data
                
                self._artifacts.append(
                    Artifact("extrapolation_results_all_ratios", extrapolation_results_json, ArtifactType.JSON)
                )
            except Exception as e:
                logger.warning(f"Failed to serialize extrapolation results to JSON artifact: {e}")
        return extrapolation_results_by_spent_ratio

        
    def get_trial_artifacts(
        self,
        trials):
        artifacts = super().get_trial_artifacts(trials) or []
        artifacts.extend(self._artifacts)
        return artifacts

    def import_trials(
        self,
        external_evaluations,
        trials):
        return []
    

def get_carbs_optimizer(space: neps.SearchSpace, seed: int, 
                        use_priors: bool, max_evaluation_flops: float, params_estimator: Callable, 
                        seen_datapoints_estimator: Callable, flops_estimator: Callable,
                        evaluate_pipeline: Callable,
                        data_parameter_name,
                        root_dir) -> tuple[Callable[..., Any] | CustomOptimizer, CARBS]:
    space = convert_neps_to_classic_search_space(space)
    space_copy = dict(copy.deepcopy(space).searchables)
    if use_priors:
        for name, param in space.searchables.items():
            if param.prior is not None:
                print(f"Using prior for param {name}: {param.prior}")
                del space_copy[name]
    param_spaces = list(convert_space_to_carbs_space(space_copy))
    checkpoint_dir = Path(root_dir) / "checkpoints"
    carbs_params = CARBSParams(
        num_random_samples=4,
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        initial_search_radius=0.15,
    )
    print("param_spaces: ", param_spaces)
    carbs_opt = CARBS(carbs_params, param_spaces)
    opt = custom(
        name="carbs", 
        optimizer=lambda pipeline_space: CARBSSuggester(
            convert_neps_to_classic_search_space(pipeline_space), carbs_opt=carbs_opt, use_priors=use_priors, 
            params_estimator=params_estimator, 
            seen_datapoints_estimator=seen_datapoints_estimator,
            flops_estimator=flops_estimator,
            max_evaluation_flops=max_evaluation_flops,
            data_parameter_name=data_parameter_name,
            evaluate_pipeline=evaluate_pipeline,
        ),
    )
    return opt, carbs_opt
