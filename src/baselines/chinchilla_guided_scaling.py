from __future__ import annotations

from collections import defaultdict
import copy
import math

import torch
import itertools
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass

import logging
from typing import (
    TYPE_CHECKING,
    Any, Literal
)
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, TimeoutError


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from neps.state import BudgetInfo, Trial
from neps.optimizers.optimizer import Artifact, ArtifactType
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer, fit_convex_hull_loglog, fit_pareto_front_loglog
from neps.optimizers.grid_search import GridSearch
from neps.optimizers.optimizer import ImportedConfig, AskFunction
from neps.optimizers.algorithms import custom, CustomOptimizer
from neps.space.search_space import HPOInteger, HPOFloat, HPOCategorical
from neps.sampling.samplers import Sampler
from neps.space import Domain as NepsDomain
from neps.space.neps_spaces.neps_space import convert_neps_to_classic_search_space
if TYPE_CHECKING:
    from neps.state import Trial
    from neps.state.pipeline_eval import UserResultDict

logger = logging.getLogger(__name__)


@dataclass
class ScalingLawParameters:
    """Fitted scaling law parameters: L = E + A/N^α + B/D^β"""
    E: float
    A: float
    B: float
    alpha: float
    beta: float

class Chinchilla_Guided_Scaling(ScalingLawGuidedOptimizer, AskFunction):
    """The Multi objective algorithm for search space including architectural choices."""

    def __init__(self,
                space,
                flops_estimator,
                params_estimator,
                seen_datapoints_estimator,
                aspect_ratio_fn,
                max_evaluation_flops,
                evaluate_pipeline,
                data_parameter_name,
                strategy: Literal["approach1", "approach2", "gemstones"] = "gemstones"
                ) -> None:
        
        
        self._scaling_law_params: ScalingLawParameters | None = None
        self._best_candidate: dict[str, Any] | None = None
        self._best_loss: float | None = None

        self.params_estimator = params_estimator
        self.evaluate_pipeline = evaluate_pipeline
        self.seen_datapoints_estimator = seen_datapoints_estimator
        self.flops_estimator = flops_estimator
        self.aspect_ratio_fn = aspect_ratio_fn
        self.space = space
        self.data_parameter_name = data_parameter_name
        self.strategy = strategy
        if self.strategy == "approach2":
            self.iso_flops = self._generate_iso_flop_list()
            logger.info(f"Generated {len(self.iso_flops)} ISO-FLOP values: {[f'{f:.2e}' for f in self.iso_flops]}... (total: {sum(self.iso_flops):.2e})")
        
        self.min_flops, self.max_flops = self.find_flops_bounds()
        
        self.target_over_spent_flops_ratios = np.arange(0.1, 2.1, 0.1)

        self.max_evaluation_flops = max_evaluation_flops
        

        match self.strategy:
            case "approach1":
                config_list, selected_model_configs = self.approach1_generate_config_list()
            case "approach2":
                config_list, selected_model_configs = self.approach2_generate_config_list(num_candidate_per_iso_flop=10)
            case "gemstones":
                config_list, selected_model_configs = self.gemstones_generate_config_list()

        self.all_possible_model_configs = selected_model_configs
        self.config_list = config_list
        print(f"config list {config_list}, total number {len(config_list)}")
        self.base_optimizer = GridSearch( #type: ignore
            configs_list=config_list,
        )
    
    def find_flops_bounds(self) -> tuple[float, float]:
        conf_max = {param_name: param.domain.upper for param_name, param in self.space.searchables.items()}
        conf_min = {param_name: param.domain.lower for param_name, param in self.space.searchables.items()}
        return self.flops_estimator(**conf_min), self.flops_estimator(**conf_max)


    def _find_data_param_value_for_ratio(self, config: dict, target_ratio: float, data_param: Any) -> float:
        # Get domain bounds
        domain_min, domain_max = data_param.domain.lower, data_param.domain.upper
        
        # Estimate N (parameters) from the current config
        n_params = self.params_estimator(**config)
        target_datapoints = target_ratio * n_params
        
        # Binary search for the data_param value
        low, high = domain_min, domain_max
        best_value = domain_min
        best_error = float('inf')
        
        for _ in range(100):  # 20 iterations of binary search
            mid = int((low + high) / 2)
            test_config = config.copy()
            test_config[self.data_parameter_name] = mid
            
            # Estimate D (datapoints) for this data_param value
            try:
                d_estimate = self.seen_datapoints_estimator(**test_config)
            except Exception as e:
                logger.warning(f"Could not estimate datapoints for {self.data_parameter_name}={mid}: {e}")
                high = mid
                continue
            
            error = abs(d_estimate - target_datapoints)
            if error < best_error:
                best_error = error
                best_value = mid
            
            if d_estimate < target_datapoints:
                low = mid
            else:
                high = mid
        
        return float(np.clip(best_value, domain_min, domain_max))
    
    def approach1_generate_config_list(self):
        parameters = self.space.searchables
        non_scaling_params = self.sample_value_for_parameters(param_types="non-scaling")
        data_param_lower = parameters[self.data_parameter_name].domain.lower
        model_params_grid = {
            param_name: sorted(torch.unique(
                param.domain.cast(
                    torch.linspace(0, 1, steps=10), frm=NepsDomain.unit_float()
                )
            ).tolist())
            for param_name, param in parameters.items() 
            if getattr(param, "is_scaling", False) and param_name != self.data_parameter_name
        }
        logger.info(f"grid for model params {model_params_grid}")
        
        param_names = list(model_params_grid.keys())
        
        current_grid_indices = {name: 0 for name in param_names}

        conf1_scaling = {
            param_name: grid[0]
            for param_name, grid in model_params_grid.items()
        }
        
        # Estimate parameters for the base configuration
        base_config = {**conf1_scaling, **non_scaling_params, self.data_parameter_name: data_param_lower}  # data parameter value will be set later
        n_params = self.params_estimator(**base_config)
        
        # Get domain bounds and data parameter
        data_param = parameters[self.data_parameter_name]
        
        # Create dynamic data grid using Chinchilla principle: find data_param values for different multipliers
        # Target multipliers: 5*N, 10*N, 20*N, 80*N (or adjusted based on domain)
        data_multipliers = [5, 10, 20, 80]
        data_grid = []
        
        for multiplier in data_multipliers:
            data_param_value = self._find_data_param_value_for_ratio(base_config, multiplier, data_param)
            data_grid.append(data_param_value)
        
        data_grid = sorted(list(set(data_grid)))  # Remove duplicates and sort
        
        logger.info(f"Estimated params: {n_params}, Dynamic data grid (multipliers {data_multipliers}): {[f'{v:.2e}' for v in data_grid]}")
        
        config_list = [
            {**conf1_scaling, **non_scaling_params, self.data_parameter_name: data_grid[i]} 
            for i in range(len(data_grid))
        ]
        
        # Track all attempted model parameter combinations
        attempted_model_configs = defaultdict(list)  
        # Add initial config
        initial_model_key = tuple(sorted((k, v) for k, v in config_list[0].items() if k != self.data_parameter_name))
        conf1_n_params = self.params_estimator(**config_list[0])
        attempted_model_configs[conf1_n_params].append(initial_model_key)
        print(config_list)
        
        assert self.flops_estimator is not None and self.max_evaluation_flops is not None
        
        used_flops = sum([self.flops_estimator(**conf) for conf in config_list])
        print(f"used flops {used_flops:.2e}")
        
        action_probabilities = torch.tensor([1.0 / len(param_names)] * len(param_names))
        action_count = 0
        
        exhausted_dims = set()
        while used_flops < self.max_evaluation_flops:
            if len(exhausted_dims) == len(param_names):
                logger.info("All parameters have reached their grid maximums.")
                break
                
            action = self._choose_action_with_sobol(action_probabilities, action_count)
            action_count += 1
            
            param_name = param_names[action]
            print(f"action is {action} parameter to increment {param_name}")
            
            if param_name in exhausted_dims:
                continue
                
            current_idx = current_grid_indices[param_name]
            next_idx = current_idx + 1
            
            if next_idx >= len(model_params_grid[param_name]):
                logger.info(f"Exhausted grid for parameter '{param_name}'.")
                exhausted_dims.add(param_name)
                action_probabilities[action] = 0.0
                if action_probabilities.sum() > 0:
                    action_probabilities = action_probabilities / action_probabilities.sum()
                continue
                
            current_config = config_list[-1].copy()
            current_config[param_name] = model_params_grid[param_name][next_idx]
            
            # Track this model parameter combination (regardless of whether data fits budget)
            model_key = tuple(sorted((k, v) for k, v in current_config.items() if k != self.data_parameter_name))
            
            # Recompute dynamic data grid for updated scaling parameters
            n_params_updated = self.params_estimator(**current_config)
            attempted_model_configs[n_params_updated].append(model_key)
            data_param = parameters[self.data_parameter_name]
            
            # Find data_param values for each multiplier using the updated N
            updated_data_grid = []
            for multiplier in [5, 10, 20, 80]:
                data_param_value = self._find_data_param_value_for_ratio(current_config, multiplier, data_param)
                updated_data_grid.append(data_param_value)
            
            updated_data_grid = sorted(list(set(updated_data_grid)))  # Remove duplicates and sort
            
            for data_grid_value in updated_data_grid:
                current_config[self.data_parameter_name] = data_grid_value
                copy_current = current_config.copy()
                estimated_flops = self.flops_estimator(**copy_current)
                if used_flops + estimated_flops > self.max_evaluation_flops:
                    logger.info(f"Cannot add config {copy_current} due to FLOPs constraint. "
                                f"Used: {used_flops:.2e}, Estimated: {estimated_flops:.2e}, Max: {self.max_evaluation_flops:.2e}")
                    continue
                config_list.append(copy_current)
                used_flops += estimated_flops
                logger.info(f"Added config {copy_current}. Total used FLOPs: {used_flops:.2e}")
            
            current_grid_indices[param_name] = next_idx
        
        # model params + non scaling params (exclude data param)
        logger.info(f"Selected {len(config_list)} unique model configs for approach1 (including those exceeding FLOP budget)")
        
        return config_list, attempted_model_configs
    
    def approach2_generate_config_list(self, num_candidate_per_iso_flop: int = 5) -> tuple[list[dict[str, Any]], defaultdict[int, list[tuple]]]:
        parameters = self.space.searchables
        non_scaling_params = self.sample_value_for_parameters(param_types="non-scaling")
        
        scaling_non_data_grid = {
            param_name: torch.unique(
                param.domain.cast(
                    torch.linspace(0, 1, steps=10), frm=NepsDomain.unit_float()
                )
            ).tolist() 
            for param_name, param in parameters.items() 
            if getattr(param, "is_scaling", False) and param_name != self.data_parameter_name
        }
        
        data_grid = sorted(torch.unique(
            parameters[self.data_parameter_name].domain.cast(
                torch.linspace(0, 1, steps=1000), frm=NepsDomain.unit_float()
            )
        ).tolist())
        
        param_names = list(scaling_non_data_grid.keys())
        current_grid_indices = {name: 0 for name in param_names}
        configs_per_iso_flop = defaultdict(list)
        
        conf_scaling = {
            param_name: grid[0]
            for param_name, grid in scaling_non_data_grid.items()
        }

        action_probabilities = torch.tensor([1.0 / len(param_names)] * len(param_names))
        action_count = 0
        exhausted_dims = set()
        used_flops = sum([
            self.flops_estimator(**conf) 
            for configs in configs_per_iso_flop.values() 
            for conf in configs
        ])
        
        # Track all selected model parameter combinations
        selected_model_configs = defaultdict(list)
        initial_non_data_conf = {**conf_scaling, **non_scaling_params}
        initial_model_key = tuple(sorted((k, v) for k, v in initial_non_data_conf.items()))
        initial_param_count = self.params_estimator(**initial_non_data_conf | {self.data_parameter_name: data_grid[0]})  # Ensure initial config is valid for params estimator
        selected_model_configs[initial_param_count].append(initial_model_key)
        
        retry_counter = 0
        while used_flops < self.max_evaluation_flops and len(exhausted_dims) < len(param_names):
            if retry_counter >= 50:
                logger.info("Retry limit reached")
                break
            
            action = self._choose_action_with_sobol(action_probabilities, action_count)
            action_count += 1
            
            param_name = param_names[action]
            
            if param_name in exhausted_dims:
                continue
            
            current_idx = current_grid_indices[param_name]
            next_idx = current_idx + 1
            
            if next_idx >= len(scaling_non_data_grid[param_name]):
                logger.info(f"Exhausted grid for parameter '{param_name}'.")
                exhausted_dims.add(param_name)
                action_probabilities[action] = 0.0
                if action_probabilities.sum() > 0:
                    action_probabilities = action_probabilities / action_probabilities.sum()
                continue
            
            conf_scaling = conf_scaling.copy()
            conf_scaling[param_name] = scaling_non_data_grid[param_name][next_idx]
            current_grid_indices[param_name] = next_idx
            
            # Track this model config
            conf_non_data = {**conf_scaling, **non_scaling_params}
            model_key = tuple(sorted((k, v) for k, v in conf_non_data.items()))
            conf_parameter_count = self.params_estimator(**conf_non_data | {self.data_parameter_name: data_grid[0]})
            selected_model_configs[conf_parameter_count].append(model_key)
            replace = False
            added = False
            for data_val in data_grid:
                config = {**conf_scaling, **non_scaling_params, self.data_parameter_name: data_val}
                conf = config.copy()
                estimated_flops = self.flops_estimator(**conf)
                available_brackets = [flops for flops in self.iso_flops if len(configs_per_iso_flop[flops]) <= num_candidate_per_iso_flop]
                if not available_brackets:
                    logger.info(f"No available ISO-FLOP brackets for config {conf} with estimated FLOPs {estimated_flops:.2e}. Skipping.")
                    replace = True
                closest_flop_bin, ratio = self._get_closest_iso_flop(estimated_flops, available_brackets or self.iso_flops)
                if ratio > 1.1 or ratio < 0.9:
                    continue
                if replace:
                    logger.info(f"Replacing config in ISO-FLOP bin {closest_flop_bin:.2e} with config {conf} "
                                f"due to lack of available brackets. Estimated FLOPs: {estimated_flops:.2e}")
                    assert configs_per_iso_flop[closest_flop_bin], f"No configs to replace in ISO-FLOP bin {closest_flop_bin:.2e}"
                    removed_conf = configs_per_iso_flop[closest_flop_bin][0]
                    removed_flops = self.flops_estimator(**removed_conf)
                    if used_flops - removed_flops + estimated_flops <= self.max_evaluation_flops:
                        configs_per_iso_flop[closest_flop_bin].pop(0)
                        configs_per_iso_flop[closest_flop_bin].append(conf)
                        used_flops += estimated_flops - removed_flops
                    continue
                if used_flops + estimated_flops > self.max_evaluation_flops:
                    continue
                configs_per_iso_flop[closest_flop_bin].append(conf)
                used_flops += estimated_flops
                added = True
            
            if not added:
                retry_counter += 1
            else:
                retry_counter = 0
        
        print(f"configs_per_iso_flop {configs_per_iso_flop}")
        config_list = [
            conf 
            for configs in configs_per_iso_flop.values() 
            for conf in configs
        ]
        
        logger.info(f"Generated {len(config_list)} configs across {len(configs_per_iso_flop)} ISO-FLOP levels, "
                   f"total FLOPs: {used_flops:.2e} / {self.max_evaluation_flops:.2e}")
        
        logger.info(f"Selected {len(config_list)} unique model configs for approach2")
        
        return config_list, selected_model_configs

    def gemstones_generate_config_list(self):
        """Generate configs using parameter binning with aspect ratio sorting.
        
        Strategy:
        1. Create parameter bins from all possible configurations
        2. Select 5 parameter bins with ±5% threshold
        3. For top 5 most populated bins, iteratively select configs
        4. Sort configs by aspect ratio and select 3 strategic indices:
           - (first+middle)/2: between start and middle
           - middle: center point
           - (middle+end)/2: between middle and end
        """
        parameters = self.space.searchables
        non_scaling_params = self.sample_value_for_parameters(param_types="non-scaling")
        steps = 30
        scaling_non_data_grid = {
            param_name: sorted(torch.unique(
                param.domain.cast(
                    torch.linspace(0, 1, steps=steps), frm=NepsDomain.unit_float()
                )
            ).tolist())
            for param_name, param in parameters.items() 
            if getattr(param, "is_scaling", False) and param_name != self.data_parameter_name
        }

        scaling_grid = {**scaling_non_data_grid, self.data_parameter_name: [parameters[self.data_parameter_name].domain.upper]}
        
        # Build all configurations and their parameter counts
        n_param_grid = {}
        all_confs_set = set()  # Use set to track unique configs
        all_confs = []
        for indices in itertools.product(*[range(len(grid)) for grid in scaling_grid.values()]):
            conf = {param_name: scaling_grid[param_name][idx] 
                    for param_name, idx in zip(scaling_grid.keys(), indices)}
            conf.update(non_scaling_params)
            
            # Convert to hashable form to check uniqueness
            conf_tuple = tuple(sorted(conf.items()))
            if conf_tuple not in all_confs_set:
                all_confs_set.add(conf_tuple)
                all_confs.append(conf)
            
            n_params = self.params_estimator(**conf)
            conf_key = tuple(sorted(conf.items()))
            n_param_grid[conf_key] = n_params

        # print(f"n_param_grid.values() {n_param_grid.values()}")
        
        # Extract valid parameter values and create bins
        n_param_values = [n for n in n_param_grid.values() if n is not None]
        if not n_param_values:
            logger.warning("No valid parameter estimates found")
            return ([], defaultdict(list))
        
        # Create 5 uniform bins
        min_n, max_n = min(n_param_values), max(n_param_values)
        param_bins = np.linspace(min_n, max_n, num=10)
        threshold = 0.15  # ±10% threshold
        
        # Group configurations into bins with ±10% threshold
        configs_per_bin = defaultdict(list)
        for conf in all_confs:
            n_params = self.params_estimator(**conf)
            # Find closest bin
            bin_idx = int(np.argmin(np.abs(param_bins - n_params)))
            bin_center = param_bins[bin_idx]
            bin_lower = bin_center * (1 - threshold)
            bin_upper = bin_center * (1 + threshold)
            
            if bin_lower <= n_params <= bin_upper:
                configs_per_bin[bin_idx].append(conf)
            
        
        # Get bins with more than 10 configs
        bins_with_enough_configs = [(bin_idx, len(confs)) for bin_idx, confs in configs_per_bin.items() if len(confs) > 5]
        if not bins_with_enough_configs:
            raise Exception("No bins with more than 5 configs found. Using all available bins.")
        
        # Sort by number of configs (descending) and select top 5
        bins_with_enough_configs.sort(key=lambda x: x[1], reverse=True)
        eligible_bin_indices = [bin_idx for bin_idx, _ in bins_with_enough_configs]
        
        logger.info(f"Binned configs into {len(configs_per_bin)} bins. Selected {len(eligible_bin_indices)} evenly-spaced bins: {eligible_bin_indices} with counts: {[len(configs_per_bin[b]) for b in eligible_bin_indices]}")
        
        conf_list = []
        used_flops = 0
        
        # Track selected model parameter configurations
        selected_model_configs = defaultdict(list)
        
        # Iterate through bins until budget exhausted
        for bin_idx in eligible_bin_indices:
            if bin_idx not in configs_per_bin or not configs_per_bin[bin_idx]:
                continue
            
            # Sort configs in this bin by aspect ratio
            bin_configs = configs_per_bin[bin_idx]
            try:
                sorted_configs = sorted(bin_configs, key=lambda c: self.aspect_ratio_fn(**c))
            except Exception as e:
                logger.warning(f"Could not sort configs by aspect ratio in bin {bin_idx}: {e}")
                sorted_configs = bin_configs
            
            n = len(sorted_configs)
            if n < 3:
                continue
            
            # Select 3 strategic indices
            first_idx = 0
            middle_idx = n // 2
            end_idx = n - 1
            
            selected_indices = [
                (first_idx + middle_idx) // 2,  # Between first and middle
                middle_idx,                      # Middle point
                (middle_idx + end_idx) // 2      # Between middle and end
            ]
            
            for idx in selected_indices:
                idx = min(idx, n - 1)  # Ensure within bounds
                conf = sorted_configs[idx]
                
                try:
                    estimated_flops = self.flops_estimator(**conf)
                except Exception as e:
                    logger.warning(f"Could not estimate FLOPs for config: {e}")
                    continue
                
                # Extract and track model config (without data param)
                model_conf = tuple(sorted((k, v) for k, v in conf.items() if k != self.data_parameter_name))
                n_param = self.params_estimator(**conf)
                selected_model_configs[n_param].append(model_conf)

                if used_flops + estimated_flops > self.max_evaluation_flops:
                    logger.info(f"Reached FLOPS budget limit. Used: {used_flops:.2e}/{self.max_evaluation_flops:.2e}")
                    continue
                
                
                
                conf_list.append(conf)
                used_flops += estimated_flops
                logger.debug(f"Added config from bin {bin_idx} (aspect_ratio={self.aspect_ratio_fn(**conf):.3f}). Total FLOPs: {used_flops:.2e}")
        
        # Store selected model parameter configs (without data param)
        logger.info(f"Selected {len(selected_model_configs)} unique model configs for gemstones")
        
        logger.info(f"Generated {len(conf_list)} configs. Total FLOPs used: {used_flops:.2e}/{self.max_evaluation_flops:.2e}")
        return conf_list, selected_model_configs
    

        
    def _get_closest_iso_flop(self, estimated_flops: float, iso_flops: list[float]) -> tuple[float, float]:
        closest = min(iso_flops, key=lambda x: abs(x - estimated_flops))
        return closest, closest / estimated_flops

    def _generate_iso_flop_list(self, num_steps: int = 10) -> list[float]:
        # assume that all the HPs are increasing, the other solution could be to get all possible grids and calc the flops for alll and get min and max there.
        
        min_flops_conf, max_flops_conf = {}, {}
        for param_name, param in self.space.searchables.items():
            min_flops_conf[param_name] = param.domain.lower
        
        for param_name, param in self.space.searchables.items():
            max_flops_conf[param_name] = param.domain.upper
        min_flops = self.flops_estimator(**min_flops_conf)
        max_flops = self.flops_estimator(**max_flops_conf)
        return torch.logspace(math.log10(min_flops), math.log10(max_flops), steps=num_steps).tolist()
    

    def _choose_action_with_sobol(self, action_probabilities: torch.Tensor, iteration_count: int) -> int:
        sobol_sampler = Sampler.sobol(ndim=1, scramble=False)
        
        samples = sobol_sampler.sample(
            iteration_count + 1,
            to=NepsDomain.unit_float()
        )
        u = samples[-1].item()
        
        cumsum = torch.cumsum(action_probabilities, dim=0)
        action = torch.searchsorted(cumsum, torch.tensor(u)).item()
        
        return min(int(action), len(action_probabilities) - 1)
    
    def sample_value_for_parameters(self, param_types:Literal["scaling", "non-scaling", "all"] = "all", centre_if_prior_unavailable: bool=False) -> Mapping[str, Any]:
        import random
        sampled_conf = {}
        for param_name, param in self.space.searchables.items():
            if (param_types == "scaling" and not getattr(param, "is_scaling", False)) or (param_types == "non-scaling" and getattr(param, "is_scaling", False)):
                continue
            if param.prior is not None:
                sampled_conf[param_name] = param.prior
            elif isinstance(param, HPOInteger):
                sampled_conf[param_name] = param.center if centre_if_prior_unavailable else param.domain.cast_one(random.random(), frm=NepsDomain.unit_float())
            elif isinstance(param, HPOFloat):
                sampled_conf[param_name] = param.center if centre_if_prior_unavailable else param.domain.cast_one(random.random(), frm=NepsDomain.unit_float())
            elif isinstance(param, HPOCategorical):
                sampled_conf[param_name] = param.domain.cast_one(random.random(), frm=NepsDomain.unit_float())
            else:
                raise ValueError(f"Unsupported parameter type for {param_name}")
        return sampled_conf

    
    def __call__(self, trials, budget_info=None, n=None):
        to_spend = None
            
        if self.flops_estimator is not None and self.max_evaluation_flops is not None:
            to_spend = self.max_evaluation_flops - sum([self.flops_estimator(**trial.config) for trial in trials.values()])
            print(f"Chinchilla_Guided_Scaling: to spend {to_spend} FLOPs")
            if to_spend <= 0:
                raise ValueError("No remaining FLOPs budget to spend on evaluation.")
            
        sample = self.base_optimizer(trials, budget_info=BudgetInfo(cost_to_spend=to_spend), n=n)
        return sample
    

    def _find_optimal_n_for_flops(
        self,
        all_possible_model_configs: dict,
        trials: Mapping[str, Trial],
        target_flops: float
    ) -> tuple[float, float, float] | None:
        """Find optimal parameter count N for given target FLOP budget using all_possible_model_configs.
        
        Returns:
            Tuple of (optimal_n, slope, intercept) or None if extrapolation fails
        """
        try:
            # Collect (N, FLOPS) pairs from all_possible_model_configs for robust estimation
            ns_list = []
            flops_list = []
            
            for n_val, model_configs in all_possible_model_configs.items():
                if not model_configs:
                    continue
                
                # Use first model config as representative
                model_config_tuple = model_configs[0]
                model_config = dict(model_config_tuple)
                
                # Estimate FLOPS with a representative data parameter value
                data_param = self.space.searchables[self.data_parameter_name]
                data_val_mid = (data_param.domain.lower + data_param.domain.upper) / 2
                model_config[self.data_parameter_name] = data_val_mid
                model_config.update(self.space.constants)
                
                try:
                    estimated_flops = self.flops_estimator(**model_config)
                    ns_list.append(n_val)
                    flops_list.append(estimated_flops)
                except Exception as e:
                    logger.debug(f"Could not estimate FLOPS for N={n_val}: {e}")
                    continue
            
            if len(ns_list) < 2:
                logger.warning("Not enough N/FLOPS pairs for fitting")
                return None
            
            # Fit in log-log space: log(N) = slope * log(FLOPS) + intercept
            fitted_flops_params = self._fit_loglog_line(flops_list, ns_list)
            if fitted_flops_params is None:
                logger.warning("Could not fit FLOPS-to-params line")
                return None
            fitted_ys, slope, intercept = fitted_flops_params
            
            logger.info(f"Extracted slope={slope:.4f}, intercept={intercept:.4f} from all_possible_model_configs (N vs FLOPs in log-log)")
            
            # Extrapolate to find required N for target_flops
            target_n = np.exp(slope * np.log(target_flops) + intercept)
            logger.info(f"For target_flops={target_flops:.2e}, estimated target N={target_n:.2e}")
            
            # Find closest N in all_possible_model_configs
            available_ns = list(all_possible_model_configs.keys())
            if not available_ns:
                logger.warning("No available N values in all_possible_model_configs")
                return None
            
            closest_n = min(available_ns, key=lambda n: abs(n - target_n))
            logger.info(f"Selected nearest available N={closest_n:.2e} (target was {target_n:.2e})")
            
            return float(closest_n), float(slope), float(intercept)
            
        except Exception as e:
            logger.error(f"Error in _find_optimal_n_for_flops: {e}")
            return None

    def _find_optimal_n_using_parametric_loss(
        self,
        all_possible_model_configs: dict,
        target_flops: float
    ) -> tuple[float, float, float] | None:
        """Find optimal parameter count N using scaling law slope beta/(beta+alpha) to fit FLOPS vs params.
        
        Returns:
            Tuple of (optimal_n, slope, intercept) or None if extrapolation fails
        """
        try:
            # Ensure scaling law parameters are fitted
            if self._scaling_law_params is None:
                logger.warning("Scaling law parameters not fitted, cannot use parametric loss method")
                return None
            
            # Compute slope from scaling law: beta / (beta + alpha)
            slope = self._scaling_law_params.beta / (self._scaling_law_params.beta + self._scaling_law_params.alpha)
            logger.info(f"Using parametric loss slope from scaling law: {slope:.4f} = {self._scaling_law_params.beta:.4f} / ({self._scaling_law_params.beta:.4f} + {self._scaling_law_params.alpha:.4f})")
            
            # Collect (N, FLOPS) pairs from all_possible_model_configs
            ns_list = []
            flops_list = []
            
            for n_val, model_configs in all_possible_model_configs.items():
                if not model_configs:
                    continue
                
                # Use first model config as representative
                model_config_tuple = model_configs[0]
                model_config = dict(model_config_tuple)
                
                # Estimate FLOPS with a representative data parameter value
                data_param = self.space.searchables[self.data_parameter_name]
                data_val_mid = (data_param.domain.lower + data_param.domain.upper) / 2
                model_config[self.data_parameter_name] = data_val_mid
                model_config.update(self.space.constants)
                
                try:
                    estimated_flops = self.flops_estimator(**model_config)
                    ns_list.append(n_val)
                    flops_list.append(estimated_flops)
                except Exception as e:
                    logger.debug(f"Could not estimate FLOPS for N={n_val}: {e}")
                    continue
            
            if len(ns_list) < 2:
                logger.warning("Not enough N/FLOPS pairs for fitting")
                return None
            
            # Fit intercept in log-log space: log(N) = slope * log(FLOPS) + intercept
            # Given fixed slope, compute intercept as mean residual
            log_ns = np.log(np.array(ns_list))
            log_flops = np.log(np.array(flops_list))
            
            if not (np.all(np.isfinite(log_ns)) and np.all(np.isfinite(log_flops))):
                logger.warning("Invalid log values in FLOPS/params data")
                return None
            
            # Compute intercept: intercept = mean(log(N) - slope * log(FLOPS))
            intercept = np.mean(log_ns - slope * log_flops)
            logger.info(f"Fitted intercept={intercept:.4f} (fixed slope={slope:.4f})")
            
            # Extrapolate to find required N for target_flops
            target_n = np.exp(slope * np.log(target_flops) + intercept)
            logger.info(f"For target_flops={target_flops:.2e}, estimated target N={target_n:.2e}")
            
            # Find closest N in all_possible_model_configs
            available_ns = list(all_possible_model_configs.keys())
            if not available_ns:
                logger.warning("No available N values in all_possible_model_configs")
                return None
            
            closest_n = min(available_ns, key=lambda n: abs(n - target_n))
            logger.info(f"Selected nearest available N={closest_n:.2e} (target was {target_n:.2e})")
            
            return float(closest_n), float(slope), float(intercept)
            
        except Exception as e:
            logger.error(f"Error in _find_optimal_n_using_parametric_loss: {e}")
            return None

    def _grid_over_data_for_optimal_n(
        self,
        optimal_n: float,
        all_possible_model_configs: dict,
        max_target_flops: float
    ) -> tuple[dict[str, Any], float, float | None, float | None] | None:
        """Grid over data parameter for a given optimal N and find config closest to target FLOPS.
        
        Returns:
            Tuple of (best config dict, predicted loss, slope_loss, intercept_loss) or None if no valid config found
        """
        try:
            # Get model configs for this N (list of tuples)
            model_configs_for_n = all_possible_model_configs[optimal_n]
            if not model_configs_for_n:
                logger.warning(f"No model configs found for N={optimal_n}")
                return None
            
            # Use first model config from the list
            model_config_tuple = model_configs_for_n[0]
            model_config = dict(model_config_tuple)
            logger.info(f"Using model config for N={optimal_n}: {model_config}")
            
            # Create fine-grained grid for data parameter with 1000 points
            data_param = self.space.searchables[self.data_parameter_name]
            data_grid = torch.linspace(0, 1, steps=1000)
            data_grid_values = data_param.domain.cast(data_grid, frm=NepsDomain.unit_float()).tolist()
            
            # Ensure scaling law parameters are fitted
            if self._scaling_law_params is None:
                logger.warning("Scaling law parameters not fitted")
                return None
            
            # Find data value that gets closest to max_target_flops (without exceeding)
            best_config = None
            best_flops_diff = float("inf")
            best_estimated_flops = None
            
            for data_val in data_grid_values:
                conf = model_config.copy()
                conf[self.data_parameter_name] = data_val
                conf.update(self.space.constants)
                
                try:
                    estimated_flops = self.flops_estimator(**conf)
                except Exception as e:
                    logger.debug(f"Could not estimate FLOPs: {e}")
                    continue
                
                # Skip if exceeds budget
                if estimated_flops > max_target_flops:
                    continue
                
                # Find closest to target without exceeding
                flops_diff = max_target_flops - estimated_flops
                if flops_diff < best_flops_diff:
                    best_flops_diff = flops_diff
                    best_config = conf
                    best_estimated_flops = estimated_flops
            
            if best_config is None:
                logger.warning(f"Could not find valid config within FLOP budget {max_target_flops:.2e}")
                return None
            
            # Estimate loss using scaling law for the selected config
            try:
                n = self.params_estimator(**best_config)
                d = self.seen_datapoints_estimator(**best_config)
            except Exception as e:
                logger.debug(f"Could not estimate params/datapoints: {e}")
                return None
            
            best_loss = (
                self._scaling_law_params.E +
                self._scaling_law_params.A / (n ** self._scaling_law_params.alpha) +
                self._scaling_law_params.B / (d ** self._scaling_law_params.beta)
            )
            
            logger.info(f"Selected config for max_target_flops={max_target_flops:.2e} N={optimal_n}: "
                       f"actual_flops={best_estimated_flops:.2e}, loss={best_loss:.6f}")
            
            # Return None for slope/intercept since we're using a single point, not fitting
            return best_config, best_loss, None, None
            
        except Exception as e:
            logger.error(f"Error in _grid_over_data_for_optimal_n: {e}")
            return None

    def extrapolate_by_param_data(
        self, 
        all_possible_model_configs: dict, 
        trials: Mapping[str, Trial], 
        max_target_flops: float
    ) -> tuple[dict[str, Any], float, float, float, float | None, float | None] | None:
        """Extrapolate to max_target_flops by finding optimal N from Pareto front, then grid over D.
        
        Returns:
            Tuple of (best config dict, predicted loss, slope_params, intercept_params, slope_loss, intercept_loss) or None
        """
        try:
            # Find optimal N for the target FLOP budget using Pareto front
            result = self._find_optimal_n_for_flops(all_possible_model_configs, trials, max_target_flops)
            if result is None:
                logger.warning("Could not find optimal N from Pareto front")
                return None
            
            optimal_n, slope_params, intercept_params = result
            
            # Fit scaling law if not already fitted
            if self._scaling_law_params is None:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(Chinchilla_Guided_Scaling.get_power_law_curvature, trials)
                        self._scaling_law_params = future.result(timeout=10)
                except TimeoutError:
                    logger.error("Scaling law fitting timed out")
                    return None
                except Exception as e:
                    logger.warning(f"Could not fit scaling law: {e}")
                    return None
            
            # Grid over data parameter and find best config
            grid_result = self._grid_over_data_for_optimal_n(optimal_n, all_possible_model_configs, max_target_flops)
            if grid_result is None:
                return None
            
            best_config, best_loss, slope_loss, intercept_loss = grid_result
            return best_config, best_loss, slope_params, intercept_params, slope_loss, intercept_loss
            
        except Exception as e:
            logger.error(f"Error in extrapolate_by_param_data: {e}")
            return None

    def extrapolate_by_parametric_loss(
        self, 
        all_possible_model_configs: dict, 
        max_target_flops: float
    ) -> tuple[dict[str, Any], float, float, float, float | None, float | None] | None:
        """Extrapolate to max_target_flops using scaling law parameters with unit slope fitting.
        
        Returns:
            Tuple of (best config dict, predicted loss, slope_params, intercept_params, slope_loss, intercept_loss) or None
        """
        try:
            # Find optimal N using scaling law-based approach
            result = self._find_optimal_n_using_parametric_loss(all_possible_model_configs, max_target_flops)
            if result is None:
                logger.warning("Could not find optimal N using scaling law")
                return None
            
            optimal_n, slope_params, intercept_params = result
            
            # Ensure scaling law parameters are fitted
            if self._scaling_law_params is None:
                logger.warning("Scaling law parameters not fitted")
                return None
            
            # Grid over data parameter and find best config
            grid_result = self._grid_over_data_for_optimal_n(optimal_n, all_possible_model_configs, max_target_flops)
            if grid_result is None:
                return None
            
            best_config, best_loss, slope_loss, intercept_loss = grid_result
            return best_config, best_loss, slope_params, intercept_params, slope_loss, intercept_loss
            
        except Exception as e:
            logger.error(f"Error in extrapolate_by_parametric_loss: {e}")
            return None

    def extrapolate(
        self, 
        trials: Mapping[str, Trial],
        spent_flops: float,
    ) -> dict:
        assert self.all_possible_model_configs is not None, "all_possible_model_configs must be set before calling extrapolate"
        
        logger.info(f"Begin extrapolation for {spent_flops:.2e}")

        target_flops = np.unique(
            np.clip(
                spent_flops * self.target_over_spent_flops_ratios, 
                self.min_flops,
                self.max_flops,
            )
        )
        logger.info(f"Target FLOPs for extrapolation: {target_flops}")
        
        # Collect slope/intercept for each method
        slopes_params_param_data = []
        intercepts_params_param_data = []
        slopes_loss_param_data = []
        intercepts_loss_param_data = []
        
        slopes_params_parametric_loss = []
        intercepts_params_parametric_loss = []
        slopes_loss_parametric_loss = []
        intercepts_loss_parametric_loss = []
        
        extrapolation_per_target_flop = {}
        
        for target_flop in target_flops:
            result_param_data = self.extrapolate_by_param_data(
                self.all_possible_model_configs, trials, target_flop,
            )
            result_parametric_loss = self.extrapolate_by_parametric_loss(
                self.all_possible_model_configs, target_flop,
            )
            
            config_by_param_data = None
            loss_by_param_data = None
            if result_param_data is not None:
                config_by_param_data, loss_by_param_data, s_p, i_p, s_l, i_l = result_param_data
                slopes_params_param_data.append(s_p)
                intercepts_params_param_data.append(i_p)
                slopes_loss_param_data.append(s_l)
                intercepts_loss_param_data.append(i_l)
            
            config_by_parametric_loss = None
            loss_by_parametric_loss = None
            if result_parametric_loss is not None:
                config_by_parametric_loss, loss_by_parametric_loss, s_p, i_p, s_l, i_l = result_parametric_loss
                slopes_params_parametric_loss.append(s_p)
                intercepts_params_parametric_loss.append(i_p)
                slopes_loss_parametric_loss.append(s_l)
                intercepts_loss_parametric_loss.append(i_l)
            
            logger.info(f"Optimal result by param_data: config={config_by_param_data is not None}, loss={loss_by_param_data}")
            logger.info(f"Optimal result by parametric_loss: config={config_by_parametric_loss is not None}, loss={loss_by_parametric_loss}")
            
            extrapolation_per_target_flop[target_flop] = (
                (config_by_param_data, loss_by_param_data) if config_by_param_data is not None else None,
                (config_by_parametric_loss, loss_by_parametric_loss) if config_by_parametric_loss is not None else None,
            )
        
        # Build fits dictionary using collected slope/intercepts
        fits = {'param_data': {}, 'parametric_loss': {}}
        
        if slopes_params_param_data:
            avg_slope_p = np.mean([s for s in slopes_params_param_data if s is not None])
            avg_intercept_p = np.mean([i for i in intercepts_params_param_data if i is not None])
            fits['param_data']['params'] = {
                'slope': avg_slope_p,
                'intercept': avg_intercept_p,
            }
        
        if slopes_loss_param_data:
            avg_slope_l = np.mean([s for s in slopes_loss_param_data if s is not None])
            avg_intercept_l = np.mean([i for i in intercepts_loss_param_data if i is not None])
            fits['param_data']['loss'] = {
                'slope': avg_slope_l,
                'intercept': avg_intercept_l,
            }
        
        if slopes_params_parametric_loss:
            avg_slope_p = np.mean([s for s in slopes_params_parametric_loss if s is not None])
            avg_intercept_p = np.mean([i for i in intercepts_params_parametric_loss if i is not None])
            fits['parametric_loss']['params'] = {
                'slope': avg_slope_p,
                'intercept': avg_intercept_p,
            }
        
        if slopes_loss_parametric_loss:
            avg_slope_l = np.mean([s for s in slopes_loss_parametric_loss if s is not None])
            avg_intercept_l = np.mean([i for i in intercepts_loss_parametric_loss if i is not None])
            fits['parametric_loss']['loss'] = {
                'slope': avg_slope_l,
                'intercept': avg_intercept_l,
            }
        
        return {
            'configs': extrapolation_per_target_flop,
            'fits': fits,
        }

    @classmethod
    def get_power_law_curvature(cls, trials: Mapping[str, Trial]) -> ScalingLawParameters:
        # L = E + A / N^alpha + B / D^beta
        # where A = e^a and B = e^b

        def huber(residual, delta=1.0):
            abs_r = np.abs(residual)
            return np.where(
                abs_r <= delta,
                0.5 * residual**2,
                delta * (abs_r - 0.5 * delta),
            )

        def objective(theta, N, D, L):
            E, a, b, alpha, beta = theta
            
            # Clip a and b to prevent overflow (exp(20) ≈ 4.85e8)
            a_clipped = np.clip(a, -10, 10)
            b_clipped = np.clip(b, -10, 10)
            A = np.exp(a_clipped)
            B = np.exp(b_clipped)
            
            # Compute with numerical stability
            with np.errstate(divide='ignore', invalid='ignore'):
                N_alpha = np.power(N, alpha, where=(N > 0), out=np.ones_like(N, dtype=np.float64))
                D_beta = np.power(D, beta, where=(D > 0), out=np.ones_like(D, dtype=np.float64))
                
                # Avoid division by zero
                N_alpha = np.where(N_alpha > 0, N_alpha, 1.0)
                D_beta = np.where(D_beta > 0, D_beta, 1.0)
                
                L_hat = E + A / N_alpha + B / D_beta
            
            # Check for NaN or Inf
            if np.any(~np.isfinite(L_hat)):
                return 1e10  # Large penalty for invalid values
            
            residual = L_hat - L
            loss = np.sum(huber(residual))
            
            if not np.isfinite(loss):
                return 1e10
                
            return loss


        l_list, n_list, d_list = [], [], []
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize) or isinstance(trial.report.objective_to_minimize, Sequence):
                continue
            if trial.report.extra is None:
                logger.warning(f"Trial {trial.id} has no extra data. Skipping.")
                continue
            try:
                n_list.append(trial.report.extra[cls.N_PARAM_KEY])
                d_list.append(trial.report.extra[cls.N_DATA_KEY])
                l_list.append(float(trial.report.objective_to_minimize))
            except KeyError as e:
                logger.error(f"Trial {trial.id} missing required key {e} in extra. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Skipping trial {trial.id}: {e}")
                continue

        if len(l_list) < 5:
            raise ValueError("Not enough data points to fit scaling law.")

        N = np.array(n_list)
        D = np.array(d_list)
        L = np.array(l_list)

        # Adaptive initialization grid based on data size
        # For small datasets, use fewer initializations
        num_points = len(l_list)
        if num_points <= 10:
            # For 5-10 points: use minimal grid (8-20 initializations)
            initialization = [
                (0, 5, 5, 0.5, 0.5),
                (-1, 0, 0, 0.5, 0.5),
                (1, 10, 10, 1.0, 1.0),
                (0.5, 5, 10, 0.5, 1.0),
                (0, 10, 5, 1.0, 0.5),
                (-0.5, 0, 10, 0.5, 1.0),
                (0, 0, 0, 0.5, 0.5),
                (0.5, 5, 5, 1.0, 1.0),
                (np.log(np.mean(L)), 5, 5, 0.5, 0.5),  # Data-informed E
            ]
        else:
            # For 11+ points: use moderate grid (2880 -> 200 initializations)
            initialization = list(itertools.product(
                [-1, -0.5, 0, 0.5, 1],        # E (5)
                [0, 5, 10, 15, 20, 25],       # a (6)
                [0, 5, 10, 15, 20, 25],       # b (6)
                [0, 0.5, 1.0, 1.5, 2.0],      # alpha (4)
                [0, 0.5, 1.0, 1.5, 2.0],      # beta (4)
            ))

        best_result = None
        best_loss = float("inf")
        early_stop_threshold = best_loss * 0.1  # Stop if loss is 10x better than initial

        for idx, theta0 in enumerate(initialization):
            # Early stopping: after finding a good solution, try only a few more
            if idx > 20 and best_loss < early_stop_threshold:
                logger.info(f"Early stopping at iteration {idx}: found good solution with loss={best_loss:.6f}")
                break

            try:
                # Bounds to prevent numerical issues
                # E: unbounded (can be negative or positive)
                # a, b: [0, 10] to prevent exp overflow
                # alpha, beta: [0.01, 1.0] for reasonable power law exponents
                bounds = [
                    (None, None),      # E
                    (0, 10),         # a
                    (0, 10),         # b
                    (0.01, 1.0),       # alpha
                    (0.01, 1.0),       # beta
                ]
                
                res = minimize(
                    objective,
                    theta0,
                    args=(N, D, L),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception as e:
                raise e

        if best_result is None:
            raise ValueError("Could not fit scaling law with any initialization.")

        E, a, b, alpha, beta = best_result.x
        A = np.exp(np.clip(a, -10, 10))
        B = np.exp(np.clip(b, -10, 10))
        return ScalingLawParameters(E=E, A=A, B=B, alpha=alpha, beta=beta)
    
    @classmethod
    def _extract_scaling_metrics(cls, trials: Mapping[str, Trial]) -> list[tuple] | None:
        """Extract (flops, n_params, n_data, loss) from trials."""
        rows = []
        for trial in trials.values():
            if (trial.report is None or trial.report.objective_to_minimize is None or 
                not np.isfinite(trial.report.objective_to_minimize) or 
                isinstance(trial.report.objective_to_minimize, Sequence)):
                continue
            if trial.report.extra is None:
                continue
            
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
                n_data = trial.report.extra[cls.N_DATA_KEY]
                loss = float(trial.report.objective_to_minimize)
                
                if n_params > 0:
                    rows.append((flops, n_params, n_data, loss))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Trial extraction failed: {e}")
                continue
        
        return rows if rows else None

    @staticmethod
    def _fit_loglog_line(xs: list, ys: list) -> tuple[list, float, float] | None:
        """Fit line in log-log space and return fitted values, slope, and intercept."""
        try:
            log_xs = np.log(np.array(xs))
            log_ys = np.log(np.array(ys))
            
            if not (np.all(np.isfinite(log_xs)) and np.all(np.isfinite(log_ys))):
                return None
            
            coeffs = np.polyfit(log_xs, log_ys, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            if not np.isfinite(slope):
                return None
            
            fit_line = np.poly1d(coeffs)
            fitted_log_ys = fit_line(log_xs)
            fitted_ys = np.exp(fitted_log_ys)
            
            return fitted_ys.tolist(), slope, intercept
        except Exception as e:
            logger.debug(f"Could not fit loglog line: {e}")
            return None

    def _create_scaling_plot(self, xs: list, ys: list, xlabel: str, ylabel: str, title: str) -> Figure:
        """Create a single scaling relationship plot with optional fitted line."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(xs, ys, marker='o', alpha=0.9)
        
        # Try to fit a line
        fit_result = self._fit_loglog_line(xs, ys)
        if fit_result is not None:
            fitted_ys, slope, intercept = fit_result
            ax.plot(xs, fitted_ys, 'r--', linewidth=2, alpha=0.8, label=f'Fit (slope={slope:.3f})')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        return fig

    def _plot_flop_vs_param_and_data(self, trials: Mapping[str, Trial]) -> tuple[Figure, Figure] | None:
        """Generate FLOPs vs Params and FLOPs vs Data plots."""
        
        use_lc_aware = self.strategy in ["approach1", "gemstones"]
        use_pareto_fit = self.strategy in ["approach1", "approach2"]
        fit_func = fit_pareto_front_loglog if use_pareto_fit else fit_convex_hull_loglog
        result = fit_func(trials, lc_aware=use_lc_aware)
        pareto_rows = result[5]
        
        xs = [r[3] for r in pareto_rows]  # flops
        ys_params = [float(r[0]) for r in pareto_rows]  # n_params
        ys_data = [float(r[1]) for r in pareto_rows]  # n_data
        
        fig1 = self._create_scaling_plot(xs, ys_params, "Flops", "Params", "FLOPs vs Params")
        fig2 = self._create_scaling_plot(xs, ys_data, "Flops", "Data Points", "FLOPs vs Data")
        
        return fig1, fig2


    def _gaussian_smooth_losses(self, flop_loss_tuples: list[tuple[float, float]], sigma: float = 1.0) -> list[tuple[float, float]]:
        """Apply Gaussian smoothing to loss values based on FLOP ordering.
        
        Args:
            flop_loss_tuples: List of (flops, loss) tuples
            sigma: Gaussian kernel standard deviation (larger = more smoothing)
        
        Returns:
            List of (flops, smoothed_loss) tuples, sorted by flops
        """
        if len(flop_loss_tuples) < 3:
            return sorted(flop_loss_tuples, key=lambda x: x[0])
        
        from scipy.ndimage import gaussian_filter1d
        
        # Sort by flops
        sorted_tuples = sorted(flop_loss_tuples, key=lambda x: x[0])
        flops = np.array([f for f, _ in sorted_tuples])
        losses = np.array([l for _, l in sorted_tuples])
        
        # Apply Gaussian smoothing along the FLOP axis
        smoothed_losses = gaussian_filter1d(losses, sigma=sigma)
        
        return list(zip(flops.tolist(), smoothed_losses.tolist()))

    @classmethod
    def _extract_trial_data(cls, trials: Mapping[str, Trial]) -> tuple[list, dict, dict] | None:
        """Extract and validate trial data for plotting."""
        rows = []
        trial_learning_curves = {}
        
        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None or not np.isfinite(trial.report.objective_to_minimize):
                continue
            if trial.report.extra is None:
                continue
            
            try:
                flops = trial.report.extra[cls.FLOPS_KEY]
                n_params = trial.report.extra[cls.N_PARAM_KEY]
                obj = trial.report.objective_to_minimize
                rows.append((n_params, obj, flops))
                
                if trial.report.learning_curve is not None and len(trial.report.learning_curve) > 0:
                    trial_learning_curves[trial.id] = (n_params, trial.report.learning_curve)
            except (KeyError, Exception) as e:
                logger.debug(f"Trial {trial.id}: {e}")
                continue
        
        return (rows, trial_learning_curves, {}) if rows else None

    @staticmethod
    def _extract_flop_loss_from_lc(learning_curve) -> list[tuple[float, float]]:
        """Extract (flops, loss) pairs from learning curve."""
        pairs = []
        for point in learning_curve:
            if isinstance(point, (tuple, list)) and len(point) >= 3:
                try:
                    flops_val = float(point[1])
                    loss_val = float(point[-1])
                    pairs.append((flops_val, loss_val))
                except (ValueError, TypeError):
                    continue
        return sorted(pairs, key=lambda x: x[0]) if pairs else []

    def _plot_flops_per_objective_with_lc(self, trials: Mapping[str, Trial]) -> Figure | None:
        """Generate training loss vs FLOPs visualization with learning curve overlay."""
        
        # Extract trial data
        data = self._extract_trial_data(trials)
        if data is None:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None
        
        rows, trial_learning_curves, _ = data
        
        # Validate multi-objective
        first_obj = rows[0][1]
        if isinstance(first_obj, Sequence) and not isinstance(first_obj, (str, bytes)):
            raise NotImplementedError("Multi-objective Scaling law not implemented yet.")
        
        # Get sorted parameter counts for coloring
        sorted_param_counts = sorted(set(n_params for n_params, _, _ in rows))
        
        # Fit Pareto front
        use_lc_aware = self.strategy in ["approach1", "gemstones"]
        use_pareto_fit = self.strategy in ["approach1", "approach2"]
        fit_func = fit_pareto_front_loglog if use_pareto_fit else fit_convex_hull_loglog
        slope, intercept, fitted_objs, flops_front, objs_front, *_ = fit_func(trials, lc_aware=use_lc_aware)
        
        # Create plot
        fig = self._create_lc_plot(sorted_param_counts, trial_learning_curves, 
                                       slope, flops_front, objs_front, fitted_objs)
        return fig

    def _create_lc_plot(self, sorted_param_counts, trial_learning_curves, 
                        slope, flops_front, objs_front, fitted_objs) -> Figure:
        """Create and configure the learning curve plot."""
        from matplotlib.colors import Normalize
        
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = Normalize(vmin=min(sorted_param_counts), vmax=max(sorted_param_counts))
        cmap = plt.colormaps['viridis']
        
        # Plot learning curves by parameter count
        for n_params in sorted_param_counts:
            color = cmap(norm(n_params))
            
            for trial_id, (tp, lc) in trial_learning_curves.items():
                if tp != n_params:
                    continue
                
                try:
                    pairs = self._extract_flop_loss_from_lc(lc)
                    if pairs:
                        xs, ys = zip(*pairs)
                        ax.plot(xs, ys, linewidth=1, alpha=0.15, color=color, zorder=1)
                except Exception as e:
                    logger.debug(f"Could not plot learning curve for trial {trial_id}: {e}")
        
        # Overlay Pareto front
        if slope is not None and fitted_objs is not None:
            if self.strategy in ["gemstones"]:
                ax.plot(flops_front, objs_front, 'b--', linewidth=2.5, alpha=0.9, 
                    label='Lower Convex Hull', zorder=3)
                ax.scatter(flops_front, objs_front, c='black', s=100, marker='x', alpha=0.9, zorder=4)
            else:
                ax.plot(flops_front, fitted_objs, 'g--', linewidth=2.5, alpha=0.9, 
                    label=f'Pareto fit (slope={slope:.3f})', zorder=3)
        
        # Configure axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOPs", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.set_title("Training Loss vs FLOPs with Pareto Front (LC-Aware)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Parameters")
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def _plot_extrapolation_horizon(self, trials: Mapping[str, Trial], extrapolation_horizon: dict) -> Figure | None:
        use_lc_aware = self.strategy in ["approach1", "gemstones"]
        use_pareto_fit = self.strategy in ["approach1", "approach2"]
        fit_func = fit_pareto_front_loglog if use_pareto_fit else fit_convex_hull_loglog
        result = fit_func(trials, lc_aware=use_lc_aware)
        pareto_rows = result[5]
        
        if not pareto_rows:
            logger.warning("No evaluated trials with flops/params to plot")
            return None
        
        actual_flops = [r[3] for r in pareto_rows]
        actual_params = [r[0] for r in pareto_rows]
        actual_loss = [r[2] for r in pareto_rows]
        
        # Extract configs and fits from extrapolation_horizon
        configs = extrapolation_horizon.get('configs', {})
        fits = extrapolation_horizon.get('fits', {})
        
        # Extract predicted points from extrapolation results
        predicted_flops_param_data = []
        predicted_params_param_data = []
        predicted_loss_param_data = []
        
        predicted_flops_parametric_loss = []
        predicted_params_parametric_loss = []
        predicted_loss_parametric_loss = []
        
        for target_flop, (result_by_param_data, result_by_parametric_loss) in configs.items():
            if result_by_param_data is not None:
                config_by_param_data, loss_by_param_data = result_by_param_data
                try:
                    flops = self.flops_estimator(**config_by_param_data)
                    params = self.params_estimator(**config_by_param_data)
                    predicted_flops_param_data.append(flops)
                    predicted_params_param_data.append(params)
                    predicted_loss_param_data.append(loss_by_param_data)
                except Exception as e:
                    logger.debug(f"Could not extract predicted values for param_data: {e}")
            
            if result_by_parametric_loss is not None:
                config_by_parametric_loss, loss_by_parametric_loss = result_by_parametric_loss
                try:
                    flops = self.flops_estimator(**config_by_parametric_loss)
                    params = self.params_estimator(**config_by_parametric_loss)
                    predicted_flops_parametric_loss.append(flops)
                    predicted_params_parametric_loss.append(params)
                    predicted_loss_parametric_loss.append(loss_by_parametric_loss)
                except Exception as e:
                    logger.debug(f"Could not extract predicted values for parametric_loss: {e}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: FLOPs vs Params (loglog)
        ax1 = axes[0]
        ax1.scatter(actual_flops, actual_params, c='blue', marker='o', s=50, alpha=0.6, label='Evaluated trials')
        
        # Plot predicted points
        if predicted_flops_param_data:
            ax1.scatter(predicted_flops_param_data, predicted_params_param_data, c='green', marker='^', s=80, 
                       alpha=0.8, edgecolors='darkgreen', linewidth=1.5, label='Predicted (param_data)', zorder=3)
        
        if predicted_flops_parametric_loss:
            ax1.scatter(predicted_flops_parametric_loss, predicted_params_parametric_loss, c='red', marker='s', s=80, 
                       alpha=0.8, edgecolors='darkred', linewidth=1.5, label='Predicted (parametric_loss)', zorder=3)
        
        # Generate fitted lines from slope/intercept for each method
        min_flops = min(actual_flops) if actual_flops else 1e6
        max_flops = max(actual_flops) if actual_flops else 1e9
        flops_range_log = np.linspace(np.log(min_flops), np.log(max_flops), 100)
        flops_range = np.exp(flops_range_log)
        
        if fits.get('param_data', {}).get('params'):
            fit_data = fits['param_data']['params']
            slope = fit_data['slope']
            intercept = fit_data['intercept']
            # Generate fitted values: log(params) = slope * log(flops) + intercept
            fitted_log_params = slope * np.log(flops_range) + intercept
            fitted_params = np.exp(fitted_log_params)
            ax1.plot(flops_range, fitted_params, 'g-', linewidth=2.5, alpha=0.8, 
                    label=f'Fit param_data (slope={slope:.3f})')
        
        if fits.get('parametric_loss', {}).get('params'):
            fit_data = fits['parametric_loss']['params']
            slope = fit_data['slope']
            intercept = fit_data['intercept']
            # Generate fitted values: log(params) = slope * log(flops) + intercept
            fitted_log_params = slope * np.log(flops_range) + intercept
            fitted_params = np.exp(fitted_log_params)
            ax1.plot(flops_range, fitted_params, 'r-', linewidth=2.5, alpha=0.8, 
                    label=f'Fit parametric_loss (slope={slope:.3f})')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('FLOPs', fontsize=12)
        ax1.set_ylabel('Parameters', fontsize=12)
        ax1.set_title('Extrapolation Horizon: FLOPs vs Parameters', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        
        # Plot 2: FLOPs vs Loss (loglog)
        ax2 = axes[1]
        ax2.scatter(actual_flops, actual_loss, c='blue', marker='o', s=50, alpha=0.6, label='Evaluated trials')
        
        # Plot predicted points
        if predicted_flops_param_data:
            ax2.scatter(predicted_flops_param_data, predicted_loss_param_data, c='green', marker='^', s=80, 
                       alpha=0.8, edgecolors='darkgreen', linewidth=1.5, label='Predicted (param_data)', zorder=3)
        
        if predicted_flops_parametric_loss:
            ax2.scatter(predicted_flops_parametric_loss, predicted_loss_parametric_loss, c='red', marker='s', s=80, 
                       alpha=0.8, edgecolors='darkred', linewidth=1.5, label='Predicted (parametric_loss)', zorder=3)
        
        # Generate fitted lines from slope/intercept for each method
        if fits.get('param_data', {}).get('loss'):
            fit_data = fits['param_data']['loss']
            slope = fit_data['slope']
            intercept = fit_data['intercept']
            # Generate fitted values: log(loss) = slope * log(flops) + intercept
            fitted_log_loss = slope * np.log(flops_range) + intercept
            fitted_loss = np.exp(fitted_log_loss)
            ax2.plot(flops_range, fitted_loss, 'g-', linewidth=2.5, alpha=0.8, 
                    label=f'Fit param_data (slope={slope:.3f})')
        
        if fits.get('parametric_loss', {}).get('loss'):
            fit_data = fits['parametric_loss']['loss']
            slope = fit_data['slope']
            intercept = fit_data['intercept']
            # Generate fitted values: log(loss) = slope * log(flops) + intercept
            fitted_log_loss = slope * np.log(flops_range) + intercept
            fitted_loss = np.exp(fitted_log_loss)
            ax2.plot(flops_range, fitted_loss, 'r-', linewidth=2.5, alpha=0.8, 
                    label=f'Fit parametric_loss (slope={slope:.3f})')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('FLOPs', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Extrapolation Horizon: FLOPs vs Loss', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def _serialize_extrapolation_results(self, spent_flops: float, extrapolation_horizon: dict) -> dict[str, Any]:
        """Serialize extrapolation results to JSON-compatible format."""
        configs = extrapolation_horizon.get('configs', {})
        fits = extrapolation_horizon.get('fits', {})
        serialized_horizon = {}
        
        for target_flop, (result_by_param_data, result_by_parametric_loss) in configs.items():
            serialized_configs = {}
            
            if result_by_param_data is not None:
                config_by_param_data, loss_by_param_data = result_by_param_data
                try:
                    flops = float(self.flops_estimator(**config_by_param_data))
                    params = float(self.params_estimator(**config_by_param_data))
                    serialized_configs['param_data'] = {
                        'target_flop': float(target_flop),
                        'flops': flops,
                        'params': params,
                        'predicted_loss': float(loss_by_param_data),
                        'config': dict(config_by_param_data),
                    }
                except Exception as e:
                    logger.debug(f"Could not serialize param_data config: {e}")
                    serialized_configs['param_data'] = None
            
            if result_by_parametric_loss is not None:
                config_by_parametric_loss, loss_by_parametric_loss = result_by_parametric_loss
                try:
                    flops = float(self.flops_estimator(**config_by_parametric_loss))
                    params = float(self.params_estimator(**config_by_parametric_loss))
                    serialized_configs['parametric_loss'] = {
                        'target_flop': float(target_flop),
                        'flops': flops,
                        'params': params,
                        'predicted_loss': float(loss_by_parametric_loss),
                        'config': dict(config_by_parametric_loss),
                    }
                except Exception as e:
                    logger.debug(f"Could not serialize parametric_loss config: {e}")
                    serialized_configs['parametric_loss'] = None
            
            serialized_horizon[float(target_flop)] = serialized_configs
        
        return {
            'spent_flops': float(spent_flops),
            'max_evaluation_flops': float(self.max_evaluation_flops),
            'configs': serialized_horizon,
            'fits': fits
        }
    
    def get_trial_artifacts(self, trials: Mapping[str, Trial]) -> list[Artifact]:
        """Consolidates all artifacts: shared scaling law visualization plots from the base class
        plus Chinchilla-specific scaling trend plot showing fitted scaling law parameters.

        Args:
            trials: Mapping of trial IDs to Trial objects. Required to generate plots.

        Returns:
            List of Artifact objects combining parent class plots and custom scaling trend plot.
        """
        if trials is None or len(trials) < 5:
            return []
        # Get base class artifacts (class method needs to be called on the class)
        artifacts = super().get_trial_artifacts(trials) or []

        spent_flops = sum([self.flops_estimator(**trial.config) for trial in trials.values()])
        
        extrapolation_horizon = None
        if self.all_possible_model_configs is not None:
            try:
                extrapolation_horizon = self.extrapolate(trials, spent_flops)
                if extrapolation_horizon is not None:
                    # Add extrapolation plot
                    try:
                        fig_extrapolation = self._plot_extrapolation_horizon(trials, extrapolation_horizon)
                        if fig_extrapolation is not None:
                            artifacts.append(
                                # Artifact(f"extrapolation_horizon_plot{spent_flops}", fig_extrapolation, ArtifactType.FIGURE)
                                Artifact(f"extrapolation_horizon_plot", fig_extrapolation, ArtifactType.FIGURE)
                            )
                    except Exception as e:
                        logger.warning(f"Failed to generate extrapolation horizon plot: {e}")
                    
                    # Add extrapolation JSON artifact
                    try:
                        extrapolation_data = self._serialize_extrapolation_results(spent_flops, extrapolation_horizon)
                        from subs.neps.utils.files import JsonWriter
                        # Store as JSON data in artifact metadata
                        artifacts.append(
                            # Artifact(f"extrapolation_results{spent_flops}", extrapolation_data, ArtifactType.JSON)
                            Artifact(f"extrapolation_results", extrapolation_data, ArtifactType.JSON)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to serialize extrapolation results: {e}")
            except Exception as e:
                logger.warning(f"Failed to perform extrapolation: {e}")

        try:
            figs = self._plot_flop_vs_param_and_data(trials)
            if figs is not None:
                fig_flops_params, fig_flops_data = figs
                artifacts.append(
                    Artifact("flops_vs_params", fig_flops_params, ArtifactType.FIGURE)
                )
                artifacts.append(
                    Artifact("flops_vs_data", fig_flops_data, ArtifactType.FIGURE)
                )
        except Exception as e:
            logger.warning(f"Failed to generate FLOPs vs Params/Data plots: {e}")
        
        if self.strategy in ["approach1", "gemstones"]:
            try:
                fig_lc = self._plot_flops_per_objective_with_lc(trials)
                if fig_lc is not None:
                    artifacts.append(
                        Artifact("flops_vs_loss_with_lc", fig_lc, ArtifactType.FIGURE)
                    )
            except Exception as e:
                logger.warning(f"Failed to generate FLOPs vs Loss plot: {e}")
                raise e
        return artifacts
    
    def import_trials(
        self,
        external_evaluations: Sequence[tuple[Mapping[str, Any], UserResultDict]],
        trials: Mapping[str, Trial],
    ) -> list[ImportedConfig]:
        return self.base_optimizer.import_trials(
            external_evaluations=external_evaluations,
            trials=trials,
        )

def chinchilla_guided_scaling(
    data_parameter_name,
    params_estimator: Callable[..., int],
    seen_datapoints_estimator: Callable[..., int],
    flops_estimator: Callable[..., int],
    aspect_ratio_fn: Callable[..., float],
    evaluate_pipeline: Callable[..., Any],
    max_evaluation_flops: int,
    strategy: Literal["approach1", "approach2", "gemstones"] = "gemstones",
) -> CustomOptimizer:
        
    return custom(name="chinchilla", optimizer=lambda space: Chinchilla_Guided_Scaling(
        space=convert_neps_to_classic_search_space(space=space), # type: ignore
        flops_estimator=flops_estimator,
        params_estimator=params_estimator,
        seen_datapoints_estimator=seen_datapoints_estimator,
        aspect_ratio_fn=aspect_ratio_fn,
        max_evaluation_flops=max_evaluation_flops,
        evaluate_pipeline=evaluate_pipeline,
        data_parameter_name=data_parameter_name,
        strategy=strategy,
    ))
