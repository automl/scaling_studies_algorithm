from __future__ import annotations

import math
import torch
import itertools
from dataclasses import dataclass
import logging
from typing import (
    TYPE_CHECKING,
    Any, Literal
)
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.baselines.utils.utils import _create_scaling_plot
from src.benchmarks.configs import BenchmarkEnum
from neps.state import BudgetInfo, Trial
from neps.optimizers.optimizer import Artifact, ArtifactType
from neps.optimizers.scaling_law_guided import ScalingLawGuidedOptimizer, fit_convex_hull_loglog, fit_pareto_front_loglog, _fit_loglog_line
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


DATA_MULTIPLIERS_PER_BENCHMARK = {
    BenchmarkEnum.TABPFN: [10 ** (k / 4) for k in range(25)],  # 1 ... 1e6, quarter-decade
    BenchmarkEnum.LLM: [5, 10, 20, 80],
    BenchmarkEnum.OPENCLIP: [0.15, 0.3, 0.5, 4, 16, 20],
}

ISO_FLOP_STEPS_PER_BENCHMARK = {BenchmarkEnum.TABPFN: 20}
DEFAULT_ISO_FLOP_STEPS = 10


@dataclass
class ScalingLawParameters:
    E: float
    A: float
    B: float
    alpha: float
    beta: float

class Chinchilla_Guided_Scaling(ScalingLawGuidedOptimizer, AskFunction):
    # Class attributes
    target_over_spent_flops_ratios = np.array([0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 3.0])
    spent_over_max_flops_ratios = [(0.015, 0.005), (0.09, 0.11), (0.45, 0.55), (0.95, 1.05), (1.45, 1.55), (1.95, 2.05)]

    def __init__(self,
                space,
                flops_estimator,
                params_estimator,
                seen_datapoints_estimator,
                aspect_ratio_fn,
                max_evaluation_flops,
                evaluate_pipeline,
                data_parameter_name,
                max_flops,
                strategy: Literal["approach1", "approach2", "gemstones"] = "gemstones",
                use_priors: bool = False,
                benchmark_type: BenchmarkEnum = None
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
        self.use_priors = use_priors        
        self.min_flops, self.max_flops = self.find_flops_bounds()
        self.max_flops = max_flops # test: override with provided max_flops
        self.max_evaluation_flops = max_evaluation_flops
        self._artifacts: list[Artifact] = []
        self.benchmark_type = benchmark_type

        match self.strategy:
            case "approach1":
                config_list, selected_model_configs = self.approach1_generate_config_list()
            case "approach2":
                config_list, selected_model_configs = self.approach2_generate_config_list(num_candidate_per_iso_flop=10)
            case "gemstones":
                config_list, selected_model_configs = self.gemstones_generate_config_list()

        self.all_possible_model_configs = selected_model_configs
        self.config_list = config_list
        print(config_list)
        self.base_optimizer = GridSearch( #type: ignore
            configs_list=config_list,
        )
        estimated_flops = sum([self.flops_estimator(**config) for config in config_list])
        print(f" estimated_flops {estimated_flops} evaluations {self.max_evaluation_flops} max_flops {self.max_flops}")
    
    def get_all_possible_model_configs(self, non_scaling_params) -> defaultdict[int, list[tuple]]:
        parameters = self.space.searchables
        data_param_lower = parameters[self.data_parameter_name].domain.lower
        model_params_grid = {
            param_name: sorted(torch.unique(
                param.domain.cast(
                    torch.linspace(0, 1, steps=30), frm=NepsDomain.unit_float()
                )
            ).tolist())
            for param_name, param in parameters.items() 
            if getattr(param, "is_scaling", False) and param_name != self.data_parameter_name
        }
        attempted_model_configs = defaultdict(list)
        for indices in itertools.product(*[range(len(grid)) for grid in model_params_grid.values()]):
            conf1 = {param_name: model_params_grid[param_name][idx] 
                    for param_name, idx in zip(model_params_grid.keys(), indices)}
            conf1.update({**non_scaling_params ,self.data_parameter_name: data_param_lower})
            model_key = tuple(sorted((k, v) for k, v in conf1.items() if k != self.data_parameter_name))
            n_params_updated = self.params_estimator(**conf1)
            attempted_model_configs[n_params_updated].append(model_key)
        return attempted_model_configs
        
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
                    torch.linspace(0, 1, steps=30), frm=NepsDomain.unit_float()
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
        try:
            data_multipliers = DATA_MULTIPLIERS_PER_BENCHMARK[self.benchmark_type]
        except KeyError:
            raise ValueError(
                f"approach1 has no data_multipliers for benchmark {self.benchmark_type!r}; "
                f"add one to DATA_MULTIPLIERS_PER_BENCHMARK"
            ) from None

        config_list = []
        
        for multiplier in data_multipliers:
            data_param_value = int(n_params * multiplier)
            if data_param_value < data_param.domain.lower or data_param_value > data_param.domain.upper:
                continue
            conf = {**conf1_scaling, **non_scaling_params, self.data_parameter_name: data_param_value}
            if self.min_flops <= self.flops_estimator(**conf) <= self.max_flops:
                config_list.append(conf)
        
        
        attempted_model_configs = defaultdict(list)  
        # Add initial config
        if len(config_list) > 0:
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
                
            current_config = config_list[-1].copy() if len(config_list) > 0 else base_config.copy()
            current_config[param_name] = model_params_grid[param_name][next_idx]
            
            model_key = tuple(sorted((k, v) for k, v in current_config.items() if k != self.data_parameter_name))
            
            # Recompute dynamic data grid for updated scaling parameters
            n_params_updated = self.params_estimator(**current_config)
            attempted_model_configs[n_params_updated].append(model_key)
            data_param = parameters[self.data_parameter_name]
            
            # Find data_param values for each multiplier using the updated N
            updated_data_grid = []
            for multiplier in data_multipliers:
                data_param_value = int(n_params_updated * multiplier)
                if data_param_value < data_param.domain.lower or data_param_value > data_param.domain.upper:
                    continue
                # data_param_value = self._find_data_param_value_for_ratio(current_config, multiplier, data_param)
                updated_data_grid.append(data_param_value)
            
            updated_data_grid = sorted(list(set(updated_data_grid)))  # Remove duplicates and sort
            
            for data_grid_value in updated_data_grid:
                current_config[self.data_parameter_name] = data_grid_value
                copy_current = current_config.copy()
                estimated_flops = self.flops_estimator(**copy_current)
                if (used_flops + estimated_flops > self.max_evaluation_flops) or (estimated_flops > self.max_flops) or (estimated_flops < self.min_flops):
                    logger.info(f"Cannot add config {copy_current} due to FLOPs constraint. "
                                f"Used: {used_flops:.2e}, Estimated: {estimated_flops:.2e}, Max: {self.max_evaluation_flops:.2e}")
                    continue
                config_list.append(copy_current)
                used_flops += estimated_flops
                logger.info(f"Added config {copy_current}. Total used FLOPs: {used_flops:.2e}")
            
            current_grid_indices[param_name] = next_idx
        
        # model params + non scaling params (exclude data param)
        logger.info(f"Selected {len(config_list)} unique model configs for approach1")
        attempted_model_configs = self.get_all_possible_model_configs(non_scaling_params)
        return config_list, attempted_model_configs
    
    def approach2_generate_config_list(self, num_candidate_per_iso_flop: int = 5) -> tuple[list[dict[str, Any]], defaultdict[int, list[tuple]]]:

        iso_flops = self._generate_iso_flop_list(
            num_steps=ISO_FLOP_STEPS_PER_BENCHMARK.get(self.benchmark_type, DEFAULT_ISO_FLOP_STEPS)
        )
        logger.info(f"Generated {len(iso_flops)} ISO-FLOP values: {[f'{f:.2e}' for f in iso_flops]}... (total: {sum(iso_flops):.2e})")
        

        parameters = self.space.searchables
        non_scaling_params = self.sample_value_for_parameters(param_types="non-scaling")
        
        scaling_non_data_grid = {
            param_name: torch.unique(
                param.domain.cast(
                    torch.linspace(0, 1, steps=30), frm=NepsDomain.unit_float()
                )
            ).tolist() 
            for param_name, param in parameters.items() 
            if getattr(param, "is_scaling", False) and param_name != self.data_parameter_name
        }
        
        data_grid = sorted(torch.unique(
            parameters[self.data_parameter_name].domain.cast(
                torch.linspace(0, 1, steps=100000), frm=NepsDomain.unit_float()
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
        
        selected_model_configs = defaultdict(list)
        initial_non_data_conf = {**conf_scaling, **non_scaling_params}
        initial_model_key = tuple(sorted((k, v) for k, v in initial_non_data_conf.items()))
        initial_param_count = self.params_estimator(**initial_non_data_conf | {self.data_parameter_name: data_grid[0]})
        selected_model_configs[initial_param_count].append(initial_model_key)
        
        while used_flops < self.max_evaluation_flops and len(exhausted_dims) < len(param_names):

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
            param_chosen_iso_flops = []
            for data_val in data_grid:
                config = {**conf_scaling, **non_scaling_params, self.data_parameter_name: data_val}
                conf = config.copy()
                estimated_flops = self.flops_estimator(**conf)
                if estimated_flops > self.max_flops:
                    logger.info(f"Config {conf} exceeds max FLOPs ({self.max_flops:.2e}). Skipping.")
                    continue
                available_brackets = [flops for flops in iso_flops if len(configs_per_iso_flop[flops]) <= num_candidate_per_iso_flop]
                if not available_brackets:
                    logger.info(f"No available ISO-FLOP brackets for config {conf} with estimated FLOPs {estimated_flops:.2e}. Skipping.")
                    replace = True
                closest_flop_bin, ratio = self._get_closest_iso_flop(estimated_flops, available_brackets or iso_flops)
                if ratio > 1.3 or ratio < 0.7:
                    continue
                if closest_flop_bin in param_chosen_iso_flops:
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
                        param_chosen_iso_flops.append(closest_flop_bin)
                        used_flops += estimated_flops - removed_flops
                    continue
                if (used_flops + estimated_flops > self.max_evaluation_flops) or (estimated_flops > self.max_flops):
                    continue
                configs_per_iso_flop[closest_flop_bin].append(conf)
                param_chosen_iso_flops.append(closest_flop_bin)
                used_flops += estimated_flops
        
        config_list = [
            conf 
            for configs in configs_per_iso_flop.values() 
            for conf in configs
        ]
        
        logger.info(f"Generated {len(config_list)} configs across {len(configs_per_iso_flop)} ISO-FLOP levels, "
                   f"total FLOPs: {used_flops:.2e} / {self.max_evaluation_flops:.2e}")
        
        logger.info(f"Selected {len(config_list)} unique model configs for approach2")
        selected_model_configs = self.get_all_possible_model_configs(non_scaling_params)
        return config_list, selected_model_configs

    def gemstones_generate_config_list(self):
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
        param_bins = np.linspace(min_n, max_n, num=100)
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
        selected_model_configs = self.get_all_possible_model_configs(non_scaling_params)
        return conf_list, selected_model_configs
    
    def _get_closest_iso_flop(self, estimated_flops: float, iso_flops: list[float]) -> tuple[float, float]:
        closest = min(iso_flops, key=lambda x: abs(x - estimated_flops))
        return closest, closest / estimated_flops

    def _generate_iso_flop_list(self, num_steps: int = DEFAULT_ISO_FLOP_STEPS) -> list[float]:
        # assume that all the HPs are increasing, the other solution could be to get all possible grids and calc the flops for alll and get min and max there.
 
        return torch.logspace(math.log10(self.min_flops * 2), math.log10(self.max_flops), steps=num_steps).tolist()

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
            if param.prior is not None and self.use_priors:
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
    

    #: Default search range for the alpha/beta exponents of `L = E + A/N^a + B/D^b`.
    #: Appropriate for a next-token/contrastive training loss, which falls fast
    #: enough in N and D to land inside it. Accuracy-like targets fall far more
    #: slowly and need a wider range -- see `get_power_law_curvature`.
    DEFAULT_EXPONENT_BOUNDS = (0.25, 0.75)

    @classmethod
    def get_power_law_curvature(
        cls, N, D, L, best_loss_guess, exponent_bounds=None
    ) -> ScalingLawParameters:
        """Fit `L = E + A/N^alpha + B/D^beta` by Huber loss in log space.

        `best_loss_guess` upper-bounds the irreducible term E (searched over
        `(best_loss_guess * 1e-3, best_loss_guess)`), so it must be on the scale
        of whatever `L` is -- for an accuracy-like target extracted as `1 - acc`
        that is `1 - <achievable accuracy ceiling>`, not the training loss floor.

        `exponent_bounds` bounds alpha and beta; defaults to
        `DEFAULT_EXPONENT_BOUNDS`. Passing a range that excludes the true
        exponents does not merely widen the error bars: both exponents rail at
        the same bound, and the derived compute-optimal exponent
        `beta / (alpha + beta)` collapses to exactly 0.5 independent of the data.
        """
        lo, hi = cls.DEFAULT_EXPONENT_BOUNDS if exponent_bounds is None else exponent_bounds
        if not 0 < lo < hi:
            raise ValueError(f"exponent_bounds must satisfy 0 < lo < hi, got {(lo, hi)}")

        def huber(residual, delta=1e-3):
            abs_r = np.abs(residual)
            return np.where(
                abs_r <= delta,
                0.5 * residual**2,
                delta * (abs_r - 0.5 * delta),
            )

        def objective(theta, N, D, L):
            E, a, b, alpha, beta = theta
            
            # WIDENED clipping to allow negative exponents for values < 1.0
            a_clipped = np.clip(a, -20, 20)
            b_clipped = np.clip(b, -20, 20)
            A = np.exp(a_clipped)
            B = np.exp(b_clipped)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                N_alpha = np.power(N, alpha, where=(N > 0), out=np.ones_like(N, dtype=np.float64))
                D_beta = np.power(D, beta, where=(D > 0), out=np.ones_like(D, dtype=np.float64))
                
                N_alpha = np.where(N_alpha > 0, N_alpha, 1.0)
                D_beta = np.where(D_beta > 0, D_beta, 1.0)
                
                L_hat = E + A / N_alpha + B / D_beta
            
            if np.any(~np.isfinite(L_hat)) or np.any(L_hat <= 0):
                return 1e10  
            
            residual = np.log(L_hat) - np.log(L)
            loss = np.sum(huber(residual))
            
            if not np.isfinite(loss):
                return 1e10
                
            return loss

        num_points = len(L)
        min_L = best_loss_guess

        # Multi-start values for alpha/beta. The literals below reproduce the
        # historical default-bounds behaviour exactly; a custom range gets a
        # geometric ladder across it instead, since 0.4/0.5 can sit outside it.
        if (lo, hi) == cls.DEFAULT_EXPONENT_BOUNDS:
            exp_starts_small = [0.5]
            exp_starts_grid = [0.4, 0.5]
        else:
            exp_starts_small = [float(np.sqrt(lo * hi))]
            exp_starts_grid = [float(v) for v in np.geomspace(lo, hi, 4)[:3]]

        # FIXED: Initialize a and b closer to 0 or negative for normalized data
        # exp(0) = 1.0, exp(-2) = 0.13. These are realistic scales for normalized A and B.
        if num_points <= 10:
            initialization = [
                (min_L * e_scale, a0, a0, exp0, exp0)
                for e_scale, a0 in ((0.9, 0), (0.8, -2), (0.7, 2))
                for exp0 in exp_starts_small
            ]
        else:
            initialization = list(itertools.product(
                [min_L * 0.8, min_L * 0.9],
                [-2, 0, 2],         # a
                [-2, 0, 2],         # b
                exp_starts_grid,    # alpha
                exp_starts_grid,    # beta
            ))

        best_result = None
        best_loss = float("inf")
        
        patience_counter = 0
        patience_limit = 100
        N_scale = np.max(N)
        D_scale = np.max(D)

        N_norm = N / N_scale
        D_norm = D / D_scale
        
        for idx, theta0 in enumerate(initialization):
            if idx > 20 and patience_counter >= patience_limit:
                break
                
            try:
                # FIXED: Allow a and b to be negative, up to -20 (A = 2e-9)
                bounds = [
                    (min_L * 0.001, min_L),
                    (-20, 20),         # a
                    (-20, 20),         # b
                    (lo, hi),          # alpha
                    (lo, hi),          # beta
                ]
                
                res = minimize(
                    objective,
                    theta0,
                    args=(N_norm, D_norm, L),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                
                # FIXED: Removed the strict `res.success` check. 
                # If SciPy throws a line search warning but finds a valid lower loss, keep it!
                if np.isfinite(res.fun) and res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            except Exception as e:
                patience_counter += 1
                continue 

        if best_result is None:
            raise ValueError("Could not fit scaling law with any initialization.")

        E, a, b, alpha, beta = best_result.x
        
        # Apply the same widened clipping before undoing the log
        A = np.exp(np.clip(a, -20, 20))
        B = np.exp(np.clip(b, -20, 20))
        
        # Correctly un-normalize A and B back to raw compute space
        A = A * (N_scale ** alpha)
        B = B * (D_scale ** beta)
        
        print(f"Fitted scaling law parameters: E={E:.6f}, A={A:.6e}, B={B:.6e}, alpha={alpha:.4f}, beta={beta:.4f}")
        return ScalingLawParameters(E=E, A=A, B=B, alpha=alpha, beta=beta)
    
    @classmethod
    def _extract_scaling_metrics(cls, trials: Mapping[str, Trial]) -> list[tuple] | None:
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

    def _plot_flop_vs_param_and_data(self, trials: Mapping[str, Trial]) -> tuple[Figure, Figure] | None:
        use_lc_aware = self.strategy in ["approach1", "gemstones"]
        use_pareto_fit = self.strategy in ["approach1", "approach2"]
        pareto_data = self.extract_pareto_front_data(trials, lc_aware=use_lc_aware, pareto_fit=use_pareto_fit)
        
        xs = pareto_data['flops']
        ys_params = pareto_data['params']
        ys_data = pareto_data['data']
        
        fig1 = _create_scaling_plot(xs, ys_params, "Flops", "Params", "FLOPs vs Params")
        fig2 = _create_scaling_plot(xs, ys_data, "Flops", "Data Points", "FLOPs vs Data")
        
        return fig1, fig2

    @classmethod
    def _extract_trial_data(cls, trials: Mapping[str, Trial]) -> tuple[list, dict] | None:
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
        
        return (rows, trial_learning_curves) if rows else None

    @staticmethod
    def _extract_flop_loss_from_lc(learning_curve) -> list[tuple[float, float]]:
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
        # Extract trial data
        data = self._extract_trial_data(trials)
        if data is None:
            logger.warning("No evaluated trials with objectives/flops to plot.")
            return None
        
        rows, trial_learning_curves = data
        
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
    
    def get_trial_artifacts(self, trials: Mapping[str, Trial]) -> list[Artifact]:
        if trials is None or len(trials) < 5:
            return []
        
        # Get base class artifacts (class method needs to be called on the class)
        artifacts = super().get_trial_artifacts(trials) or []
        artifacts.extend(self._artifacts)
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
    max_flops: int,
    benchmark_type: BenchmarkEnum,
    strategy: Literal["approach1", "approach2", "gemstones"] = "gemstones",
    use_priors: bool = False,
) -> CustomOptimizer:
        
    return custom(name="chinchilla", optimizer=lambda space: Chinchilla_Guided_Scaling(
        space=convert_neps_to_classic_search_space(space), # type: ignore
        flops_estimator=flops_estimator,
        params_estimator=params_estimator,
        seen_datapoints_estimator=seen_datapoints_estimator,
        aspect_ratio_fn=aspect_ratio_fn,
        max_evaluation_flops=max_evaluation_flops,
        evaluate_pipeline=evaluate_pipeline,
        data_parameter_name=data_parameter_name,
        strategy=strategy,
        use_priors=use_priors,
        max_flops=max_flops,
        benchmark_type=benchmark_type,
    ))
