# from lcbench import mfpbench_benches
from benchmarks.synthetic_transformer import ChinchillaSyntheticBench
import numpy as np
import neps
import socket
import os
import logging
from neps.optimizers.algorithms import bo_guided_scaling, chinchilla_guided_scaling, random_search, grid_search
# from lcbench import estimate_lcbench_flops, Objective

# perfect power law between number of parameters and loss: L(N, D) =  E + 𝐴 / 𝑁 ^ 𝛼 + 𝐵 / 𝐷 ^ 𝛽 
# where 𝛼=0.34 and 𝛽=0.28 and 𝐸 = 1.69, 𝐴 = 406.4, 𝐵 = 410.7
# and flops = 6 * D * N

idxx = 0
def _get_layer_widths(max_units, num_layers):
    """Reconstructs the funnel shape: [input, L1, L2, ..., output]"""
    widths = []
    
    current_width = max_units
    
    # LCBench Funnel Strategy: Halve width each layer, but keep >= 16 or num_classes
    for _ in range(num_layers):
        widths.append(current_width)
        current_width = max(16, int(current_width / 2))
    return widths

# def get_number_of_parameters( num_layers, max_units, learning_rate, batch_size, epoch=52):
def get_number_of_parameters(max_units, learning_rate, batch_size, num_layers, epoch=52):
    """Calculates total trainable parameters (Weights + Biases)."""
    widths = _get_layer_widths(max_units, num_layers)
    total_params = 0
    
    for i in range(len(widths) - 1):
        n_in = widths[i]
        n_out = widths[i+1]
        
        # Linear Layer: Weights + Biases
        weights = n_in * n_out
        biases = n_out
        total_params += weights + biases
    return total_params

# def evaluate_pipeline(num_layers, max_units, learning_rate, batch_size, epoch=52):
def evaluate_pipeline(max_units, learning_rate, batch_size, num_layers, epoch=52):
    global idxx
    bench = ChinchillaSyntheticBench(noise_std=0.005) # Small noise
    N = get_number_of_parameters( num_layers=num_layers, max_units=max_units, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    D = seen_datapoints_estimator(num_layers=num_layers, max_units=max_units, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    result = bench.query(N=N, D=D)
    assert result["loss"] is not None and result["loss"] is not np.nan, "Bench returned invalid loss"
    # idxx += 1
    # if idxx < 5:
    return {"objective_to_minimize": result["loss"], 
        "cost": result["flops"], 
        "info_dict": {
            "n_param": N,
            "flops": 6 * N * D,
            "n_data": D}}
    # else:
    #     return None

def get_total_flops(max_units, learning_rate, batch_size, num_layers, epoch=52):
    N = get_number_of_parameters(num_layers=num_layers, max_units=max_units, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    D = seen_datapoints_estimator(num_layers=num_layers, max_units=max_units, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    assert N is not None
    assert D is not None
    return 6 * N * D

def seen_datapoints_estimator(max_units, learning_rate, batch_size, num_layers, epoch=52):
    return epoch * batch_size

class DummySpace(neps.PipelineSpace):
    epoch=52
    num_layers = neps.Integer(lower=2, upper=4, is_scaling=True)
    max_units = neps.Integer(lower=64, upper=1024, is_scaling=True)
    learning_rate = neps.Float(lower=0.0001, upper=0.1)
    batch_size = neps.Integer(lower=64, upper=512, is_scaling=True)

    _grid_sizes = {
        "batch_size": 10,
        "learning_rate": 5,
    }

max_target_flops = 8 * 10**9
pipeline_space = DummySpace()

logging.basicConfig(level=logging.INFO,)
print("max flops")
print(f"{(6* 52 * 512 * get_number_of_parameters(num_layers=4, max_units=1024, learning_rate=0.1, batch_size=512, epoch=52)):e}")

optimizers = {
    "bo_space_expansion": lambda space: bo_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        seen_datapoints_estimator=seen_datapoints_estimator,
        flops_estimator=get_total_flops,
        max_evaluation_flops=max_target_flops * 32,
        max_target_flops=max_target_flops,
        device='cpu',
        sampling_strategy="space_expansion",
    ),
    
    
    "bo_left_budget": lambda space: bo_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        seen_datapoints_estimator=seen_datapoints_estimator,
        flops_estimator=get_total_flops,
        max_evaluation_flops=max_target_flops * 32,
        max_target_flops=max_target_flops,
        device='cpu',
        sampling_strategy="left_budget_contraction",
    ),
    
    
    "grid_search": lambda space: chinchilla_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        seen_datapoints_estimator=seen_datapoints_estimator,
        flops_estimator=get_total_flops,
        max_evaluation_flops=max_target_flops * 32,
        base_optimizer=grid_search(
            space,
            ignore_fidelity=True,
            size_per_numerical_dimension=5,
        )
    ),
    
    "random_search": lambda space: chinchilla_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        seen_datapoints_estimator=seen_datapoints_estimator,
        flops_estimator=get_total_flops,
        max_evaluation_flops=max_target_flops * 32,
        base_optimizer=random_search(
            space,
            use_priors=False,
            ignore_fidelity=True,
        )
    ),
    "carbs": '',
}
import torch
import random

# for optimizer_name, optimizer in optimizers.items():
#     for seed in [42, 43, 44]:
#         print(f"Running {optimizer_name} with seed {seed}...")
#         # set seeds
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if optimizer_name == 'carbs':
#             optimizer, carbs_opt = get_carbs_optimizer(pipeline_space, seed)
#         try:
#             neps.run(
#                 evaluate_pipeline=evaluate_pipeline if optimizer_name != 'carbs' else evaluate_pipeline_wrapper(carbs_opt),
#                 pipeline_space=pipeline_space,
#                 root_directory=f"results/{optimizer_name}/seed_{seed}",
#                 cost_to_spend=max_target_flops * 32,
#                 worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
#                 overwrite_root_directory=True,
#                 optimizer=optimizer,
#             )
#         except Exception as e:
#             print(f"Error occured {e}")
#     print("Next optimizer...")

