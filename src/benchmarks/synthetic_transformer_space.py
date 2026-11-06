from collections.abc import Callable
from typing import Any
import neps
import numpy as np
import socket
import os
import logging
import torch
import random

from src.baselines.configs import OptimizerEnum, get_optimizer_wrapper
from src.benchmarks.synthetic_transformer import ChinchillaSyntheticBench

# perfect power law between number of parameters and loss: L(N, D) =  E + 𝐴 / 𝑁 ^ 𝛼 + 𝐵 / 𝐷 ^ 𝛽 
# where 𝛼=0.34 and 𝛽=0.28 and 𝐸 = 1.69, 𝐴 = 406.4, 𝐵 = 410.7
# and flops = 6 * D * N

idxx = 0
MAX_EPOCHS = 120
def get_number_of_parameters(d_model, num_layers, num_heads, vocab_size, learning_rate, batch_size, max_seq_length=128, epoch=MAX_EPOCHS):
    """
    Cheap estimate of transformer parameter count.
    
    Typical transformer param breakdown:
    - Embeddings: vocab_size * d_model + max_seq_length * d_model
    - Each layer: ~4 * d_model^2 (self-attn + FFN)
    - Total: roughly num_layers * 4 * d_model^2
    """
    d_ff = 4 * d_model  # FFN hidden dimension (standard)
    
    # Embedding layer
    embedding_params = vocab_size * d_model + max_seq_length * d_model
    
    # Per transformer layer: self-attention + FFN
    # Self-attention: Q, K, V projections + output projection ~ 4 * d_model^2
    # FFN: 2 linear layers ~ 2 * d_model * d_ff ~ 8 * d_model^2
    per_layer_params = 4 * d_model * d_model + 2 * d_model * d_ff
    
    # Total: embeddings + num_layers * per_layer_params + output layer
    total_params = embedding_params + num_layers * per_layer_params + d_model * vocab_size
    
    return int(total_params)

def evaluate_pipeline(d_model, num_layers, num_heads, vocab_size, learning_rate, batch_size,max_seq_length=128, epoch=MAX_EPOCHS):
    global idxx
    bench = ChinchillaSyntheticBench(noise_std=0.005) # Small noise
    N = get_number_of_parameters(d_model=d_model, num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, max_seq_length=max_seq_length, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    D = seen_datapoints_estimator(d_model=d_model, num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, max_seq_length=max_seq_length, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    result = bench.query(N=N, D=D, num_points=4, lr_penalty_factor=0.1)
    assert result["loss"] is not None and result["loss"] is not np.nan, "Bench returned invalid loss"
    # idxx += 1
    # if idxx < 5:
    return {
        "objective_to_minimize": float(result["loss"]), 
        "cost": float(result["flops"]), 
        "learning_curve": result["learning_curve"],
        "info_dict": {
            "n_param": int(N),
            "flops": float(result["flops"]),
            "n_data": int(D),
        },
    }

def get_total_flops(d_model, num_layers, num_heads, vocab_size, learning_rate, batch_size, max_seq_length=128, epoch=MAX_EPOCHS):
    N = get_number_of_parameters(d_model=d_model, num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, max_seq_length=max_seq_length, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    D = seen_datapoints_estimator(d_model=d_model, num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, max_seq_length=max_seq_length, learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
    assert N is not None
    assert D is not None
    return 6 * N * D

def aspect_ratio_fn(d_model, num_layers, num_heads, vocab_size, learning_rate, batch_size, max_seq_length=128, epoch=MAX_EPOCHS):
    return d_model / num_layers

def seen_datapoints_estimator(d_model, num_layers, num_heads, vocab_size, learning_rate, batch_size, max_seq_length=128, epoch=MAX_EPOCHS):
    """
    Cheap estimate of total tokens seen during training.
    Total tokens = num_epochs * batch_size * sequence_length
    This scales with batch_size and sequence_length, independent of model capacity.
    """
    return epoch * batch_size * max_seq_length

class PipelineScalingConfig(neps.PipelineSpace):
    epoch = MAX_EPOCHS
    max_seq_length = 2048
    # max_seq_length = neps.Integer(lower=128, upper=1024, is_scaling=True)  # Max sequence length (affects params and data)
    d_model = neps.Integer(lower=32, upper=256, is_scaling=True)  # Embedding dimension
    num_layers = neps.Integer(lower=2, upper=12, is_scaling=True)  # Number of transformer blocks
    num_heads = neps.Integer(lower=2, upper=8, is_scaling=False)  # Attention heads (usually fixed)
    vocab_size = neps.Integer(lower=1000, upper=10000, is_scaling=True)  # Vocabulary size
    learning_rate = neps.Float(lower=0.0001, upper=0.1)
    batch_size = neps.Integer(lower=32, upper=8096, is_scaling=True)  # Batch size affects total tokens

    _grid_sizes = {
        "batch_size": 5,
        "learning_rate": 4,
        "num_heads": 3,
        "vocab_size": 3,
    }

pipeline_space = PipelineScalingConfig()

logging.basicConfig(level=logging.INFO,)
print("max flops")
# Max flops calculation with transformer at max scale
max_params = get_number_of_parameters(d_model=256, num_layers=12, num_heads=8, vocab_size=10000, max_seq_length=2048, learning_rate=0.1, batch_size=8096, epoch=MAX_EPOCHS)
max_data = seen_datapoints_estimator(d_model=256, num_layers=12, num_heads=8, vocab_size=10000, max_seq_length=2048, learning_rate=0.1, batch_size=8096, epoch=MAX_EPOCHS)
min_params = get_number_of_parameters(d_model=32, num_layers=2, num_heads=2, vocab_size=1000, max_seq_length=2048, learning_rate=0.0001, batch_size=32, epoch=MAX_EPOCHS)
min_data = seen_datapoints_estimator(d_model=32, num_layers=2, num_heads=2, vocab_size=1000, max_seq_length=2048, learning_rate=0.0001, batch_size=32, epoch=MAX_EPOCHS)
print(f"{max_params:e}  parameters, {max_data:e} data points, total flops: {(6 * max_params * max_data):e}")
print(f"{min_params:e}  parameters, {min_data:e} data points, total flops: {(6 * min_params * min_data):e}")
# 1.252147e+07  parameters, 5.242880e+08 data points, total flops: 3.938915e+16
# 1.221120e+05  parameters, 1.638400e+07 data points, total flops: 1.200410e+13

MAX_TAGET_FLOPS = 4 * 10**16
MAX_EVALUATION_FLOPS = 1 * 1 * max_params * max_data


if __name__ == '__main__':
    get_optimizer = get_optimizer_wrapper(
            pipeline_space=pipeline_space,
            param_estimator=get_number_of_parameters,
            seen_datapoints_estimator=seen_datapoints_estimator,
            flops_estimator=get_total_flops,
            aspect_ratio_fn=aspect_ratio_fn,
            evaluate_pipeline=evaluate_pipeline,
            data_parameter_name="batch_size"
    )
    optimizers = [
        OptimizerEnum.CHINCHILLA_APPROACH1,
        # OptimizerEnum.CHINCHILLA_APPROACH2,
    ]
    for optimizer_name in optimizers:
        for seed in [40]:
            print(f"Running {optimizer_name} with seed {seed}...")
            # set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            optimizer, kwargs = get_optimizer(
                OptimizerEnum(optimizer_name), 
                max_evaluation_flops=MAX_EVALUATION_FLOPS, 
                max_target_flops=MAX_TAGET_FLOPS, 
                seed=seed,
            )
            try:
                neps.run(
                    evaluate_pipeline=evaluate_pipeline, # for carbs it will be overrided by the one in kwargs
                    pipeline_space=pipeline_space,
                    root_directory=f"results/synthetic_transformer/{optimizer_name}/seed_{seed}",
                    cost_to_spend=0,
                    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
                    overwrite_root_directory=False,
                    optimizer=optimizer,
                    **kwargs,
                )
            except Exception as e:
                print(f"Error occured {e}")
                raise e
        print("Next optimizer...")
