from lcbench import mfpbench_benches
import numpy as np
import neps
import socket
import os
import logging
from neps.optimizers.algorithms import kaplan_guided_scaling
from lcbench import estimate_lcbench_flops, Objective

dataset_size= 1000 #TODO
bench_name, bench_desc = list(mfpbench_benches.items())[3]
bench = bench_desc.load(bench_desc)

conf_const = dict(
    max_dropout = 0.2442070308082,
    momentum = 0.99,
    weight_decay = 0.0473271620733,
    # epoch=52,
)

def _get_layer_widths(max_units, num_layers):
    """Reconstructs the funnel shape: [input, L1, L2, ..., output]"""
    widths = []
    
    current_width = max_units
    
    # LCBench Funnel Strategy: Halve width each layer, but keep >= 16 or num_classes
    for _ in range(num_layers):
        widths.append(current_width)
        current_width = max(16, int(current_width / 2))
    return widths

def get_number_of_parameters( num_layers, max_units, learning_rate, batch_size, epoch=52):
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

def evaluate_pipeline(num_layers, max_units, learning_rate, batch_size, epoch=52):
    conf = dict(epoch=epoch, num_layers=num_layers, max_units=max_units, learning_rate=learning_rate, batch_size=batch_size)
    conf.update(conf_const)
    r = bench.query(objectives=[Objective.VAL_ACCURACY, Objective.TIME], **conf)
    return {"objective_to_minimize": r[0], "cost": get_total_flops(num_layers, max_units, learning_rate, batch_size, epoch)}


def get_total_flops(num_layers, max_units, learning_rate, batch_size, epoch=52):
    conf = dict(epoch=epoch, num_layers=num_layers, max_units=max_units, batch_size=batch_size, learning_rate=learning_rate)
    conf.update(conf_const)
    return bench.query(objectives=[Objective.TIME], **conf)[0]

def seen_datapoints_estimator(num_layers, max_units, learning_rate, batch_size, epoch=52):
    return epoch * batch_size


pipeline_space = dict(
    epoch=neps.Integer(lower=1, upper=52),
    num_layers = neps.Integer(lower=2, upper=4),
    max_units = neps.Integer(lower=64, upper=1024),
    learning_rate = neps.Float(lower=0.0001, upper=0.1, log=True),
    batch_size = neps.Integer(lower=64, upper=512),
)


logging.basicConfig(level=logging.DEBUG)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results3/hyperparameters_example",
    # cost_to_spend=1000000000000000,
    evaluations_to_spend=1000000,
    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
    overwrite_root_directory=True,
    optimizer=lambda space: kaplan_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        flops_estimator=get_total_flops,
        seen_datapoints_estimator=seen_datapoints_estimator,
        max_evaluation_flops=1000000000000000,
        max_target_flop=200000000000000,
    ),
)

