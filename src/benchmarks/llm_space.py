import argparse
import numpy as np
import socket
import os
import logging
import torch
import copy
import random
import inspect
import json
from pathlib import Path

import neps


from dep.scan_bench_suite.scan_benchmark.llm.api import LLMBenchmark, LLMConfig, LLMTarget

LLM_TARGETS = [
    LLMTarget.VAL_LOSS,
    # LLMTarget.TEST_LOSS,
    # LLMTarget.HELMASWAG_ACC,
    # LLMTarget.ARC_CHALLENGE_ACC,
    # LLMTarget.ARC_EASY_ACC,
    # LLMTarget.COPA_ACC,
    # LLMTarget.OPENBOOKQA_ACC,
    # LLMTarget.PIQA_ACC
]

def get_benchmark(seed: int = 1) -> LLMBenchmark:
    return LLMBenchmark(
        targets=LLM_TARGETS) # type: ignore

def find_closest_divisible_n_heads(d_model: int, n_heads: int) -> int:
    for nh in range(n_heads, 6, -2):
        if d_model % nh == 0:
            return nh
    for nh in range(n_heads, 24, +2):
        if d_model % nh == 0:
            return nh
    raise ValueError(f"Could not find a valid n_heads for d_model {d_model} close to {n_heads}")
    

def parameters_estimator(**conf):
    conf["d_model"] = int(conf["d_model"]) * 64
    conf["n_heads"] = find_closest_divisible_n_heads(conf["d_model"], int(conf["n_heads"]) * 2)
    
    conf["n_layers"] = int(conf["n_layers"])
    conf["warmup_steps"] = 0.1
    config = LLMConfig(**conf)
    stats = config.compute_model_stats()
    return stats["total_params"]
    
def evaluate_pipeline_wrapper(bench: LLMBenchmark, lc_aware: bool = False):
    def evaluate_pipeline(lr, weight_decay, beta1, beta2, cooldown_steps, d_model, n_layers, n_heads, n_tokens) -> dict:
        d_model = 64 * int(d_model)
        n_heads = find_closest_divisible_n_heads(d_model, 2 * int(n_heads))

        config = LLMConfig(
            d_model=d_model,
            n_layers=int(n_layers),
            n_heads=n_heads,
            lr=round(lr, 7),
            weight_decay=round(weight_decay, 7),
            beta1=round(beta1, 7),
            beta2=round(beta2, 7),
            warmup_steps=0.1,
            cooldown_steps=round(cooldown_steps, 7),
            n_tokens=n_tokens,
            training_progress=1.0
        )
        lc = None
        results = []
        print(f"lc_aware {lc_aware}")
        if lc_aware:
            lc = []
            confs = []
            for i in range(1, 4 + 1): # num_points in learning curve
                config.training_progress = i / 4
                confs.append(copy.deepcopy(config))
                results = bench.query_many(confs)
                for result in results:
                    # if result["failed"]:
                        # raise ValueError("Benchmark query failed with config: {config}")
                    n_data = config.training_progress * n_tokens # num tokens seen so far
                    flops = float(result["model_stats"]["total_training_flops"])
                    lc.append((n_data, flops, float(result["predictions"]["valid_loss"]["mean"])))
        config.training_progress = 1.0 # for double caution
        try:
            result = bench.query(config) if not lc_aware else results[-1]
        except Exception as e:
            return {
                "objective_to_minimize": float('nan'), 
                "cost": float(config.compute_model_stats()["total_training_flops"]), 
                "learning_curve": lc,
                "info_dict": {
                    "n_param": int(config.compute_model_stats()["total_params"]),
                    "flops": float(config.compute_model_stats()["total_training_flops"]),
                    "n_data": n_tokens,
                    "predictions": None,
                    "model_stats": None,
                },
            }
        # if result["failed"]:
            # print(f"Benchmark query failed with config: {config}")
        return {
            "objective_to_minimize": float((result.get("predictions") or {}).get("valid_loss", {}).get('mean', float('nan'))), 
            "cost": float(result["model_stats"]["total_training_flops"]), 
            "learning_curve": lc,
            "info_dict": {
                "n_param": int(result["model_stats"]["total_params"]),
                "flops": float(result["model_stats"]["total_training_flops"]),
                "n_data": n_tokens,
                "predictions": result["predictions"],
                "model_stats": result["model_stats"],
            },
        }
    return evaluate_pipeline

def flops_estimator(**conf):
    conf["d_model"] = int(conf["d_model"]) * 64
    conf["n_heads"] = find_closest_divisible_n_heads(conf["d_model"], int(conf["n_heads"]) * 2)
    conf["n_layers"] = int(conf["n_layers"])
    conf["warmup_steps"] = 0.1
    config = LLMConfig(**conf)
    stats = config.compute_model_stats()
    return stats["total_training_flops"]

def aspect_ratio_fn(**conf) -> float:
    return 64 * conf["d_model"] / conf["n_layers"]

def seen_datapoints_estimator(**conf):
    return conf["n_tokens"]

class PipelineScalingConfig(neps.PipelineSpace):
    # hps
    lr = neps.Float(
        lower=1.0e-5,
        upper=1.0e-2,
        log=True,
        prior=3.0e-3,
        prior_confidence='high',
    )
    weight_decay = neps.Float(
        lower=1.0e-3,
        upper=0.2,
        log=True,
        prior=1.0e-1,
        prior_confidence='high',
    )
    
    beta1 = neps.Float(lower=0.8, upper=0.99, log=True, prior=0.9, prior_confidence='high')
    beta2 = neps.Float(lower=0.9, upper=0.999, log=True, prior=0.95, prior_confidence='high')
    cooldown_steps = neps.Float(lower=0, upper=0.3, prior=0.1, prior_confidence='high')

    # scale parameters
    d_model = neps.Integer( #64
        lower=4,
        upper=18,
        is_scaling=True,
    )
    
    n_layers = neps.Integer(
        lower=4,
        upper=24,
        is_scaling=True,
    )

    n_heads = neps.Integer( #2
        lower=3,
        upper=12,
        prior=4,
        prior_confidence='high',
    )

    n_tokens = neps.Integer(
        lower=int(2e8),
        upper=int(1.6e10),
        log=True,
        is_scaling=True,
    )

logging.basicConfig(level=logging.INFO,)

def get_macro_params_bounds(bench: LLMBenchmark) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    max_flop_conf = dict(
        lr=1.0e-2, weight_decay=2.0e-1, beta1=0.9, beta2=0.999, cooldown_steps=0.3, d_model=18, n_layers=24, n_heads=12, n_tokens=1.6e10
    )
    
    min_flop_conf = dict(
        lr=1.0e-2, weight_decay=2.0e-1, beta1=0.8, beta2=0.9, cooldown_steps=0, d_model=4, n_layers=4, n_heads=4, n_tokens=2e8
    )

    res = evaluate_pipeline_wrapper(bench)(
        **max_flop_conf
    )
    max_params = res["info_dict"]["n_param"]
    max_data = res["info_dict"]["n_data"]
    max_flops = res["info_dict"]["flops"]
    

    res = evaluate_pipeline_wrapper(bench)(
        **min_flop_conf
    )
    min_params = res["info_dict"]["n_param"]
    min_data = res["info_dict"]["n_data"]
    min_flops = res["info_dict"]["flops"]

    print(f"{max_params:e} parameters, {max_data:e} data points, total flops: {max_flops:e}")
    print(f"{min_params:e} parameters, {min_data:e} data points, total flops: {min_flops:e}")
    return ((min_params, min_data, min_flops), (max_params, max_data, max_flops))

# 4.401815e+08 parameters, 1.600000e+10 data points, total flops: 5.312364e+19
# 1.628109e+07 parameters, 2.000000e+08 data points, total flops: 2.456771e+16

def get_pipeline_space():
    space = PipelineScalingConfig()
    return space

if __name__ == '__main__':
    get_macro_params_bounds(get_benchmark())