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

from scan_benchmark.base_performance_benchmark import PerformancePredictorType
import neps


from scan_benchmark.llm.api import LLMBenchmark, LLMConfig, LLMTarget

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

def get_benchmark(seed: int = 1, predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN,) -> LLMBenchmark:
    # target=None would fall back to LLMTarget.default() ("test_loss"), which
    # is NOT what LLM_TARGETS enables (VAL_LOSS / "valid_loss") -- pass the
    # list explicitly so LLMBenchmark predicts the intended target.
    return LLMBenchmark(
        target=LLM_TARGETS,
        predictor_type=predictor_type
    )

MIN_N_HEADS, MAX_N_HEADS = 6, 38  # must match valid_n_heads in scan_benchmark.llm.config.LLMConfig

def find_closest_divisible_n_heads(d_model: int, n_heads: int) -> int:
    for nh in range(n_heads, MIN_N_HEADS - 2, -2):
        if d_model % nh == 0:
            return nh
    for nh in range(n_heads + 2, MAX_N_HEADS + 2, 2):
        if d_model % nh == 0:
            return nh
    raise ValueError(f"Could not find a valid n_heads for d_model {d_model} close to {n_heads}")


def normalize_llm_config(raw_conf: dict) -> dict:
    """Snap config values to the nearest valid values in the LLM search space.
    
    - Integer dimensions are rounded to nearest valid integer and clamped.
    - Float dimensions are clamped to their valid domain bounds.
    - Scales d_model by 64 and adjusts n_heads accordingly.
    """
    conf = dict(raw_conf)
    
    float_bounds = {
        "lr": (1.0e-5, 1.0e-2),
        "weight_decay": (1.0e-3, 0.2),
        "beta1": (0.8, 0.99),
        "beta2": (0.9, 0.999),
        "cooldown_steps": (0.0, 0.3),
    }
    int_bounds = {
        "d_model": (4, 24),
        "n_layers": (4, 30),
        "n_heads": (3, 19),
        "n_tokens": (int(2e8), int(4e10)),
    }
    
    for name, value in list(conf.items()):
        if name in int_bounds:
            lower_i, upper_i = int_bounds[name]
            snapped = int(round(float(value)))
            conf[name] = min(max(snapped, lower_i), upper_i)
        elif name in float_bounds:
            lower_f, upper_f = float_bounds[name]
            v = float(value)
            conf[name] = min(max(v, lower_f), upper_f)
    
    # Scale d_model by 64 and adjust n_heads
    conf["d_model"] = int(conf["d_model"]) * 64
    conf["n_heads"] = find_closest_divisible_n_heads(conf["d_model"], int(conf["n_heads"]) * 2)
    
    return conf


def parameters_estimator(**conf):
    conf = normalize_llm_config(conf)
    conf["n_layers"] = int(conf["n_layers"])
    conf["warmup_steps"] = 0.1
    config = LLMConfig(**conf)
    stats = config.compute_model_stats()
    return stats["total_params"]
    
def evaluate_pipeline_wrapper(bench: LLMBenchmark, lc_aware: bool = False):
    def evaluate_pipeline(lr, weight_decay, beta1, beta2, cooldown_steps, d_model, n_layers, n_heads, n_tokens) -> dict:
        normalized_conf = normalize_llm_config({
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "cooldown_steps": cooldown_steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_tokens": n_tokens,
        })
        
        lr = normalized_conf["lr"]
        weight_decay = normalized_conf["weight_decay"]
        beta1 = normalized_conf["beta1"]
        beta2 = normalized_conf["beta2"]
        cooldown_steps = normalized_conf["cooldown_steps"]
        d_model = normalized_conf["d_model"]
        n_layers = normalized_conf["n_layers"]
        n_heads = normalized_conf["n_heads"]
        n_tokens = normalized_conf["n_tokens"]

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
                    lc.append((n_data, flops, float(result["predictions"]["mean"])))
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
            "objective_to_minimize": float((result.get("predictions") or {}).get("mean", float('nan'))),
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
    conf = normalize_llm_config(conf)
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
        upper=24,
        is_scaling=True,
    )
    
    n_layers = neps.Integer(
        lower=4,
        upper=30,
        is_scaling=True,
    )

    n_heads = neps.Integer( #2
        lower=3,
        upper=19,
        prior=4,
        prior_confidence='high',
    )

    n_tokens = neps.Integer(
        lower=int(2e8),
        upper=int(4e10),
        log=True,
        is_scaling=True,
    )

logging.basicConfig(level=logging.INFO,)

def get_macro_params_bounds(bench: LLMBenchmark) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    max_flop_conf = dict(
        lr=1.0e-2, weight_decay=2.0e-1, beta1=0.9, beta2=0.999, cooldown_steps=0.3, d_model=24, n_layers=30, n_heads=19, n_tokens=4e10
    )
    
    min_flop_conf = dict(
        lr=1.0e-2, weight_decay=2.0e-1, beta1=0.8, beta2=0.9, cooldown_steps=0, d_model=4, n_layers=4, n_heads=3, n_tokens=2e8
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

# 9.266657e+08 parameters, 4.000000e+10 data points, total flops: 2.676758e+20
# 1.628109e+07 parameters, 2.000000e+08 data points, total flops: 2.456771e+16

def get_pipeline_space():
    space = PipelineScalingConfig()
    return space

if __name__ == '__main__':
    get_macro_params_bounds(get_benchmark())