import logging
import copy

import neps

from surrogate_benchmark.vlm.api import VLMBenchmark, VLMConfig, Target


OPENCLIP_TARGETS = [
    Target.VAL_LOSS, 
    Target.TEST_LOSS, 
    Target.RETRIEVAL_MSCOCO, 
    Target.CIFAR10, 
    # Target.VTAB_CIFAR100, 
    Target.IMAGENET1K,
    # Target.IMAGENETV2, 
    # Target.MNIST,
    # Target.FAIRNESS_DOLLAR_STREET,
]
def get_benchmark():
    return VLMBenchmark(targets=OPENCLIP_TARGETS)

def parameters_estimator_wrapper(bench: VLMBenchmark):
    def parameters_estimator(**conf):
        conf["vision_width"] = int(conf["vision_width"] * 64 if conf["vision_width"] > 0 else 32)
        conf["text_width"] = int(conf["text_width"] * 64 if conf["text_width"] > 0 else 32)
        conf["total_samples_planned"] = int(conf["train_num_samples"] * 1e6)
        conf["training_progress"] = 1.0
        conf.pop("train_num_samples")
        config = VLMConfig(**conf)
        stats = bench.get_model_stats(config)
        return stats["mparams"] * 1e6

    return parameters_estimator
    
def evaluate_pipeline_wrapper(bench: VLMBenchmark, lc_aware: bool = False):
    def evaluate_pipeline(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples):
        vision_width = vision_width * 64 if vision_width > 0 else 32
        text_width = text_width * 64 if text_width > 0 else 32
        train_num_samples *= 1e6
        train_num_samples = int(train_num_samples)
        config = VLMConfig(
            lr=round(lr, 7),
            wd=round(wd, 7),
            beta1=round(beta1, 7),
            beta2=round(beta2, 7),
            warmup_fraction=round(warmup_fraction, 7),
            eps=round(eps, 10),
            vision_width=vision_width,
            text_width=text_width,
            total_samples_planned=train_num_samples,
            training_progress=1.0
        )
        lc = None
        print(f"lc_aware {lc_aware}")
        if lc_aware:
            lc = []
            confs = []
            for i in range(1, 4 + 1): # num_points in learning curve
                config.training_progress = i / 4
                confs.append(copy.deepcopy(config))
                results = bench.query_many(confs)
                for result in results:
                    if result["failed"]:
                        return {
                            "objective_to_minimize": float('nan'), 
                            "cost": float(result["model_stats"]["total_gflops"]), 
                            "learning_curve": None,
                            "info_dict": {
                                "n_param": int(result["model_stats"]["mparams"]) * 1e6,
                                "flops": float(result["model_stats"]["total_gflops"]),
                                "n_data": train_num_samples,
                                "predictions": result["predictions"],
                                "model_stats": result["model_stats"],
                            },
                        }
                    n_data = config.training_progress * train_num_samples # num tokens seen so far
                    flops = float(result["model_stats"]["total_gflops"])
                    lc.append((n_data, flops, float(result["predictions"][Target.VAL_LOSS.value]["mean"])))
        config.training_progress = 1.0 # for double caution
        result = bench.query(config)
        if result["failed"]:
            print(f"Benchmark query failed with config: {config}")
        return {
            "objective_to_minimize": float((result.get("predictions") or {}).get(Target.VAL_LOSS.value, {}).get('mean', float('nan'))), 
            "cost": float(result["model_stats"]["total_gflops"]), 
            "learning_curve": lc,
            "info_dict": {
                "n_param": int(result["model_stats"]["mparams"]) * 1e6,
                "flops": float(result["model_stats"]["total_gflops"]),
                "n_data": train_num_samples,
                "predictions": result["predictions"],
                "model_stats": result["model_stats"],
            },
        }
    return evaluate_pipeline

def flops_estimator_wrapper(bench: VLMBenchmark):
    def flops_estimator(**conf):
        conf["vision_width"] = int(conf["vision_width"] * 64 if conf["vision_width"] > 0 else 32)
        conf["text_width"] = int(conf["text_width"] * 64 if conf["text_width"] > 0 else 32)
        conf["total_samples_planned"] = int(conf["train_num_samples"] * 1e6)
        conf["training_progress"] = 1.0
        conf.pop("train_num_samples")
        config = VLMConfig(**conf)
        
        stats = bench.get_model_stats(config)
        return stats["total_gflops"]

    return flops_estimator

def aspect_ratio_fn(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples) -> float:
    vision_width = vision_width * 64 if vision_width > 0 else 32
    text_width = text_width * 64 if text_width > 0 else 32
    train_num_samples *= 1e6
    train_num_samples = int(train_num_samples)
    import math
    vision_ratio = vision_width  / 12 # constant n_layers
    text_ratio = text_width  / 12 # constant n_layers
    return vision_ratio / text_ratio

def seen_datapoints_estimator(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples):
    vision_width = vision_width * 64 if vision_width > 0 else 32
    text_width = text_width * 64 if text_width > 0 else 32
    train_num_samples *= 1e6
    train_num_samples = int(train_num_samples)
    return train_num_samples

class PipelineScalingConfig(neps.PipelineSpace):
    # hps
    lr = neps.Float(
        lower=1.0e-5,
        upper=5.0e-2,
        log=True,
        prior=2.0e-3,
        prior_confidence='high',
    )
    wd = neps.Float(
        lower=1.0e-6,
        upper=2.0e-1,
        log=True,
        prior=1.0e-3,
        prior_confidence='high',
    )
    warmup_fraction = neps.Float(
        lower=0.0,
        upper=0.75,
        prior=0.25,
        prior_confidence='high',
    )
    beta1 = neps.Float(lower=0.9, upper=0.99)
    beta2 = neps.Float(lower=0.95, upper=0.999)
    eps = neps.Float(lower=1e-8, upper=1e-6, log=True)

    # scale space
    train_num_samples = neps.Float(
        lower=0.6,
        upper=60,
        log=True,
        is_scaling=True,
    )
    vision_width = neps.Integer(
        lower=0, upper=8,  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
        is_scaling=True,
    )
    text_width = neps.Integer(
        lower=0, upper=8,
        is_scaling=True,
    )

logging.basicConfig(level=logging.INFO,)

def get_macro_params_bounds(bench: VLMBenchmark) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    max_flop_conf = dict(
        lr=1.0e-3,
        wd=1.0e-6,
        beta1=0.99,
        beta2=0.999,
        warmup_fraction=0.75,
        eps=1e-8,
        vision_width=8,
        text_width=8,
        train_num_samples=60,
    )
    
    min_flop_conf = dict(
        lr=1.0e-5,
        wd=1.0e-6,
        beta1=0.99,
        beta2=0.999,
        warmup_fraction=0.75,
        eps=1e-8,
        vision_width=0,
        text_width=0,
        train_num_samples=0.6,
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

# 1.020000e+02 parameters, 6.000000e+07 data points, total flops: 5.970000e+08
# 2.000000e+00 parameters, 6.000000e+05 data points, total flops: 3.600000e+04
def get_pipeline_space():
    space = PipelineScalingConfig()
    return space

# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description='Run scaling study on OpenCLIP benchmark')
#     parser.add_argument('--study-name', type=str, required=True, 
#                         help='Name of the study to be created in neptune')
    
#     parser.add_argument('--optimizer', type=str, default="", choices=[opt.value for opt in OptimizerEnum],
#                         help='Name of the optimizer to be used')
#     parser.add_argument('--seed', type=int, default=0,
#                         help='Seed for reproducability')
    
    
#     parser.add_argument('--use-priors', action='store_true',
#                         help='Use prior for the sampling approach')
#     args = parser.parse_args()
    
#     pipeline_space = PipelineScalingConfig()

#     bench = get_benchmark()

#     # (min_params, min_data, min_flops), (max_params, max_data, max_flops) = get_macro_params_bounds(bench)
#     # print(f"Macro bounds - Params: ({min_params:e}, {max_params:e}), Data: ({min_data:e}, {max_data:e}), Flops: ({min_flops:e}, {max_flops:e})")
    
# def get_optimizer_openclip():
#     get_optimizer = get_optimizer_wrapper(
#         pipeline_space=pipeline_space,
#         evaluate_pipeline=evaluate_pipeline_wrapper(
#             bench, 
#             lc_aware=lc_aware,
#         ),
#         param_estimator=parameters_estimator_wrapper(bench),
#         seen_datapoints_estimator=seen_datapoints_estimator,
#         flops_estimator=flops_estimator_wrapper(bench),
#         aspect_ratio_fn=aspect_ratio_fn,
#         data_parameter_name="train_num_samples",
#         size_per_numerical_dimension={
#             "lr": 5,
#             "wd": 5,
#             "warmup_fraction": 5,
#             "beta1": 1,
#             "beta2": 1,
#             "eps": 1,
#             "vision_width": 10,
#             "text_width": 10,
#             "train_num_samples": 10,
#         } # type: ignore
#     )