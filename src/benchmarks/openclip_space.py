import logging
import copy

import neps

from scan_bench import VLMBenchmark, VLMConfig, VLMTarget, PerformancePredictorType


WIDTH_CHOICES = [32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024]

OPENCLIP_TARGETS = [
    VLMTarget.VAL_LOSS, 
]
OPENCLIP_DOWN_TARGET = VLMTarget.IMAGENET1K
AVAILABLE_TARGETS = [
    VLMTarget.VAL_LOSS,
    VLMTarget.TEST_LOSS,
    VLMTarget.RETRIEVAL_MSCOCO, 
    VLMTarget.CIFAR10, 
    VLMTarget.VTAB_CIFAR100, 
    VLMTarget.IMAGENET1K,
    VLMTarget.MNIST,
    VLMTarget.FAIRNESS_DOLLAR_STREET,
]
def get_benchmark(predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN,
                 target: VLMTarget | None = None,
                ) -> VLMBenchmark:
    if target is None:
        target = OPENCLIP_TARGETS[0]
    return VLMBenchmark(target=target, predictor_type=predictor_type)


def normalize_openclip_config(raw_conf: dict) -> dict:
    """Snap config values to the nearest valid values in the OpenCLIP search space.

    - Integer dimensions are rounded to nearest valid integer and clamped.
    - Float dimensions are clamped to their valid domain bounds.
    - `vision_width`/`text_width` are ordinal indices into WIDTH_CHOICES and are
      resolved to their actual widths here.
    """
    conf = dict(raw_conf)

    float_bounds = {
        "lr": (1.0e-5, 5.0e-2),
        "wd": (1.0e-6, 2.0e-1),
        "warmup_fraction": (0.0, 0.75),
        "beta1": (0.9, 0.99),
        "beta2": (0.95, 0.999),
        "eps": (1.0e-8, 1.0e-6),
    }
    int_bounds = {
        "vision_width": (0, len(WIDTH_CHOICES) - 1),
        "text_width": (0, len(WIDTH_CHOICES) - 1),
        "train_num_samples": (6e5, 2e8),
    }
    ordinal_choices = {
        "vision_width": WIDTH_CHOICES,
        "text_width": WIDTH_CHOICES,
    }

    for name, value in list(conf.items()):
        if name in int_bounds:
            lower_i, upper_i = int_bounds[name]
            snapped = int(round(float(value)))
            snapped = min(max(snapped, lower_i), upper_i)
            conf[name] = ordinal_choices[name][snapped] if name in ordinal_choices else snapped
        elif name in float_bounds:
            lower_f, upper_f = float_bounds[name]
            v = float(value)
            conf[name] = min(max(v, lower_f), upper_f)

    return conf

def parameters_estimator_wrapper(bench: VLMBenchmark):
    def parameters_estimator(**conf):
        conf = normalize_openclip_config(conf)
        conf["total_samples_planned"] = int(conf["train_num_samples"])
        conf["training_progress"] = 1.0
        conf.pop("train_num_samples")
        config = VLMConfig(**conf)
        stats = bench.get_model_stats(config)
        return stats["params"]

    return parameters_estimator
    
def build_config(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples) -> VLMConfig:
    """Snap raw search-space values and turn them into a fully trained `VLMConfig`."""
    normalized_conf = normalize_openclip_config({
        "lr": lr,
        "wd": wd,
        "beta1": beta1,
        "beta2": beta2,
        "warmup_fraction": warmup_fraction,
        "eps": eps,
        "vision_width": vision_width,
        "text_width": text_width,
        "train_num_samples": train_num_samples,
    })

    return VLMConfig(
        lr=round(normalized_conf["lr"], 7),
        wd=round(normalized_conf["wd"], 7),
        beta1=round(normalized_conf["beta1"], 7),
        beta2=round(normalized_conf["beta2"], 7),
        warmup_fraction=round(normalized_conf["warmup_fraction"], 7),
        eps=round(normalized_conf["eps"], 10),
        vision_width=normalized_conf["vision_width"],
        text_width=normalized_conf["text_width"],
        total_samples_planned=int(normalized_conf["train_num_samples"]),
        training_progress=1.0
    )


def build_info_dict(result: dict, config: VLMConfig) -> dict:
    """The `info_dict` neps stores as `report.yaml:extra` for a single benchmark query."""
    return {
        "n_param": int(result["model_stats"]["params"]),
        "flops": float(result["model_stats"]["total_flops"]),
        "n_data": int(config.total_samples_planned),
        "predictions": result["predictions"],
        "model_stats": result["model_stats"],
    }


def query_wrapper(bench: VLMBenchmark):
    """Query the surrogate for one config and return its `info_dict`.

    Shared by `evaluate_pipeline` and by post-hoc evaluation of stored configs
    (`src/posthoc_downstream_eval.py`), which only needs the predictions.
    """
    def query(config: VLMConfig | None = None, **raw_conf) -> dict:
        if config is None:
            config = build_config(**raw_conf)
        config.training_progress = 1.0  # for double caution
        result = bench.query(config)
        if result["failed"]:
            print(f"Benchmark query failed with config: {config}")
        return build_info_dict(result, config)

    return query


def evaluate_pipeline_wrapper(bench: VLMBenchmark, lc_aware: bool = False):
    query = query_wrapper(bench)

    def evaluate_pipeline(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples):
        config = build_config(
            lr=lr,
            wd=wd,
            beta1=beta1,
            beta2=beta2,
            warmup_fraction=warmup_fraction,
            eps=eps,
            vision_width=vision_width,
            text_width=text_width,
            train_num_samples=train_num_samples,
        )
        train_num_samples = config.total_samples_planned
        lc = None
        logging.info(f"lc_aware option is set to {lc_aware}")
        if lc_aware:
            lc = []
            confs = []
            for i in range(1, 4 + 1): # num_points in learning curve
                config.training_progress = i / 4
                confs.append(copy.deepcopy(config))
            try:
                results = bench.query_many(confs)
            except Exception as e:

                config.training_progress = 1.0
                stats = bench.get_model_stats(config)
                results = [{"failed": True, "model_stats": stats, "predictions": None}]
            for result in results:
                if result["failed"]:
                    return {
                        "objective_to_minimize": float('nan'),
                        "cost": float(result["model_stats"]["total_flops"]),
                        "learning_curve": None,
                        "info_dict": build_info_dict(result, config),
                    }
                n_data = config.training_progress * train_num_samples # num tokens seen so far
                flops = float(result["model_stats"]["total_flops"])
                lc.append((n_data, flops, float(result["predictions"]["mean"])))
        info_dict = query(config=config)
        return {
            "objective_to_minimize": float((info_dict["predictions"] or {}).get("mean", float('nan'))),
            "cost": info_dict["flops"],
            "learning_curve": lc,
            "info_dict": info_dict,
        }
    return evaluate_pipeline

def flops_estimator_wrapper(bench: VLMBenchmark):
    def flops_estimator(**conf):
        conf = normalize_openclip_config(conf)
        conf["total_samples_planned"] = int(conf["train_num_samples"])
        conf["training_progress"] = 1.0
        conf.pop("train_num_samples")
        config = VLMConfig(**conf)
        
        stats = bench.get_model_stats(config)
        return stats["total_flops"]

    return flops_estimator

def aspect_ratio_fn(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples) -> float:
    normalized_conf = normalize_openclip_config({
        "vision_width": vision_width,
        "text_width": text_width,
    })
    vision_width = normalized_conf["vision_width"]
    text_width = normalized_conf["text_width"]
    vision_ratio = vision_width  / 12 # constant n_layers
    text_ratio = text_width  / 12 # constant n_layers
    return vision_ratio / text_ratio

def seen_datapoints_estimator(lr, wd, beta1, beta2, warmup_fraction, eps, vision_width, text_width, train_num_samples):
    return int(train_num_samples)

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
    train_num_samples = neps.Integer(
        lower=int(6e5),
        upper=int(2e8),
        log=True,
        is_scaling=True,
    )
    vision_width = neps.Integer(
        lower=0, upper=len(WIDTH_CHOICES) - 1,  # ordinal index into WIDTH_CHOICES
        is_scaling=True,
    )
    text_width = neps.Integer(
        lower=0, upper=len(WIDTH_CHOICES) - 1,  # ordinal index into WIDTH_CHOICES
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
        vision_width=10,
        text_width=10,
        train_num_samples=int(200 * 1e6),
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
        train_num_samples=int(0.6 * 1e6),
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

# 3.569725e+08 parameters, 2.000000e+08 data points, total flops: 7.815374e+18
# 2.013121e+06 parameters, 6.000000e+05 data points, total flops: 3.605023e+13
def get_pipeline_space():
    space = PipelineScalingConfig()
    return space
    
if __name__ == '__main__':

    bench = get_benchmark()
    get_macro_params_bounds(bench)