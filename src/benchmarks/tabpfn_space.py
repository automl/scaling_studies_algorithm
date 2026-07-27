import logging
import copy

import neps

from scan_benchmark.tabpfn.api import TabPFNBenchmark, TabPFNConfig, TabPFNTarget
from scan_benchmark.base_performance_benchmark import PerformancePredictorType

TABPFN_TARGETS = [
    TabPFNTarget.VAL_LOSS,
    # TabPFNTarget.NLL,
    # TabPFNTarget.ROC_AUC,
]

# Task-distribution parameters — held fixed for a given scaling analysis (not searched).
# They define the training/eval prior (TabICLPriorDataLoader), not model capacity, so
# they are injected as constants when a config is built rather than being HPs in the space.
MAX_FEATURES = 32
NUM_DATAPOINTS_MAX = 128


def get_benchmark(predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN) -> TabPFNBenchmark:
    return TabPFNBenchmark(target=TABPFN_TARGETS, predictor_type=predictor_type)


def normalize_tabpfn_config(raw_conf: dict) -> dict:
    raw_conf = copy.deepcopy(raw_conf)
    # Inject the fixed task-distribution constants; explicit values (if any) are preserved.
    raw_conf.setdefault("max_features", MAX_FEATURES)
    raw_conf.setdefault("num_datapoints_max", NUM_DATAPOINTS_MAX)
    raw_conf["total_cells"] = int(raw_conf["total_cells"])
    raw_conf["effective_batch_size"] = int(raw_conf["effective_batch_size"])
    raw_conf["embedding_size"] = int(raw_conf["embedding_size"])
    raw_conf["num_layers"] = int(raw_conf["num_layers"])
    raw_conf["max_features"] = int(raw_conf["max_features"])
    raw_conf["num_datapoints_max"] = int(raw_conf["num_datapoints_max"])
    return raw_conf

def parameters_estimator_wrapper(bench: TabPFNBenchmark):
    def parameters_estimator(**conf):
        conf = normalize_tabpfn_config(conf)
        config = TabPFNConfig(**conf)
        return bench.model_params(config)
    return parameters_estimator
    
def evaluate_pipeline_wrapper(bench: TabPFNBenchmark, lc_aware: bool = False):
    def evaluate_pipeline(lr, effective_batch_size, embedding_size, total_cells, num_layers):
        res = normalize_tabpfn_config(
            dict(
                lr=lr,
                effective_batch_size=effective_batch_size,
                embedding_size=embedding_size,
                total_cells=total_cells,
                num_layers=num_layers,
            )
        )
        total_cells = res["total_cells"]

        # max_features / num_datapoints_max injected as constants by normalize_tabpfn_config.
        config = TabPFNConfig(**res)
        lc = None
        if lc_aware:
            lc = []
            confs = []
            for i in range(4): # num_points in learning curve
                config.total_cells = int(total_cells * (i + 1) / 4)
                confs.append(copy.deepcopy(config))
            try:
                results = bench.query_many(confs)
            except Exception as e:
                results = [
                    {
                        "objective_to_minimize": float('nan'),
                        "model_stats": {}, 
                        "predictions": None,
                    },
                ]
            for i, result in enumerate(results):
                n_data = int(total_cells * (i + 1) / 4)
                flops = float(result.get("model_stats", {}).get("flops", float('nan')))
                objective = float(result.get("predictions", {}).get("mean", float('nan')))
                if objective is float('nan') or flops is float('nan'):
                    return {
                        "objective_to_minimize": float('nan'), 
                        "cost": flops,
                        "learning_curve": None,
                        "info_dict": {
                            "n_param": int(result.get("model_stats", {}).get("params", float('nan'))),
                            "flops": flops,
                            "n_data": n_data,
                            "predictions": result["predictions"],
                            "model_stats": result.get("model_stats", {}),
                        },
                    }
                lc.append((n_data, flops, objective))
        config.total_cells = total_cells
        result = bench.query(config)
        objective = float((result.get("predictions") or {}).get('mean', float('nan')))
        if objective is float('nan'):
            print(f"Benchmark query failed with config: {config} result: {result}")
        return {
            "objective_to_minimize": objective, 
            "cost": float(result["model_stats"]["flops"]), 
            "learning_curve": lc,
            "info_dict": {
                "n_param": int(result["model_stats"]["params"]),
                "flops": float(result["model_stats"]["flops"]),
                "n_data": total_cells,
                "predictions": result["predictions"],
                "model_stats": result["model_stats"],
            },
        }
    return evaluate_pipeline

def flops_estimator_wrapper(bench: TabPFNBenchmark):
    def flops_estimator(**conf):
        conf = normalize_tabpfn_config(conf)
        config = TabPFNConfig(**conf)
        return bench.flops(config)
    return flops_estimator

def aspect_ratio_fn(embedding_size, num_layers, **_) -> float:
    return embedding_size / num_layers

def seen_datapoints_estimator(total_cells, **_):
    return int(total_cells)

class PipelineScalingConfig(neps.PipelineSpace):
    # hps
    lr = neps.Float(
        lower=0.0001,
        upper=0.05,
        prior=0.005,
        prior_confidence='high',
        log=True
    )
    effective_batch_size = neps.Integer(
         lower=16,
         upper=256,
         log=True,
         prior=16,
         prior_confidence='high',
         log_base=2,
    )
    total_cells = neps.Integer(
        lower=1048576,
        upper=34359738368,
        log=True,
        log_base=2,
        is_scaling=True,
    )
    embedding_size = neps.Integer(
         lower=16,
         upper=256,
         log=True,
         log_base=2,
         is_scaling=True,
    )
    num_layers= neps.Integer(
        lower=2,
        upper=32,
        log=True,
        log_base=2,
        is_scaling=True,
    )
    # max_features= neps.Integer(
    #     lower=32,
    #     upper=128,
    #     prior=32,
    #     log=True,
    #     log_base=2,
    #     prior_confidence='high',
    # )
    # num_datapoints_max= neps.Integer(
    #     lower=128,
    #     upper=512,
    #     prior=128,
    #     log=True,
    #     log_base=2,
    #     prior_confidence='high',
    # )
   
logging.basicConfig(level=logging.INFO,)

def get_macro_params_bounds(bench: TabPFNBenchmark) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # max_features / num_datapoints_max are fixed constants injected at config creation.
    max_flop_conf = dict(
        lr=0.05,
        effective_batch_size=256,
        embedding_size=256,
        total_cells=34359738368,
        num_layers=32,
    )

    min_flop_conf = dict(
        lr=0.0001,
        effective_batch_size=16,
        embedding_size=16,
        total_cells=1048576,
        num_layers=2,
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

# 3.398452e+07 parameters, 3.435974e+10 data points, total flops: 7.060368e+18
# 1.060200e+04 parameters, 1.048576e+06 data points, total flops: 5.676123e+10

def get_pipeline_space():
    space = PipelineScalingConfig()
    return space

if __name__ == '__main__':
    # min_flop_conf = dict(
    #     lr=0.0001,
    #     effective_batch_size=64,
    #     embedding_size=256,
    #     total_cells=34359738368,
    #     num_layers=32,
    #     max_features=35,
    #     num_datapoints_max=236,
    # )

    bench = get_benchmark()
    # res = evaluate_pipeline_wrapper(bench)(
    #     **min_flop_conf
    # )
    # print(res)

    get_macro_params_bounds(bench)

