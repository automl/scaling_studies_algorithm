# query the benchmark for random pipelines
from functools import partial
from pathlib import Path
from src.benchmarks.lcbench import mfpbench_benches
from typing import Iterator, Mapping, Any
from hpoglue.benchmark import BenchmarkDescription
from src.benchmarks.lcbench import estimate_lcbench_flops
from hpoglue import Benchmark
from collections import defaultdict
import numpy as np
import csv
from dataclasses import asdict
import json
from src.benchmarks.lcbench import Objective


objectives = [
            Objective.VAL_ACCURACY, 
            # Objective.TIME, 
            Objective.VAL_CROSS_ENTROPY, 
            Objective.VAL_BALANCED_ACCURACY,
        ]
def get_flop_config_map(bench: Benchmark, flops_bins: list[float], max_samples: int = 100) -> None:
    # randomly select a benchmark and get its flop estimate for a sample config
    flop_config_map = defaultdict(list)
    rows = []
    bin_to_configs = defaultdict(list)
    header_written = False
    

    with open(f"output-{bench.name}.csv", "a", newline="") as f:
        writer = None
        for i in range(max_samples):
            sample = bench.benchmark.sample()
            flop_count = estimate_lcbench_flops(bench=bench, config=sample, epoch=52)  # max epoch in lcbench
            flop_config_map[flop_count] = sample
            flop_bin = find_flop_bin(flop_count, flops_bins)
            rows.append({"flops":flop_count, "flop_bin": flop_bin,  "task": bench.name, "config": asdict(sample)})
            if flop_bin  in bin_to_configs and len(bin_to_configs[flop_bin]) < 100:
                bin_to_configs[flop_bin].append(sample)
            else:
                continue
        if rows:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                if not header_written:
                    writer.writeheader()
            writer.writerows(rows)

def create_priors(
    bench: Benchmark,
):
    prior_centers: dict[str, dict[str, float]] = defaultdict(dict)
    prior_confidences: dict[str, dict[str, float]] = defaultdict(dict)
    # distribute configs into flop bins
    bin_to_configs = defaultdict(list)
    with open(f"output-{bench.name}.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    epoch = 52  # max epoch in lcbench
    eval_func = lambda conf: bench.query(
        objectives=objectives, 
        epoch=epoch, **conf
    )
    flop_conf_map = defaultdict(list)
    for row in rows:
        t = row["config"]
        t_fixed = t.replace("'", '"')
        conf = json.loads(t_fixed)
        flop_conf_map[row["flop_bin"]].append({"config": conf, "result": eval_func(conf)})
    
    for flop_bin, conf_results in flop_conf_map.items():
        # find best configs (pareto front) based on val_accuracy
        # for i, objective_name in enumerate(objectives):
        best_conf = None
        best_obj = -np.inf
        for conf_result in conf_results:
            # for each objective fine the best config
                obj = conf_result["result"][0]
                if obj > best_obj:
                    best_obj = obj
                best_conf = conf_result["config"]
        if best_conf is not None:
            prior_centers[flop_bin] = best_conf
            # --- CALCULATE CONFIDENCE ---
            
            all_scores = [entry["result"][0] for entry in conf_results]
            
            # If we look at ALL scores, bad configs will skew the variance.
            all_scores.sort(reverse=True)
            n_top = max(3, int(len(all_scores) * 0.2))
            top_scores = all_scores[:n_top]
            
            if len(top_scores) < 2:
                confidence = 0.1 
            else:
                mean_score = np.mean(top_scores)
                std_score = np.std(top_scores)
                
                # Lower CV means the top scores are tightly clustered (High Confidence)
                cv = std_score / (abs(mean_score) + 1e-6)
                
                # 4. Convert CV to a 0-1 Confidence Score
                #    If CV is 0 (perfect stability) -> Confidence = 1.0
                #    If CV is high (e.g., > 0.5) -> Confidence drops
                #    Using an exponential decay function is standard for this mapping.
                confidence = np.exp(-5.0 * cv)
            
            prior_confidences[flop_bin] = float(confidence)
    # write priors to yaml files
    import yaml
    priors_dir = Path("data/priors")
    priors_dir.mkdir(parents=True, exist_ok=True)
    for flop_bin in prior_centers.keys():
        prior_path = priors_dir / f"{bench.name}_flopbin_{flop_bin}_prior.yaml"
        prior_data = {
            "benchmark": bench.name,
            "centers": prior_centers[flop_bin],
            "confidences": prior_confidences[flop_bin],
            "objective": objectives[0],
        }
        with prior_path.open("w") as file:
            yaml.dump(prior_data, file)
    return prior_centers, prior_confidences

def _get_flop_bins():
    # define flop bins from 1e11 to 1e15 with logarithmic spacing
    flops_bins = np.logspace(9, 15, num=100)
    return flops_bins

def find_flop_bin(flop_count: float, flops_bins: range) -> int:
    if flop_count >= flops_bins[-1]:
        return flops_bins[len(flops_bins) - 1]
    if flop_count <= flops_bins[0]:
        return flops_bins[0]
    for i in range(len(flops_bins) - 1):
        if flops_bins[i] <= flop_count < flops_bins[i + 1]:
            return flops_bins[i]
    return -1

if __name__ == "__main__":
    flops_bins = _get_flop_bins()
    for bench_name, bench_desc in mfpbench_benches.items():
        bench = bench_desc.load(bench_desc)
        # get_flop_config_map(bench, flops_bins, max_samples=10000)
        create_priors(bench)
        break
