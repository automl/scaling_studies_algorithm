# from benchmarks.synthetic_transformer_space import get_total_flops, get_number_of_parameters, seen_datapoints_estimator, optimizers, evaluate_pipeline, pipeline_space, MAX_TAGET_FLOPS

# from src.benchmarks.llm_space import flops_estimator, parameters_estimator, seen_datapoints_estimator, aspect_ratio_fn
from src.benchmarks.configs import get_benchmark, BenchmarkEnum

conf_list = [{'embedding_size': 16, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 2120400}, {'embedding_size': 16, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 4240800}, {'embedding_size': 16, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 8481600}, {'embedding_size': 16, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 33926400}, {'embedding_size': 32, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 7925200}, {'embedding_size': 32, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 15850400}, {'embedding_size': 32, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 31700800}, {'embedding_size': 32, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 126803200}, {'embedding_size': 64, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 30594000}, {'embedding_size': 64, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 61188000}, {'embedding_size': 64, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 122376000}, {'embedding_size': 64, 'num_layers': 2, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 489504000}, {'embedding_size': 64, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 57294800}, {'embedding_size': 64, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 114589600}, {'embedding_size': 64, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 229179200}, {'embedding_size': 64, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 916716800}, {'embedding_size': 128, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 225998800}, {'embedding_size': 128, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 451997600}, {'embedding_size': 128, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 903995200}, {'embedding_size': 256, 'num_layers': 4, 'effective_batch_size': 64, 'lr': 0.023406639648053677, 'max_features': 35, 'num_datapoints_max': 236, 'total_cells': 897640400}]

seen = set()
for conf in conf_list:
    conf_tuple = tuple(sorted(conf.items()))
    if conf_tuple in seen:
        print("Duplicate found:", conf)
    else:
        seen.add(conf_tuple)

bench = get_benchmark(BenchmarkEnum.TABPFN, lc_aware=False)
# # flop count
total_flops = 0
for conf in conf_list:
    flops = bench.flops_estimator(**conf)
    params = bench.params_estimator(**conf)
    total_flops += flops
    print(f"Config: {conf} Flops: {flops:e}, Parameters: {params:e} Data {bench.seen_datapoints_estimator(**conf):e}")
print(len(conf_list))
print(f"Total Flops: {total_flops:e}")
