from benchmarks.synthetic_transformer_space import get_total_flops, get_number_of_parameters, seen_datapoints_estimator, optimizers, evaluate_pipeline, pipeline_space, MAX_TAGET_FLOPS

conf_list = [{'d_model': 148, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 171, 'num_layers': 6, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 179, 'num_layers': 3, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 117, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 125, 'num_layers': 6, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 156, 'num_layers': 4, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 179, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 187, 'num_layers': 6, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 218, 'num_layers': 4, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 78, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 86, 'num_layers': 6, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 101, 'num_layers': 4, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 194, 'num_layers': 10, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 218, 'num_layers': 8, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 233, 'num_layers': 6, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 218, 'num_layers': 11, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 225, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 241, 'num_layers': 8, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 225, 'num_layers': 11, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 241, 'num_layers': 10, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 249, 'num_layers': 9, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 32, 'num_layers': 7, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 32, 'num_layers': 5, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 32, 'num_layers': 4, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}, {'d_model': 241, 'num_layers': 12, 'batch_size': 8096, 'learning_rate': 0.04195746284425259, 'num_heads': 7, 'vocab_size': 8741}]
# check for duplicates
seen = set()
for conf in conf_list:
    conf_tuple = tuple(sorted(conf.items()))
    if conf_tuple in seen:
        print("Duplicate found:", conf)
    else:
        seen.add(conf_tuple)

# flop count
total_flops = 0
for conf in conf_list:
    flops = get_total_flops(**conf)
    total_flops += flops
    print(f"Config: {conf}, Flops: {flops:e}")
print(len(conf_list))
print(f"Total Flops: {total_flops:e}")
