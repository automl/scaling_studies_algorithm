import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class ScalingParameter:
    hidden_size: int
    num_layers: int
    vocab_size: int = 50257
    seq_len: int = 2048

@dataclass
class RegularHyperparameters:
    learning_rate: float
    global_batch_size: int
    num_training_steps: int

@dataclass
class BenchmarkConfig:
    scaling_params: ScalingParameter
    regular_hyperparams: RegularHyperparameters
    description: str = ""

def get_param_count(d_model, vocab_size, num_layers):
    return (12 * num_layers * d_model**2) + (vocab_size * d_model)

def get_total_flops(params, total_tokens_processed):
    total_flops_consumed = 6 * params * total_tokens_processed
    return total_flops_consumed
def calculate_metrics(config):
    """
    Calculates the 'cost_flops' and 'loss' for a given
    model and training configuration.
    """
    
    sp = config['scaling_params']
    hp = config['regular_hyperparams']
    
    L = sp['num_layers']
    d_model = sp['hidden_size']
    V = sp['vocab_size']
    seq_len = sp['seq_len']
    
    lr = hp['learning_rate']
    global_batch_size = hp['global_batch_size']
    num_training_steps = hp['num_training_steps']
    
   
    params = get_param_count(d_model, V, L)
    
    # cost_flops_per_seq = 6 * params * seq_len # 6 N * seq_len
    total_tokens_processed = global_batch_size * num_training_steps * seq_len

    total_flops_consumed = get_total_flops(
        params, total_tokens_processed)

    
    E = 1.7 # Simulate data noise
    # A / (N^a)
    A = 500 
    alpha = 0.09
    model_loss_term = A / (params**alpha)
    
    # B / (D^b)
    B = 800
    beta = 0.15
    data_loss_term = B / (total_tokens_processed**beta)
    
    # HPO Term (Penalty for bad learning rate)
    # Simulate a "U-shape" around an optimal LR.
    optimal_lr = 1e-4
    # This term is 1.0 at optimal_lr and grows quadratically
    lr_penalty = 1.0 + (np.log10(lr) - np.log10(optimal_lr))**2 * 0.5

    final_loss = (E + model_loss_term + data_loss_term) * lr_penalty
    
    return {
        "params": params,
        # "cost_flops_per_seq": cost_flops_per_seq,
        "total_flops_consumed": total_flops_consumed,
        "loss": final_loss
    }


benchmark_data = [
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=12, hidden_size=768),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-4,
            global_batch_size=512,
            num_training_steps=100000
        ),
        description="Baseline Small Model"
    ),
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=12, hidden_size=768),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-3,  # Too high
            global_batch_size=512,
            num_training_steps=100000
        ),
        description="Small Model (Bad LR)"
    ),
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=12, hidden_size=768),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-4,
            global_batch_size=512,
            num_training_steps=200000 # More steps
        ),
        description="Small Model (Longer Train)"
    ),
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=24, hidden_size=1024),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-4,
            global_batch_size=512,
            num_training_steps=100000
        ),
        description="Medium Model"
    ),
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=24, hidden_size=1024),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-4,
            global_batch_size=1024, # Bigger batch
            num_training_steps=100000
        ),
        description="Medium Model (Bigger Batch)"
    ),
    BenchmarkConfig(
        scaling_params=ScalingParameter(num_layers=24, hidden_size=1024),
        regular_hyperparams=RegularHyperparameters(
            learning_rate=1e-4,
            global_batch_size=1024, # Bigger batch
            num_training_steps=100000
        ),
        description="Medium Model (Bigger Batch)"
    )
]



print("--- Mock LLM Benchmark Results ---")
for i, config in enumerate(benchmark_data):
    config = asdict(config)
    metrics = calculate_metrics(config)
    
    print(f"\n--- Config {i+1} ---")
    print(f"  Config data: {config}")
    print(f"  Results:")
    print(f"    Params: {metrics['params'] / 1e6:.2f} M")
    # print(f"    Cost (FLOPs/seq): {metrics['cost_flops_per_seq'] / 1e9:.2f} GFLOPs")
    print(f"    Total (FLOPs consumed): {metrics['total_flops_consumed'] / 1e18:.2f} EFLOPs")
    print(f"    Final Loss: {metrics['loss']:.4f}")