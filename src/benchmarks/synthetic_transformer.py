import numpy as np

class ChinchillaSyntheticBench:
    def __init__(self, 
                 alpha=0.34, beta=0.28, 
                 E=1.69, A=406.4, B=410.7, 
                 noise_std=0.0):
        """
        Simulates the Chinchilla Scaling Law.
        L(N, D) = E + A/N^alpha + B/D^beta
        """
        self.alpha = alpha
        self.beta = beta
        self.E = E
        self.A = A
        self.B = B
        self.noise_std = noise_std

    def query(self, N, D, num_points=100, lr_penalty_factor=0.1):
        tokens = np.linspace(max(1, D * 0.01), D, num_points, dtype=np.int64)

        term_N = self.A / (N ** self.alpha)
        term_D = self.B / (D ** self.beta)
        
        # Learning curve: loss decreases with tokens following power-law (Chinchilla scaling)
        # L(tokens) = E + A/N^alpha + B/tokens^beta
        # This naturally shows exponential decay as tokens increase during training
        token_term = self.B / (tokens ** self.beta)
        curve_loss = self.E + term_N + token_term
        
        if self.noise_std > 0:
            scale = curve_loss * self.noise_std
            noise = np.random.normal(0, scale, size=num_points)
            
            # Simulating bad-data-batch spikes 5% of the time across the curve
            spike_mask = np.random.rand(num_points) < 0.05
            spikes = np.random.exponential(scale * 5, size=num_points) * spike_mask
            
            curve_loss += noise + spikes
            
        flops = 6 * N * tokens

        return {
            'flops': flops[-1],
            'learning_curve': list(zip(tokens, flops, curve_loss)),
            'loss': curve_loss[-1]
        }
