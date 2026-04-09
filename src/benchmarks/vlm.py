import numpy as np
import pandas as pd
from pathlib import Path


class ModelType(str):
    VIT_S_32 = "vit-s-32"
    VIT_M_32 = "vit-m-32"


# Model specifications (publicly available ViT specifications)
MODEL_SPECS = {
    ModelType.VIT_S_32: {
        "num_parameters": 22_863_752,  # ViT-Small parameters
        "flops_per_sample": 4.6e9,  # FLOPs per image (patch 32x32, 224x224 images)
        "csv_file": "vit_s_32.csv",
        "datapoints_seen": 1.28e6,
    },
    ModelType.VIT_M_32: {
        "num_parameters": 39_347_072,  # ViT-Medium parameters
        "flops_per_sample": 7.8e9,  # FLOPs per image (patch 32x32, 224x224 images)
        "csv_file": "vit_m_32.csv",
        "datapoints_seen": 1.28e6,
    }
}


class VLMBench:
    def __init__(self, 
                 model=ModelType.VIT_S_32,
                 data_dir=None):
        """
        VLM Benchmark Simulator
        
        Args:
            model (str): Model type - either "vit-s-32" or "vit-m-32"
            data_dir (str): Path to directory containing CSV files. 
                          Defaults to current directory.
        """
        self.model = model
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.df = self._load_data()
        
    def _load_data(self):
        """Load CSV data for the selected model."""
        csv_file = self.data_dir / MODEL_SPECS[self.model]["csv_file"]
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        return pd.read_csv(csv_file)
    
    def _find_matching_trial(self, conf_dict_unpacked):
        """
        Find the trial in the CSV that matches the given configuration.
        
        Config parameters to match:
        - grad_clip_norm (optional, for vit_s_32)
        - lr
        - warmup_fraction
        - wd
        
        Returns the first trial that matches, or None if no match found.
        """
        # Prepare filter conditions
        filters = []
        
        # grad_clip_norm is only in vit_s_32
        if "grad_clip_norm" in conf_dict_unpacked and self.model == ModelType.VIT_S_32:
            filters.append(
                np.isclose(self.df["grad_clip_norm"], 
                          conf_dict_unpacked["grad_clip_norm"], 
                          rtol=1e-5)
            )
        
        # Match lr (learning rate)
        if "lr" in conf_dict_unpacked:
            filters.append(
                np.isclose(self.df["lr"], 
                          conf_dict_unpacked["lr"], 
                          rtol=1e-5)
            )
        
        # Match warmup_fraction
        if "warmup_fraction" in conf_dict_unpacked:
            filters.append(
                np.isclose(self.df["warmup_fraction"], 
                          conf_dict_unpacked["warmup_fraction"], 
                          rtol=1e-5)
            )
        
        # Match wd (weight decay)
        if "wd" in conf_dict_unpacked:
            filters.append(
                np.isclose(self.df["wd"], 
                          conf_dict_unpacked["wd"], 
                          rtol=1e-5)
            )
        
        # Apply all filters
        if filters:
            mask = np.all(filters, axis=0) if filters else np.ones(len(self.df), dtype=bool)
        else:
            # If no matching config params, return first row
            mask = np.ones(len(self.df), dtype=bool)
        
        matching_rows = self.df[mask]
        
        if len(matching_rows) > 0:
            return matching_rows.iloc[0]
        else:
            # Return first row as fallback
            return self.df.iloc[0]
    
    def get_flops_count(self, conf_dict_unpacked):
        """
        Get the total FLOPs for training with the given configuration.
        
        Estimates FLOPs based on model architecture and assumes:
        - Standard training protocol from the CSV data
        - Number of samples seen can be estimated from datapoints_seen
        
        Args:
            conf_dict_unpacked (dict): Configuration dictionary with keys like
                                      'lr', 'warmup_fraction', 'wd', 'grad_clip_norm'
        
        Returns:
            float: Total FLOPs for the training run
        """
        # Get model specs
        flops_per_sample = MODEL_SPECS[self.model]["flops_per_sample"]
        
        # Find matching trial to get training stats
        trial = self._find_matching_trial(conf_dict_unpacked)
        
        # Estimate number of samples seen (assuming standard training)
        # Using loss as a proxy - better performance = more training
        # Typical training: assume ~100 epochs on ~50k samples (ImageNet scale)
        num_samples_seen = self._estimate_samples_seen(trial)
        
        # Total FLOPs = FLOPs per sample * number of samples
        total_flops = flops_per_sample * num_samples_seen
        
        return int(total_flops)
    
    def get_datapoint_seen(self, conf_dict_unpacked):
        """
        Get the number of datapoints (samples) seen during training.
        
        Args:
            conf_dict_unpacked (dict): Configuration dictionary
        
        Returns:
            int: Number of datapoints seen
        """
        trial = self._find_matching_trial(conf_dict_unpacked)
        return self._estimate_samples_seen(trial)
    
    def get_num_parameters(self, conf_dict_unpacked):
        """
        Get the total number of parameters in the model.
        
        Note: The parameters do not depend on the configuration for ViT models,
        as the architecture is fixed.
        
        Args:
            conf_dict_unpacked (dict): Configuration dictionary (not used for ViT)
        
        Returns:
            int: Total number of model parameters
        """
        return MODEL_SPECS[self.model]["num_parameters"]
    
    def _estimate_samples_seen(self, trial_row):
        """
        Estimate the number of training samples seen based on trial metrics.
        
        Uses the loss value as an indicator of training progress.
        Lower loss generally correlates with more training and thus more samples seen.
        
        Assumes: ~100 epochs on ~50k-100k samples (standard training protocol)
        """
        return MODEL_SPECS[self.model]["datapoints_seen"]
        # train_loss = trial_row["train_loss"]
        
        # # Inverse relationship: lower loss = more training
        # # Normalize loss to estimate training progress
        # # Loss range typically 0-7 for these experiments
        
        # # Estimate based on loss: assume minimum loss ~3.2 (good training) 
        # # and maximum loss ~6.3 (poor/early training)
        # min_loss = 3.2
        # max_loss = 6.3
        
        # # Clamp loss to range
        # loss_clamped = max(min_loss, min(max_loss, train_loss))
        
        # # Normalize to [0, 1]
        # progress = 1.0 - (loss_clamped - min_loss) / (max_loss - min_loss)
        
        # # Estimate total datapoints: ~100 epochs * 50k samples (standard ImageNet-scale)
        # # = 5 million samples for full training
        # max_datapoints = 5_000_000
        
        # # Scale based on progress
        # estimated_datapoints = max(100_000, int(max_datapoints * progress))
        
        # return estimated_datapoints
    
    def query(self, conf_dict_unpacked):
        """
        Query benchmark results for a given configuration.
        
        Returns metrics from the matching trial in the CSV.
        
        Args:
            conf_dict_unpacked (dict): Configuration dictionary with keys like
                                      'lr', 'warmup_fraction', 'wd', 'grad_clip_norm'
        
        Returns:
            dict: Dictionary with keys:
                - 'flops': Total FLOPs
                - 'datapoints_seen': Number of datapoints seen
                - 'num_parameters': Number of model parameters
                - 'val_loss': Validation loss from CSV
                - 'test_loss': Test loss from CSV
                - 'train_loss': Training loss from CSV
                - 'imagenet1k_acc1': ImageNet-1K top-1 accuracy
                - 'cifar10_acc1': CIFAR-10 accuracy
                - 'mscoco_image_recall@5': MSCOCO image recall@5
                - 'mscoco_text_recall@5': MSCOCO text recall@5
        """
        trial = self._find_matching_trial(conf_dict_unpacked)
        
        result = {
            'flops': self.get_flops_count(conf_dict_unpacked),
            'datapoints_seen': self.get_datapoint_seen(conf_dict_unpacked),
            'num_parameters': self.get_num_parameters(conf_dict_unpacked),
            'val_loss': trial['val_loss'],
            'test_loss': trial['test_loss'],
            'train_loss': trial['train_loss'],
            'imagenet1k_acc1': trial['imagenet1k_acc1'],
            'cifar10_acc1': trial['cifar10_acc1'],
            'mscoco_image_recall@5': trial['mscoco_image_recall@5'],
            'mscoco_text_recall@5': trial['mscoco_text_recall@5'],
        }
        
        return result
        

