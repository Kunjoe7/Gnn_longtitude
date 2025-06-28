# config.py
"""
Configuration file for TGB experiments
"""

import os
import torch


class Config:
    """Configuration class for TGB experiments"""
    
    def __init__(self):
        # Data settings
        self.dataset = 'tgbl-wiki'  # tgbl-wiki, tgbl-review, tgbl-coin, etc.
        self.data_root = './data'
        
        # Model settings
        self.model_name = 'tgn'  # tgn, dyrep, jodie, gat, sage
        self.node_dim = 100
        self.time_dim = 100
        self.edge_dim = 172  # Will be set based on dataset
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.num_neighbors = 10
        
        # Memory settings (for memory-based models)
        self.memory_dim = 100
        self.memory_update = 'gru'  # gru, rnn, transformer
        
        # Training settings
        self.epochs = 50
        self.batch_size = 200
        self.lr = 0.001
        self.weight_decay = 0.0
        self.neg_sample_ratio = 1
        self.patience = 10
        self.min_epochs = 20
        
        # System settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.seed = 42
        
        # Logging settings
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoints'
        self.result_dir = './results'
        self.save_interval = 10
        self.verbose = True
        
        # Create directories
        self._create_dirs()
    
    def _create_dirs(self):
        """Create necessary directories"""
        dirs = [self.data_root, self.log_dir, self.checkpoint_dir, self.result_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Config has no attribute {k}")
    
    def __str__(self):
        """String representation of config"""
        lines = ["Configuration:"]
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        return '\n'.join(lines)