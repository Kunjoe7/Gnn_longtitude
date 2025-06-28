# data_module.py
"""
Data module for TGB datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.linkproppred.evaluate import Evaluator as LinkEvaluator
from tgb.nodeproppred.evaluate import Evaluator as NodeEvaluator


class TGBData:
    """TGB Dataset wrapper"""
    
    def __init__(self, dataset_name, root='./data'):
        self.dataset_name = dataset_name
        self.root = root
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        if dataset_name.startswith('tgbl'):
            self.dataset = LinkPropPredDataset(name=dataset_name, root=root)
            self.task = 'link_prediction'
            self.evaluator = LinkEvaluator(name=dataset_name)
        elif dataset_name.startswith('tgbn'):
            self.dataset = NodePropPredDataset(name=dataset_name, root=root)
            self.task = 'node_classification'
            self.evaluator = NodeEvaluator(name=dataset_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Extract data
        self._extract_data()
        
    def _extract_data(self):
        """Extract data from dataset"""
        # Get data from full_data
        data = self.dataset.full_data
        
        # Basic arrays
        self.sources = data['sources']
        self.destinations = data['destinations']
        self.timestamps = data['timestamps']
        self.edge_feat = data.get('edge_feat', None)
        self.edge_label = data.get('edge_label', None)
        
        # Masks
        self.train_mask = self.dataset.train_mask
        self.val_mask = self.dataset.val_mask
        self.test_mask = self.dataset.test_mask
        
        # Dataset info
        self.num_nodes = self.dataset.num_nodes
        self.num_edges = self.dataset.num_edges
        self.edge_dim = self.edge_feat.shape[1] if self.edge_feat is not None else 0
        
        # Negative sampler
        self.negative_sampler = getattr(self.dataset, 'negative_sampler', None)
        
        # Print statistics
        self._print_stats()
        
    def _print_stats(self):
        """Print dataset statistics"""
        print(f"\nDataset: {self.dataset_name}")
        print(f"  Task: {self.task}")
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Edges: {self.num_edges:,}")
        print(f"  Edge features: {self.edge_dim}")
        print(f"  Train: {self.train_mask.sum():,} ({self.train_mask.sum()/self.num_edges*100:.1f}%)")
        print(f"  Val: {self.val_mask.sum():,} ({self.val_mask.sum()/self.num_edges*100:.1f}%)")
        print(f"  Test: {self.test_mask.sum():,} ({self.test_mask.sum()/self.num_edges*100:.1f}%)")
        
    def get_split_data(self, split='train'):
        """Get data for specific split"""
        if split == 'train':
            mask = self.train_mask
        elif split == 'val':
            mask = self.val_mask
        elif split == 'test':
            mask = self.test_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return {
            'sources': self.sources[mask],
            'destinations': self.destinations[mask],
            'timestamps': self.timestamps[mask],
            'edge_feat': self.edge_feat[mask] if self.edge_feat is not None else None,
            'edge_label': self.edge_label[mask] if self.edge_label is not None else None,
            'mask': mask
        }


class TemporalDataset(Dataset):
    """PyTorch dataset for temporal data"""
    
    def __init__(self, sources, destinations, timestamps, edge_feat=None, edge_label=None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_feat = edge_feat
        self.edge_label = edge_label
        self.length = len(sources)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = {
            'src': self.sources[idx],
            'dst': self.destinations[idx],
            'ts': self.timestamps[idx]
        }
        
        if self.edge_feat is not None:
            data['feat'] = self.edge_feat[idx]
            
        if self.edge_label is not None:
            data['label'] = self.edge_label[idx]
            
        return data


def get_data_loader(data_dict, batch_size, shuffle=True, num_workers=0):
    """Create data loader from data dictionary"""
    dataset = TemporalDataset(
        sources=data_dict['sources'],
        destinations=data_dict['destinations'],
        timestamps=data_dict['timestamps'],
        edge_feat=data_dict.get('edge_feat', None),
        edge_label=data_dict.get('edge_label', None)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


class NeighborSampler:
    """Temporal neighbor sampler"""
    
    def __init__(self, sources, destinations, timestamps):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        
        # Build adjacency list
        self.adj_list = {}
        for i, (src, dst, ts) in enumerate(zip(sources, destinations, timestamps)):
            if src not in self.adj_list:
                self.adj_list[src] = []
            if dst not in self.adj_list:
                self.adj_list[dst] = []
                
            self.adj_list[src].append((dst, ts, i))
            self.adj_list[dst].append((src, ts, i))
            
        # Sort by timestamp
        for node in self.adj_list:
            self.adj_list[node].sort(key=lambda x: x[1])
            
    def sample_neighbors(self, nodes, timestamps, num_neighbors=10):
        """Sample temporal neighbors"""
        neighbors = []
        edge_times = []
        edge_idxs = []
        
        for node, ts in zip(nodes, timestamps):
            node_neighbors = self.adj_list.get(node, [])
            # Get neighbors before timestamp
            valid_neighbors = [(n, t, idx) for n, t, idx in node_neighbors if t < ts]
            
            if len(valid_neighbors) > num_neighbors:
                # Sample most recent neighbors
                sampled = valid_neighbors[-num_neighbors:]
            else:
                sampled = valid_neighbors
                
            if sampled:
                neighbors.append([n for n, _, _ in sampled])
                edge_times.append([t for _, t, _ in sampled])
                edge_idxs.append([idx for _, _, idx in sampled])
            else:
                neighbors.append([])
                edge_times.append([])
                edge_idxs.append([])
                
        return neighbors, edge_times, edge_idxs