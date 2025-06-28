# models.py
"""
Temporal GNN models for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEncoder(nn.Module):
    """Time encoding module"""
    
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)))
            .float().reshape(self.time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim).float())
        
    def forward(self, t):
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output


class MergeLayer(nn.Module):
    """Merge different embeddings"""
    
    def __init__(self, dim1, dim2, dim3, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim1 + dim2 + dim3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        return self.fc(x)


class TGN(nn.Module):
    """Temporal Graph Network"""
    
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, memory_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        
        # Memory
        self.memory = nn.Parameter(torch.zeros((num_nodes, memory_dim)))
        self.memory_update = nn.GRUCell(memory_dim + edge_dim + time_dim, memory_dim)
        self.memory_attention = nn.MultiheadAttention(memory_dim, num_heads=4, dropout=dropout)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Edge encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        
        # Message function
        self.message_function = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim)
        )
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(node_dim * 2 + memory_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Get embeddings
        src_embed = self.node_embedding(src)
        dst_embed = self.node_embedding(dst)
        
        # Get memory
        src_memory = self.memory[src]
        dst_memory = self.memory[dst]
        
        # Time encoding
        time_embed = self.time_encoder(ts)
        
        # Edge encoding
        if edge_feat is not None and self.edge_encoder is not None:
            edge_embed = self.edge_encoder(edge_feat)
        else:
            edge_embed = torch.zeros(len(src), self.node_dim).to(src.device)
        
        # Predict
        h = torch.cat([src_embed, dst_embed, src_memory, dst_memory], dim=1)
        return self.out(h).squeeze()
    
    def update_memory(self, src, dst, ts, edge_feat=None):
        
        with torch.no_grad():
            # Get embeddings
            src_embed = self.node_embedding(src)
            dst_embed = self.node_embedding(dst)
            
            # Time encoding
            time_embed = self.time_encoder(ts)
            
            # Edge encoding - FIX: Use correct dimensions
            if edge_feat is not None and self.edge_encoder is not None:
                edge_embed = self.edge_encoder(edge_feat)
                edge_dim = edge_embed.shape[1]  # This will be hidden_dim
            else:
                # Use the same dimension as what edge_encoder would produce
                edge_dim = self.edge_encoder.out_features if self.edge_encoder else edge_feat.shape[1] if edge_feat is not None else 0
                edge_embed = torch.zeros(len(src), edge_dim).to(src.device)
            
            # Compute messages
            src_message = self.message_function(
                torch.cat([src_embed, dst_embed, edge_embed, time_embed], dim=1)
            )
            dst_message = self.message_function(
                torch.cat([dst_embed, src_embed, edge_embed, time_embed], dim=1)
            )
            
            # Update memory - FIX: Handle edge dimensions properly
            if edge_feat is not None:
                # Use raw edge features for memory update, not processed edge_embed
                memory_input_src = torch.cat([self.memory[src], edge_feat, time_embed], dim=1)
                memory_input_dst = torch.cat([self.memory[dst], edge_feat, time_embed], dim=1)
            else:
                # If no edge features, use zeros with correct dimension
                zero_edge_feat = torch.zeros(len(src), 172).to(src.device)  # 172 from your dataset
                memory_input_src = torch.cat([self.memory[src], zero_edge_feat, time_embed], dim=1)
                memory_input_dst = torch.cat([self.memory[dst], zero_edge_feat, time_embed], dim=1)
            
            self.memory[src] = self.memory_update(memory_input_src, self.memory[src])
            self.memory[dst] = self.memory_update(memory_input_dst, self.memory[dst])

class DyRep(nn.Module):
    """DyRep model"""
    
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Dynamics
        self.dynamics = nn.LSTM(
            node_dim * 2 + time_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Get embeddings
        src_embed = self.node_embedding(src)
        dst_embed = self.node_embedding(dst)
        
        # Time encoding
        time_embed = self.time_encoder(ts)
        
        # Combine
        h = torch.cat([src_embed, dst_embed, time_embed], dim=1)
        h = h.unsqueeze(1)  # Add sequence dimension
        
        # Apply dynamics
        out, _ = self.dynamics(h)
        out = out.squeeze(1)
        
        return self.out(out).squeeze()


class JODIE(nn.Module):
    """JODIE model"""
    
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Static embeddings
        self.static_embedding = nn.Embedding(num_nodes, node_dim)
        
        # Dynamic embeddings
        self.dynamic_embedding = nn.Parameter(torch.zeros(num_nodes, node_dim))
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2 + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Projection network
        self.project_net = nn.Sequential(
            nn.Linear(node_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(node_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Static embeddings
        src_static = self.static_embedding(src)
        dst_static = self.static_embedding(dst)
        
        # Dynamic embeddings
        src_dynamic = self.dynamic_embedding[src]
        dst_dynamic = self.dynamic_embedding[dst]
        
        # Combine
        h = torch.cat([src_static, dst_static, src_dynamic, dst_dynamic], dim=1)
        return self.out(h).squeeze()
    
    def update_embedding(self, src, dst, ts):
        """Update dynamic embeddings"""
        with torch.no_grad():
            # Get current embeddings
            src_dynamic = self.dynamic_embedding[src]
            dst_dynamic = self.dynamic_embedding[dst]
            
            # Time encoding
            time_embed = self.time_encoder(ts)
            
            # Update
            src_new = self.update_net(torch.cat([src_dynamic, dst_dynamic, time_embed], dim=1))
            dst_new = self.update_net(torch.cat([dst_dynamic, src_dynamic, time_embed], dim=1))
            
            self.dynamic_embedding[src] = src_new
            self.dynamic_embedding[dst] = dst_new


class GraphSAGE(nn.Module):
    """GraphSAGE for temporal graphs"""
    
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            self.convs.append(nn.Linear(in_dim * 2 + time_dim, hidden_dim))
            in_dim = hidden_dim
            
        # Output
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Get embeddings
        src_embed = self.node_embedding(src)
        dst_embed = self.node_embedding(dst)
        
        # Time encoding
        time_embed = self.time_encoder(ts)
        
        # Apply SAGE layers
        src_h = src_embed
        dst_h = dst_embed
        
        for conv in self.convs:
            # Aggregate (simplified - just using direct neighbor)
            src_agg = torch.cat([src_h, dst_h, time_embed], dim=1)
            dst_agg = torch.cat([dst_h, src_h, time_embed], dim=1)
            
            # Update
            src_h = F.relu(conv(src_agg))
            dst_h = F.relu(conv(dst_agg))
        
        # Predict
        h = torch.cat([src_h, dst_h], dim=1)
        return self.out(h).squeeze()


class GAT(nn.Module):
    """Graph Attention Network for temporal graphs"""
    
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, num_layers=2, dropout=0.1, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, node_dim)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Attention layers
        self.attentions = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            self.attentions.append(
                nn.MultiheadAttention(in_dim + time_dim, num_heads, dropout=dropout)
            )
            in_dim = in_dim + time_dim
            
        # Output
        self.out = nn.Sequential(
            nn.Linear((node_dim + time_dim) * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Get embeddings
        src_embed = self.node_embedding(src)
        dst_embed = self.node_embedding(dst)
        
        # Time encoding
        time_embed = self.time_encoder(ts)
        
        # Add time to embeddings
        src_h = torch.cat([src_embed, time_embed], dim=1)
        dst_h = torch.cat([dst_embed, time_embed], dim=1)
        
        # Apply attention layers
        for attn in self.attentions:
            # Self-attention (simplified)
            src_h = src_h.unsqueeze(0)  # Add sequence dimension
            dst_h = dst_h.unsqueeze(0)
            
            src_h, _ = attn(src_h, src_h, src_h)
            dst_h, _ = attn(dst_h, dst_h, dst_h)
            
            src_h = src_h.squeeze(0)
            dst_h = dst_h.squeeze(0)
        
        # Predict
        h = torch.cat([src_h, dst_h], dim=1)
        return self.out(h).squeeze()


def get_model(model_name, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, **kwargs):
    """Get model by name with proper parameter handling"""
    
    # Define which models use which parameters
    model_params = {
        'tgn': ['num_nodes', 'node_dim', 'edge_dim', 'time_dim', 'memory_dim', 'hidden_dim'],
        'dyrep': ['num_nodes', 'node_dim', 'edge_dim', 'time_dim', 'hidden_dim'],
        'jodie': ['num_nodes', 'node_dim', 'edge_dim', 'time_dim', 'hidden_dim'],
        'sage': ['num_nodes', 'node_dim', 'edge_dim', 'time_dim', 'hidden_dim'],
        'gat': ['num_nodes', 'node_dim', 'edge_dim', 'time_dim', 'hidden_dim']
    }
    
    models = {
        'tgn': TGN,
        'dyrep': DyRep,
        'jodie': JODIE,
        'sage': GraphSAGE,
        'gat': GAT
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Prepare arguments based on model requirements
    model_args = {
        'num_nodes': num_nodes,
        'node_dim': node_dim,
        'edge_dim': edge_dim,
        'time_dim': time_dim,
        'hidden_dim': hidden_dim
    }
    
    # Add memory_dim only for models that need it
    if 'memory_dim' in model_params.get(model_name, []):
        model_args['memory_dim'] = kwargs.get('memory_dim', hidden_dim)
    
    # Add any additional valid kwargs for this model
    valid_params = model_params.get(model_name, [])
    for key, value in kwargs.items():
        if key in valid_params or key in ['num_layers', 'dropout', 'num_heads']:
            model_args[key] = value
    
    return models[model_name](**model_args)