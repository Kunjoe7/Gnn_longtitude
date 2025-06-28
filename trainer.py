# trainer.py
"""
Training and evaluation functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import json
from datetime import datetime


class Trainer:
    """Trainer class for temporal GNNs"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_auc': [],
            'test_auc': []
        }
        
        # Best model
        self.best_val_auc = 0
        self.best_epoch = 0
        
    def train_epoch(self, data, mask):
        """Train for one epoch"""
        self.model.train()
        
        # Get training data
        train_src = data.sources[mask]
        train_dst = data.destinations[mask]
        train_ts = data.timestamps[mask]
        train_feat = data.edge_feat[mask] if data.edge_feat is not None else None
        
        # Shuffle
        n = len(train_src)
        perm = np.random.permutation(n)
        
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(range(0, n, self.config.batch_size), desc="Training")
        
        for i in pbar:
            # Get batch
            idx = perm[i:i+self.config.batch_size]
            
            src = torch.LongTensor(train_src[idx]).to(self.device)
            dst = torch.LongTensor(train_dst[idx]).to(self.device)
            ts = torch.FloatTensor(train_ts[idx]).to(self.device)
            
            if train_feat is not None:
                feat = torch.FloatTensor(train_feat[idx]).to(self.device)
            else:
                feat = None
            
            # Forward - positive samples
            pos_out = self.model(src, dst, ts, feat)
            
            # Negative sampling
            neg_dst = torch.randint(0, self.model.num_nodes, (len(src),)).to(self.device)
            neg_out = self.model(src, neg_dst, ts, feat)
            
            # Loss
            pos_loss = self.criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = self.criterion(neg_out, torch.zeros_like(neg_out))
            loss = pos_loss + neg_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update memory if model has it
            if hasattr(self.model, 'update_memory'):
                self.model.update_memory(src, dst, ts, feat)
            elif hasattr(self.model, 'update_embedding'):
                self.model.update_embedding(src, dst, ts)
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self, data, mask):
        """Evaluate model"""
        self.model.eval()
        
        # Get eval data
        eval_src = data.sources[mask]
        eval_dst = data.destinations[mask]
        eval_ts = data.timestamps[mask]
        eval_feat = data.edge_feat[mask] if data.edge_feat is not None else None
        
        all_scores = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(range(0, len(eval_src), self.config.batch_size), desc="Evaluating")
        
        with torch.no_grad():
            for i in pbar:
                # Get batch
                src = torch.LongTensor(eval_src[i:i+self.config.batch_size]).to(self.device)
                dst = torch.LongTensor(eval_dst[i:i+self.config.batch_size]).to(self.device)
                ts = torch.FloatTensor(eval_ts[i:i+self.config.batch_size]).to(self.device)
                
                if eval_feat is not None:
                    feat = torch.FloatTensor(eval_feat[i:i+self.config.batch_size]).to(self.device)
                else:
                    feat = None
                
                # Positive samples
                pos_out = self.model(src, dst, ts, feat)
                all_scores.extend(pos_out.sigmoid().cpu().numpy())
                all_labels.extend(np.ones(len(pos_out)))
                
                # Negative samples
                neg_dst = torch.randint(0, self.model.num_nodes, (len(src),)).to(self.device)
                neg_out = self.model(src, neg_dst, ts, feat)
                all_scores.extend(neg_out.sigmoid().cpu().numpy())
                all_labels.extend(np.zeros(len(neg_out)))
        
        # Compute AUC
        auc = roc_auc_score(all_labels, all_scores)
        
        return auc
    
    def train(self, data, epochs=None):
        """Full training loop"""
        if epochs is None:
            epochs = self.config.epochs
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(data, data.train_mask)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate
            val_auc = self.evaluate(data, data.val_mask)
            self.history['val_auc'].append(val_auc)
            
            # Test (optional)
            test_auc = self.evaluate(data, data.test_mask)
            self.history['test_auc'].append(test_auc)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUC: {val_auc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_auc)
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model! Val AUC: {val_auc:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if epoch - self.best_epoch > self.config.patience and epoch >= self.config.min_epochs:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch+1}")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        # Final test evaluation
        final_test_auc = self.evaluate(data, data.test_mask)
        print(f"Final Test AUC: {final_test_auc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': int(epoch),  # Convert to Python int to avoid numpy types
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_auc': float(self.best_val_auc),  # Convert to Python float
            'best_epoch': int(self.best_epoch),  # Convert to Python int
            'config': self.config.__dict__
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, f'{self.config.model_name}_best.pt')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'{self.config.model_name}_epoch{epoch}.pt')
        
        # Save without numpy objects and use legacy format for compatibility
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
        
    def load_checkpoint(self, path=None, is_best=False):
        """Load model checkpoint with robust error handling"""
        if path is None:
            if is_best:
                path = os.path.join(self.config.checkpoint_dir, f'{self.config.model_name}_best.pt')
            else:
                # Load latest checkpoint
                checkpoints = [f for f in os.listdir(self.config.checkpoint_dir) 
                             if f.startswith(self.config.model_name) and 'epoch' in f]
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoints.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
                path = os.path.join(self.config.checkpoint_dir, checkpoints[-1])
        
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        
        # Try multiple loading methods for robustness
        checkpoint = None
        try:
            # First try with weights_only=True and safe globals
            import numpy as np
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            print(f"Loaded checkpoint with weights_only=True")
        except Exception as e:
            print(f"Failed to load with weights_only=True: {e}")
            try:
                # Fallback to weights_only=False (less secure but works)
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                print(f"Loaded checkpoint with weights_only=False")
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}")
                return None
        
        if checkpoint is None:
            print("Failed to load checkpoint")
            return None
        
        # Load the checkpoint data
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch} with best val AUC: {self.best_val_auc:.4f}")
        
        return checkpoint


# def evaluate_tgb(model, data, evaluator, split='test', batch_size=200, device='cuda'):
#     """Evaluate using TGB's official evaluator"""
#     model.eval()
    
#     # Get mask
#     if split == 'train':
#         mask = data.train_mask
#     elif split == 'val':
#         mask = data.val_mask
#     else:
#         mask = data.test_mask
    
#     # Get data
#     eval_src = data.sources[mask]
#     eval_dst = data.destinations[mask]
#     eval_ts = data.timestamps[mask]
#     eval_feat = data.edge_feat[mask] if data.edge_feat is not None else None
    
#     pos_preds = []
#     neg_preds = []
    
#     with torch.no_grad():
#         for i in range(0, len(eval_src), batch_size):
#             # Get batch
#             src = torch.LongTensor(eval_src[i:i+batch_size]).to(device)
#             dst = torch.LongTensor(eval_dst[i:i+batch_size]).to(device)
#             ts = torch.FloatTensor(eval_ts[i:i+batch_size]).to(device)
            
#             if eval_feat is not None:
#                 feat = torch.FloatTensor(eval_feat[i:i+batch_size]).to(device)
#             else:
#                 feat = None
            
#             # Positive predictions
#             pos_out = model(src, dst, ts, feat)
#             pos_preds.append(pos_out.sigmoid().cpu().numpy())
            
#             # Negative predictions
#             neg_dst = torch.randint(0, model.num_nodes, (len(src),)).to(device)
#             neg_out = model(src, neg_dst, ts, feat)
#             neg_preds.append(neg_out.sigmoid().cpu().numpy())
    
#     # Concatenate predictions
#     pos_preds = np.concatenate(pos_preds)
#     neg_preds = np.concatenate(neg_preds)
    
#     # Prepare input for evaluator
#     input_dict = {
#         "y_pred_pos": pos_preds,
#         "y_pred_neg": neg_preds.reshape(-1, 1),
#         "eval_metric": ["mrr", "hits@1", "hits@3", "hits@10"]
#     }
    
#     # Evaluate
#     try:
#         results = evaluator.eval(input_dict)
#         return results
#     except Exception as e:
#         print(f"TGB evaluator error: {e}")
#         # Fallback to AUC
#         labels = np.concatenate([np.ones(len(pos_preds)), np.zeros(len(neg_preds))])
#         scores = np.concatenate([pos_preds, neg_preds])
#         auc = roc_auc_score(labels, scores)
#         return {"auc": auc}
def evaluate_tgb(model, data, evaluator, split='test', batch_size=200, device='cuda'):
    model.eval()
    
    # Get mask
    if split == 'train':
        mask = data.train_mask
    elif split == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask
    
    # Get data
    eval_src = data.sources[mask]
    eval_dst = data.destinations[mask]
    eval_ts = data.timestamps[mask]
    eval_feat = data.edge_feat[mask] if data.edge_feat is not None else None
    
    pos_preds = []
    neg_preds = []
    
    with torch.no_grad():
        for i in range(0, len(eval_src), batch_size):
            # Get batch
            src = torch.LongTensor(eval_src[i:i+batch_size]).to(device)
            dst = torch.LongTensor(eval_dst[i:i+batch_size]).to(device)
            ts = torch.FloatTensor(eval_ts[i:i+batch_size]).to(device)
            
            if eval_feat is not None:
                feat = torch.FloatTensor(eval_feat[i:i+batch_size]).to(device)
            else:
                feat = None
            
            # Positive predictions
            pos_out = model(src, dst, ts, feat)
            pos_preds.append(pos_out.sigmoid().cpu().numpy())
            
            # Negative predictions
            neg_dst = torch.randint(0, model.num_nodes, (len(src),)).to(device)
            neg_out = model(src, neg_dst, ts, feat)
            neg_preds.append(neg_out.sigmoid().cpu().numpy())
    
    # Concatenate predictions
    pos_preds = np.concatenate(pos_preds)
    neg_preds = np.concatenate(neg_preds)
    
    # Prepare input for evaluator - FIX: Use correct metric format
    input_dict = {
        "y_pred_pos": pos_preds,
        "y_pred_neg": neg_preds.reshape(-1, 1),
        "eval_metric": ["mrr"]  # Use only supported metrics
    }
    
    # Evaluate
    try:
        results = evaluator.eval(input_dict)
        return results
    except Exception as e:
        print(f"TGB evaluator error: {e}")
        # Fallback to AUC calculation
        labels = np.concatenate([np.ones(len(pos_preds)), np.zeros(len(neg_preds))])
        scores = np.concatenate([pos_preds, neg_preds])
        auc = roc_auc_score(labels, scores)
        return {"auc": float(auc)}  # Ensure it's JSON serializable