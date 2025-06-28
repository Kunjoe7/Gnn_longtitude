# utils.py
"""
Utility functions for TGB pipeline
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.device):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting their __dict__
        return make_json_serializable(obj.__dict__)
    else:
        return obj


def save_results(results, config, path=None):
    """Save experiment results"""
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.dataset}_{config.model_name}_{timestamp}.json"
        path = os.path.join(config.result_dir, filename)
    
    # Prepare results dictionary and make it JSON serializable
    output = {
        'config': make_json_serializable(config.__dict__),
        'results': make_json_serializable(results),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON
    with open(path, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Results saved to {path}")
    
    return path


def load_results(path):
    """Load experiment results"""
    with open(path, 'r') as f:
        results = json.load(f)
    return results


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    axes[0].plot(history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    # Validation AUC
    axes[1].plot(history['val_auc'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    axes[1].grid(True)
    
    # Test AUC
    axes[2].plot(history['test_auc'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Test AUC')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()


def compare_models(results_dict, metric='test_auc', save_path=None):
    """Compare multiple models"""
    models = list(results_dict.keys())
    
    # Extract metrics
    if isinstance(list(results_dict.values())[0], dict):
        # Single run results
        values = [results_dict[model].get(metric, 0) for model in models]
        errors = None
    else:
        # Multiple run results
        values = []
        errors = []
        for model in models:
            model_values = [r.get(metric, 0) for r in results_dict[model]]
            values.append(np.mean(model_values))
            errors.append(np.std(model_values))
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    
    if errors:
        plt.bar(x, values, yerr=errors, capsize=5)
    else:
        plt.bar(x, values)
    
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(x, models)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        if errors:
            plt.text(i, v + errors[i], f'{v:.3f}', ha='center', va='bottom')
        else:
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()


def analyze_dataset(data, save_dir=None):
    """Analyze and visualize dataset statistics"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Basic statistics
    print(f"\nDataset Analysis: {data.dataset_name}")
    print("="*50)
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Edge features: {data.edge_dim}")
    print(f"Average degree: {(2 * data.num_edges) / data.num_nodes:.2f}")
    
    # Temporal statistics
    time_range = data.timestamps.max() - data.timestamps.min()
    print(f"\nTemporal statistics:")
    print(f"Time range: {time_range:.2f}")
    print(f"Average inter-event time: {time_range / data.num_edges:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Temporal distribution
    axes[0, 0].hist(data.timestamps, bins=50, alpha=0.7)
    axes[0, 0].axvline(data.timestamps[data.train_mask].max(), 
                       color='red', linestyle='--', label='Train/Val split')
    axes[0, 0].axvline(data.timestamps[data.val_mask].max(), 
                       color='green', linestyle='--', label='Val/Test split')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Number of edges')
    axes[0, 0].set_title('Temporal Distribution of Edges')
    axes[0, 0].legend()
    
    # 2. Node degree distribution
    node_degrees = np.bincount(data.sources) + np.bincount(data.destinations)
    degree_counts = np.bincount(node_degrees)
    
    axes[0, 1].loglog(range(1, len(degree_counts)), degree_counts[1:], 'o-', alpha=0.7)
    axes[0, 1].set_xlabel('Degree')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Node Degree Distribution (log-log)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Inter-event time distribution
    sorted_times = np.sort(data.timestamps)
    inter_times = np.diff(sorted_times)
    inter_times = inter_times[inter_times > 0]  # Remove zero intervals
    
    axes[1, 0].hist(np.log10(inter_times + 1e-10), bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('log10(Inter-event time)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Inter-event Time Distribution (log scale)')
    
    # 4. Activity over time
    time_bins = np.linspace(data.timestamps.min(), data.timestamps.max(), 100)
    activity, _ = np.histogram(data.timestamps, bins=time_bins)
    
    axes[1, 1].plot(time_bins[:-1], activity, alpha=0.7)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Number of edges')
    axes[1, 1].set_title('Network Activity Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{data.dataset_name}_analysis.png'))
    else:
        plt.show()
        
    plt.close()
    
    # Return statistics
    stats = {
        'num_nodes': int(data.num_nodes),
        'num_edges': int(data.num_edges),
        'avg_degree': float((2 * data.num_edges) / data.num_nodes),
        'time_range': float(time_range),
        'avg_inter_event_time': float(time_range / data.num_edges),
        'train_ratio': float(data.train_mask.sum() / data.num_edges),
        'val_ratio': float(data.val_mask.sum() / data.num_edges),
        'test_ratio': float(data.test_mask.sum() / data.num_edges)
    }
    
    return stats


def create_results_table(results_dir, save_path=None):
    """Create a summary table of all results"""
    # Find all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    # Load results
    all_results = []
    for file in result_files:
        path = os.path.join(results_dir, file)
        result = load_results(path)
        
        # Extract key information
        config = result['config']
        metrics = result['results']
        
        row = {
            'dataset': config['dataset'],
            'model': config['model_name'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'lr': config['lr'],
            'timestamp': result['timestamp']
        }
        
        # Add metrics
        if isinstance(metrics, dict):
            row.update(metrics)
        else:
            # Handle history format
            if 'history' in metrics:
                history = metrics['history']
                row['final_train_loss'] = history['train_loss'][-1]
                row['best_val_auc'] = max(history['val_auc'])
                row['final_test_auc'] = history['test_auc'][-1]
        
        all_results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by dataset and model
    df = df.sort_values(['dataset', 'model'])
    
    # Save or display
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")
    else:
        print(df.to_string())
    
    return df