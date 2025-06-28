# main.py
"""
Main script for TGB experiments
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import numpy as np

from config import Config
from data_module import TGBData
from models import get_model
from trainer import Trainer, evaluate_tgb
from utils import set_seed, save_results, plot_training_history, analyze_dataset


def run_experiment(config):
    """Run a single experiment"""
    print("="*60)
    print(f"Running experiment: {config.model_name} on {config.dataset}")
    print("="*60)
    print(config)
    print("="*60)
    
    # Set seed
    set_seed(config.seed)
    
    # Load data
    print("\n1. Loading data...")
    data = TGBData(config.dataset, config.data_root)
    
    # Update config with dataset info
    config.edge_dim = data.edge_dim
    config.num_nodes = data.num_nodes
    
    # Create model
    print(f"\n2. Creating model: {config.model_name}")
    model = get_model(
        model_name=config.model_name,
        num_nodes=config.num_nodes,
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        time_dim=config.time_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        memory_dim=getattr(config, 'memory_dim', config.hidden_dim)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = Trainer(model, config, config.device)
    
    # Train model
    print("\n4. Training...")
    history = trainer.train(data, epochs=config.epochs)
    
    # Evaluate with TGB evaluator
    print("\n5. Final evaluation...")
    final_results = evaluate_tgb(
        model, data, data.evaluator, 
        split='test', 
        batch_size=config.batch_size, 
        device=config.device
    )
    
    # Prepare results
    results = {
        'history': history,
        'final_metrics': final_results,
        'best_val_auc': trainer.best_val_auc,
        'best_epoch': trainer.best_epoch
    }
    
    # Save results
    save_results(results, config)
    
    # Plot training history
    plot_path = os.path.join(config.result_dir, f"{config.dataset}_{config.model_name}_history.png")
    plot_training_history(history, plot_path)
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Best Val AUC: {trainer.best_val_auc:.4f}")
    print(f"Final Test Metrics: {final_results}")
    print("="*60)
    
    return results


def run_multiple_experiments(config, num_runs=5):
    """Run multiple experiments with different seeds"""
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{num_runs}")
        print(f"{'='*60}")
        
        # Update seed
        config.seed = config.seed + run
        
        # Run experiment
        results = run_experiment(config)
        all_results.append(results)
    
    # Aggregate results
    final_metrics = [r['final_metrics'] for r in all_results]
    
    # Calculate mean and std
    aggregated = {}
    for metric in final_metrics[0].keys():
        values = [m[metric] for m in final_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    print("\n" + "="*60)
    print("Aggregated Results")
    print("="*60)
    for metric, stats in aggregated.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return all_results, aggregated


def main():
    parser = argparse.ArgumentParser(description='TGB Pipeline')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='tgbl-wiki',
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='tgn',
                        choices=['tgn', 'dyrep', 'jodie', 'sage', 'gat'],
                        help='Model name')
    parser.add_argument('--node_dim', type=int, default=100,
                        help='Node embedding dimension')
    parser.add_argument('--time_dim', type=int, default=100,
                        help='Time embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Experiment arguments
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze dataset before training')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.update(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model,
        node_dim=args.node_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed
    )
    
    # Analyze dataset if requested
    if args.analyze:
        print("\nAnalyzing dataset...")
        data = TGBData(config.dataset, config.data_root)
        stats = analyze_dataset(data, save_dir=config.result_dir)
        print(f"\nDataset statistics saved to {config.result_dir}")
        return
    
    # Run experiments
    if args.num_runs == 1:
        results = run_experiment(config)
    else:
        results, aggregated = run_multiple_experiments(config, args.num_runs)
        
        # Save aggregated results
        save_results(aggregated, config, 
                    os.path.join(config.result_dir, f"{config.dataset}_{config.model_name}_aggregated.json"))
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()