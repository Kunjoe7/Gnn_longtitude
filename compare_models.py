# compare_models.py
"""
Script to compare different models on TGB datasets
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np

from config import Config
from data_module import TGBData
from models import get_model
from trainer import Trainer
from utils import set_seed, save_results, compare_models, create_results_table, make_json_serializable
from main import run_experiment


def compare_all_models(dataset='tgbl-wiki', models=['tgn', 'dyrep', 'jodie', 'sage', 'gat'], 
                      epochs=50, num_runs=3):
    """Compare all models on a dataset"""
    
    print(f"="*60)
    print(f"Comparing models on {dataset}")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}, Runs: {num_runs}")
    print(f"="*60)
    
    # Base configuration
    base_config = Config()
    base_config.dataset = dataset
    base_config.epochs = epochs
    
    # Results storage
    all_results = {}
    
    # Run each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")
        
        model_results = []
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}")
            
            # Create config for this run
            config = Config()
            config.dataset = dataset
            config.model_name = model_name
            config.epochs = epochs
            config.seed = 42 + run
            
            # Run experiment
            try:
                results = run_experiment(config)
                model_results.append(results)
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                continue
        
        if model_results:
            all_results[model_name] = model_results
    
    # Save comparison results with JSON serialization fix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join('results', f'comparison_{dataset}_{timestamp}.json')
    
    # Make sure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Convert to JSON serializable format
    serializable_results = make_json_serializable(all_results)
    
    with open(comparison_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nComparison results saved to {comparison_path}")
    
    # Create comparison plots
    plot_path = os.path.join('results', f'comparison_{dataset}_{timestamp}.png')
    
    # Extract final test metrics for comparison
    final_metrics = {}
    for model, runs in all_results.items():
        if runs:  # Check if we have results for this model
            # Try to extract metrics from different possible locations
            model_metrics = []
            for run_result in runs:
                if isinstance(run_result, dict):
                    # Look for metrics in different possible keys
                    if 'final_metrics' in run_result:
                        metric_val = run_result['final_metrics'].get('auc', 0)
                    elif 'auc' in run_result:
                        metric_val = run_result['auc']
                    elif 'history' in run_result:
                        # Get best validation AUC from history
                        history = run_result['history']
                        if 'val_auc' in history and history['val_auc']:
                            metric_val = max(history['val_auc'])
                        else:
                            metric_val = 0
                    else:
                        metric_val = 0
                    model_metrics.append(float(metric_val))
                else:
                    model_metrics.append(0)
            
            final_metrics[model] = model_metrics
    
    # Only create comparison plot if we have results
    if final_metrics:
        try:
            compare_models(final_metrics, metric='auc', save_path=plot_path)
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Results")
    print("="*60)
    
    for model, runs in all_results.items():
        if runs:
            # Extract AUC values
            aucs = []
            for run_result in runs:
                if isinstance(run_result, dict):
                    if 'final_metrics' in run_result:
                        auc_val = run_result['final_metrics'].get('auc', 0)
                    elif 'auc' in run_result:
                        auc_val = run_result['auc']
                    elif 'history' in run_result:
                        history = run_result['history']
                        if 'val_auc' in history and history['val_auc']:
                            auc_val = max(history['val_auc'])
                        else:
                            auc_val = 0
                    else:
                        auc_val = 0
                    aucs.append(float(auc_val))
            
            if aucs:
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"{model}: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
            else:
                print(f"{model}: No valid results")
        else:
            print(f"{model}: Failed to run")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Compare models on TGB datasets')
    
    parser.add_argument('--dataset', type=str, default='tgbl-wiki',
                        help='Dataset name')
    parser.add_argument('--models', nargs='+', 
                        default=['tgn', 'dyrep', 'jodie', 'sage', 'gat'],
                        help='Models to compare')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per model')
    parser.add_argument('--create_table', action='store_true',
                        help='Create summary table of all results')
    
    args = parser.parse_args()
    
    if args.create_table:
        # Create summary table
        df = create_results_table('results', 'results/summary_table.csv')
        print("\nSummary table created!")
    else:
        # Run comparison
        results = compare_all_models(
            dataset=args.dataset,
            models=args.models,
            epochs=args.epochs,
            num_runs=args.num_runs
        )
        
        print("\nComparison completed!")


if __name__ == "__main__":
    main()