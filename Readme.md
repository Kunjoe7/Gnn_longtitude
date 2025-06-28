# TGB Pipeline - Complete Framework for Temporal GNN Experiments

A modular and extensible pipeline for running experiments on Temporal Graph Benchmark (TGB) datasets with various temporal GNN models.

## 📁 Project Structure

```
tgb-pipeline/
├── config.py           # Configuration management
├── data_module.py      # Data loading and preprocessing
├── models.py           # Temporal GNN model implementations
├── trainer.py          # Training and evaluation logic
├── utils.py            # Utility functions
├── main.py             # Main experiment script
├── compare_models.py   # Model comparison script
├── data/               # Dataset directory
├── logs/               # Training logs
├── checkpoints/        # Model checkpoints
├── results/            # Experiment results
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torch-geometric numpy pandas scikit-learn tqdm matplotlib seaborn py-tgb
```

### 2. Run a Single Experiment

```bash
# Basic experiment
python main.py --dataset tgbl-wiki --model tgn --epochs 50

# With custom parameters
python main.py \
    --dataset tgbl-wiki \
    --model tgn \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.0005 \
    --hidden_dim 256 \
    --num_layers 3
```

### 3. Compare Multiple Models

```bash
# Compare all models on a dataset
python compare_models.py --dataset tgbl-wiki --models tgn dyrep jodie sage gat --epochs 50 --num_runs 3

# Compare specific models
python compare_models.py --dataset tgbl-review --models tgn jodie --epochs 30 --num_runs 5
```

### 4. Analyze Dataset

```bash
# Analyze and visualize dataset statistics
python main.py --dataset tgbl-wiki --analyze
```

## 🏗️ Architecture

### Configuration (`config.py`)
- Centralized configuration management
- Easy parameter tuning
- Automatic directory creation

### Data Module (`data_module.py`)
- `TGBData`: Wrapper for TGB datasets
- `TemporalDataset`: PyTorch dataset for temporal data
- `NeighborSampler`: Temporal neighbor sampling

### Models (`models.py`)
Implemented models:
- **TGN** (Temporal Graph Networks): Memory-based with attention
- **DyRep**: Dynamic representation learning
- **JODIE**: Joint dynamic user-item embeddings
- **GraphSAGE**: Adapted for temporal graphs
- **GAT**: Graph attention networks for temporal data

### Trainer (`trainer.py`)
- `Trainer`: Handles training loop, evaluation, checkpointing
- Early stopping with patience
- Learning rate scheduling
- TGB official evaluator integration

### Utilities (`utils.py`)
- Experiment tracking and result saving
- Visualization functions
- Model comparison tools
- Dataset analysis

## 📊 Supported Datasets

### Link Prediction
- `tgbl-wiki`: Wikipedia edits (9K nodes, 157K edges)
- `tgbl-review`: Amazon reviews (352K nodes, 4.9M edges)
- `tgbl-coin`: Cryptocurrency (639K nodes, 27.7M edges)
- `tgbl-comment`: Reddit comments (994K nodes, 44.3M edges)
- `tgbl-flight`: Flight records (18K nodes, 67M edges)

### Node Classification
- `tgbn-trade`: International trade (255 nodes, 507K edges)
- `tgbn-genre`: Movie genres (22K nodes, 22.5M edges)
- `tgbn-reddit`: Reddit communities (11K nodes, 672M edges)

## 🔧 Adding New Models

1. Implement your model in `models.py`:

```python
class MyModel(nn.Module):
    def __init__(self, num_nodes, node_dim, edge_dim, time_dim, hidden_dim, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        # Your model architecture
        
    def forward(self, src, dst, ts, edge_feat=None):
        # Forward pass
        return predictions
```

2. Register in `get_model()` function:

```python
models = {
    'my_model': MyModel,
    # ... other models
}
```

3. Run experiments:

```bash
python main.py --dataset tgbl-wiki --model my_model
```

## 📈 Results and Visualization

Results are automatically saved in JSON format with:
- Training history (loss, metrics per epoch)
- Final evaluation metrics
- Model configuration
- Timestamps

Visualizations include:
- Training curves
- Model comparison plots
- Dataset analysis plots

## 🔍 Advanced Usage

### Multiple Runs

```bash
# Run experiment 5 times with different seeds
python main.py --dataset tgbl-wiki --model tgn --num_runs 5
```

### Create Results Summary

```bash
# Generate summary table of all experiments
python compare_models.py --create_table
```

### Custom Configuration

Create a custom config in your script:

```python
from config import Config
from main import run_experiment

config = Config()
config.dataset = 'tgbl-review'
config.model_name = 'tgn'
config.epochs = 100
config.batch_size = 512
config.lr = 0.0001
config.hidden_dim = 256

results = run_experiment(config)
```

## 📊 Example Output

```
============================================================
Running experiment: tgn on tgbl-wiki
============================================================
Configuration:
  dataset: tgbl-wiki
  model_name: tgn
  epochs: 50
  batch_size: 200
  lr: 0.001
============================================================

1. Loading data...
Dataset: tgbl-wiki
  Task: link_prediction
  Nodes: 9,227
  Edges: 157,474
  Train: 122,657 (77.9%)
  Val: 17,337 (11.0%)
  Test: 17,480 (11.1%)

2. Creating model: tgn
Model parameters: 1,234,567

3. Training...
Epoch 1/50
  Train Loss: 0.6931
  Val AUC: 0.7234
  Test AUC: 0.7156

...

Best Val AUC: 0.8912 at epoch 35
Final Test AUC: 0.8876
```

## 🛠️ Troubleshooting

1. **Out of Memory**: Reduce `batch_size` or `hidden_dim`
2. **Slow Training**: Use GPU with `--device cuda`
3. **Poor Performance**: Try different learning rates or more epochs

## 📚 Citation

If you use this pipeline, please cite:

```bibtex
@article{huang2023temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author={Huang, Shenyang and others},
  journal={arXiv preprint arXiv:2307.01026},
  year={2023}
}
```

## 📝 License

MIT License