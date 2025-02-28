#!/usr/bin/env python
# Example script for training a multi-task GNN model with enhanced data preparation

import torch
from torch_geometric.loader import DataLoader
from modules.data_setup import prepare_data
from modules.model import MultiTaskGNN, train_model
from modules.utils import set_device, set_seed
import argparse
import os
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train a multi-task GNN model for ADMET prediction')
    
    # Data preparation arguments
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for data')
    parser.add_argument('--add_h', action='store_true', help='Add hydrogen atoms to molecules')
    parser.add_argument('--calculate_charges', action='store_true', help='Calculate Gasteiger charges')
    parser.add_argument('--use_3d', action='store_true', help='Use 3D coordinates as features')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--cache_graphs', action='store_true', help='Cache processed graphs')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = set_device()
    print(f"Using device: {device}")
    
    # Prepare data with enhanced features
    print("Preparing data...")
    dataset = prepare_data(
        root=args.data_root,
        add_h=args.add_h,
        calculate_charges=args.calculate_charges,
        use_3d=args.use_3d,
        augment=args.augment,
        cache_graphs=args.cache_graphs
    )
    
    # Get task information
    task_info = dataset.get_task_info()
    print(f"Number of tasks: {len(task_info['task_names'])}")
    print(f"Task types: {task_info['task_types']}")
    
    # Get train/test split
    train_indices, test_indices = dataset.get_split_indices(
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    print(f"Train set size: {len(train_indices)}, Test set size: {len(test_indices)}")
    
    # Create data loaders
    train_loader = DataLoader(
        [dataset[i] for i in train_indices],
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        [dataset[i] for i in test_indices],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Determine input dimension from the first graph
    input_dim = dataset[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = MultiTaskGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        task_info=task_info,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm
    )
    
    # Print model summary
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train model
    print("Training model...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        task_info=task_info,
        device=device,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Save model
    torch.save(trained_model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        history_serializable = {
            'train_loss': [float(loss) for loss in history['train_loss']],
            'val_metrics': []
        }
        
        for metrics in history['val_metrics']:
            metrics_serializable = {
                'regression': {
                    'mse': {str(k): float(v) for k, v in metrics['regression']['mse'].items()},
                    'mae': {str(k): float(v) for k, v in metrics['regression']['mae'].items()},
                    'r2': {str(k): float(v) for k, v in metrics['regression']['r2'].items()}
                },
                'classification': {
                    'accuracy': {str(k): float(v) for k, v in metrics['classification']['accuracy'].items()},
                    'precision': {str(k): float(v) for k, v in metrics['classification']['precision'].items()},
                    'recall': {str(k): float(v) for k, v in metrics['classification']['recall'].items()},
                    'f1': {str(k): float(v) for k, v in metrics['classification']['f1'].items()}
                }
            }
            history_serializable['val_metrics'].append(metrics_serializable)
        
        json.dump(history_serializable, f, indent=4)
    
    print(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 