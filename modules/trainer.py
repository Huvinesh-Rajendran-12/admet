import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import numpy as np
import logging
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import os
import json

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for ADMET prediction models
    """
    def __init__(
        self,
        model,
        dataset,
        train_indices,
        val_indices,
        test_indices=None,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-5,
        num_epochs=100,
        patience=10,
        device=None,
        output_dir='results',
        use_subset=True
    ):
        """
        Initialize Trainer
        
        Args:
            model: GNN model
            dataset: ADMET dataset (ADMETDataset or LazyADMETDataset)
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices (optional)
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of training epochs
            patience: Early stopping patience
            device: Device to use (cuda or cpu)
            output_dir: Directory to save results
            use_subset: Whether to use PyG Subset (set to False for standard PyTorch datasets)
        """
        self.model = model
        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.use_subset = use_subset
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataLoaders
        if self.use_subset:
            # Using PyG Subset for efficient lazy loading
            self.train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=batch_size,
                shuffle=True,
                exclude_keys=['raw_y'],
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=batch_size,
                shuffle=False,
                exclude_keys=['raw_y'],
                drop_last=True
            )
            
            if test_indices is not None:
                self.test_loader = DataLoader(
                    Subset(dataset, test_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    exclude_keys=['raw_y'],
                    drop_last=True
                )
            else:
                self.test_loader = None
        else:
            # Alternative approach using list comprehension (less memory efficient)
            self.train_loader = DataLoader(
                [dataset[i] for i in train_indices],
                batch_size=batch_size,
                shuffle=True,
                exclude_keys=['raw_y'],
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                [dataset[i] for i in val_indices],
                batch_size=batch_size,
                shuffle=False,
                exclude_keys=['raw_y'],
                drop_last=True
            )
            
            if test_indices is not None:
                self.test_loader = DataLoader(
                    [dataset[i] for i in test_indices],
                    batch_size=batch_size,
                    shuffle=False,
                    exclude_keys=['raw_y'],
                    drop_last=True
                )
            else:
                self.test_loader = None
        
        # Get task information
        self.task_info = dataset.get_task_info()
        self.task_types = self.task_info['task_types']
        self.task_names = self.task_info['task_names']
        
        # Set up optimizer and loss functions
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions based on task types
        self.loss_fns = []
        for task_type in self.task_types:
            if task_type == 'classification':
                self.loss_fns.append(nn.BCEWithLogitsLoss())
            else:  # regression
                self.loss_fns.append(nn.MSELoss())
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses = [0] * len(self.task_types)
        task_counts = [0] * len(self.task_types)
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for data in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Group by task_id for batch efficiency
            task_outputs = {}
            for task_id in range(len(self.task_types)):
                # Find examples for this task
                mask = data.task_id == task_id
                if mask.sum() > 0:
                    # Extract task data
                    task_data = data[mask]
                    print(task_data)
                    from torch_geometric.data import Batch
                    if isinstance(task_data, list):
                        task_data = Batch.from_data_list(task_data)
                    x, edge_index, edge_attr, batch = task_data.x, task_data.edge_index, task_data.edge_attr, task_data.batch
                    
                    # Forward pass
                    output = self.model(x, edge_index, edge_attr, batch, task_id)
                    
                    # Compute loss
                    target = task_data.y
                    if task_data.to_data_list()[0].task_type=='classification':
                        target = target.view(-1, 1)
                        print(output)
                        print(target)
                    loss = self.loss_fns[task_id](output, target)
                    
                    # Add to totals
                    task_losses[task_id] += loss.item() * mask.sum().item()
                    task_counts[task_id] += mask.sum().item()
                    total_loss += loss.item() * mask.sum().item()
                    
                    # Backward pass
                    loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / sum(task_counts) if sum(task_counts) > 0 else 0})
        
        # Calculate average loss per task
        avg_task_losses = []
        for i in range(len(self.task_types)):
            if task_counts[i] > 0:
                avg_task_losses.append(task_losses[i] / task_counts[i])
            else:
                avg_task_losses.append(0)
        
        # Calculate average loss
        avg_loss = total_loss / sum(task_counts) if sum(task_counts) > 0 else 0
        
        return avg_loss, avg_task_losses
    
    def validate(self, loader=None):
        """Validate the model"""
        if loader is None:
            loader = self.val_loader
            
        self.model.eval()
        total_loss = 0
        task_losses = [0] * len(self.task_types)
        task_counts = [0] * len(self.task_types)
        
        # For metrics calculation
        task_preds = [[] for _ in range(len(self.task_types))]
        task_targets = [[] for _ in range(len(self.task_types))]
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # Group by task_id for batch efficiency
                for task_id in range(len(self.task_types)):
                    # Find examples for this task
                    mask = data.task_id == task_id
                    if mask.sum() > 0:
                        # Extract task data
                        task_data = data[mask]
                        x = task_data.x
                        edge_index = task_data.edge_index
                        edge_attr = task_data.edge_attr
                        batch = task_data.batch
                        
                        # Forward pass
                        output = self.model(x, edge_index, edge_attr, batch, task_id)
                        
                        # Compute loss
                        target = task_data.y
                        loss = self.loss_fns[task_id](output, target)
                        
                        # Add to totals
                        task_losses[task_id] += loss.item() * mask.sum().item()
                        task_counts[task_id] += mask.sum().item()
                        total_loss += loss.item() * mask.sum().item()
                        
                        # Store predictions and targets for metrics
                        if self.task_types[task_id] == 'classification':
                            preds = torch.sigmoid(output).cpu().numpy()
                        else:
                            preds = output.cpu().numpy()
                        
                        task_preds[task_id].extend(preds.flatten().tolist())
                        task_targets[task_id].extend(target.cpu().numpy().flatten().tolist())
        
        # Calculate average loss per task
        avg_task_losses = []
        for i in range(len(self.task_types)):
            if task_counts[i] > 0:
                avg_task_losses.append(task_losses[i] / task_counts[i])
            else:
                avg_task_losses.append(0)
        
        # Calculate average loss
        avg_loss = total_loss / sum(task_counts) if sum(task_counts) > 0 else 0
        
        # Calculate metrics per task
        metrics = {}
        for i in range(len(self.task_types)):
            task_name = self.task_names[i]
            if not task_preds[i]:  # Skip if no predictions
                continue
                
            task_metrics = {}
            if self.task_types[i] == 'classification':
                # Classification metrics
                try:
                    task_metrics['auc'] = roc_auc_score(task_targets[i], task_preds[i])
                except:
                    task_metrics['auc'] = float('nan')  # In case of single class
            else:
                # Regression metrics
                task_metrics['rmse'] = np.sqrt(mean_squared_error(task_targets[i], task_preds[i]))
                task_metrics['mae'] = mean_absolute_error(task_targets[i], task_preds[i])
                task_metrics['r2'] = r2_score(task_targets[i], task_preds[i])
            
            metrics[task_name] = task_metrics
        
        return avg_loss, avg_task_losses, metrics
    
    def train(self):
        """Train the model"""
        logger.info(f"Starting training on device: {self.device}")
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        best_metrics = None
        best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        
        # Training loop
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_task_losses = self.train_epoch()
            
            # Validate
            val_loss, val_task_losses, val_metrics = self.validate()
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Time: {epoch_time:.2f}s")
            
            # Log metrics
            for task_name, metrics in val_metrics.items():
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  Task {task_name}: {metric_str}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = val_metrics
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, best_model_path)
                
                logger.info(f"  Saved new best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if epoch - best_epoch >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Test if test set is available
        if self.test_loader is not None:
            logger.info("Evaluating on test set...")
            test_loss, test_task_losses, test_metrics = self.validate(self.test_loader)
            
            # Log test metrics
            logger.info(f"Test Loss: {test_loss:.4f}")
            for task_name, metrics in test_metrics.items():
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  Task {task_name}: {metric_str}")
            
            # Save test metrics
            with open(os.path.join(self.output_dir, 'test_metrics.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        # Return best metrics
        return best_metrics