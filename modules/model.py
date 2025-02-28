import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm

class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for molecular graphs
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout: float = 0.2,
            use_batch_norm: bool = True
    ):
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # First convolution layer
        self.conv_layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        
        # Batch normalization layers (if enabled)
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        
        # Additional convolution layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass through the GNN encoder
        
        Args:
            data: PyG Data or Batch object containing graph data
            
        Returns:
            Graph embeddings tensor
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Process through GNN layers
        for i in range(self.num_layers):
            # Apply convolution
            if edge_attr is not None and hasattr(self.conv_layers[i], 'forward_edge_attr'):
                x = self.conv_layers[i](x, edge_index, edge_attr)
            else:
                x = self.conv_layers[i](x, edge_index)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level embeddings
        x = global_mean_pool(x, batch)
        
        return x

class TaskHead(nn.Module):
    """
    Task-specific head for either regression or classification
    """
    def __init__(
            self,
            input_dim: int,
            task_type: str,
            num_classes: int = 2,
            hidden_dim: Optional[int] = None
    ):
        super(TaskHead, self).__init__()
        
        self.task_type = task_type
        self.hidden_dim = hidden_dim or input_dim // 2
        
        # Task-specific layers
        if task_type == 'classification':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:  # regression
            self.layers = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the task head
        
        Args:
            x: Input tensor (graph embeddings)
            
        Returns:
            Task predictions
        """
        out = self.layers(x)
        
        # Apply appropriate activation for classification
        if self.task_type == 'classification':
            # No softmax here - will be applied in loss function
            return out
        
        return out

class MultiTaskDecoder(nn.Module):
    """
    Multi-task decoder with task-specific heads
    """
    def __init__(
            self,
            input_dim: int,
            task_info: Dict
    ):
        super(MultiTaskDecoder, self).__init__()
        
        self.task_heads = nn.ModuleDict()
        self.task_types = task_info['task_types']
        self.task_names = task_info['task_names']
        
        # Create a head for each task
        for i, task_name in enumerate(self.task_names):
            task_type = self.task_types[i]
            self.task_heads[str(i)] = TaskHead(
                input_dim=input_dim,
                task_type=task_type,
                num_classes=2 if task_type == 'classification' else 1
            )
    
    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the multi-task decoder
        
        Args:
            x: Input tensor (graph embeddings)
            task_ids: Tensor of task IDs for each graph
            
        Returns:
            List of task-specific predictions
        """
        outputs = []
        
        # Process each graph with its corresponding task head
        for i, embedding in enumerate(x):
            task_id = task_ids[i].item()
            task_head = self.task_heads[str(task_id)]
            out = task_head(embedding.unsqueeze(0))
            outputs.append(out)
        
        return outputs

class MultiTaskGNN(nn.Module):
    """
    Multi-task Graph Neural Network for ADMET prediction
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            task_info: Dict,
            num_layers: int = 2,
            dropout: float = 0.2,
            use_batch_norm: bool = True
    ):
        super(MultiTaskGNN, self).__init__()
        
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        self.decoder = MultiTaskDecoder(
            input_dim=hidden_dim,
            task_info=task_info
        )
        
        self.task_info = task_info
    
    def forward(self, data: Union[Data, Batch]) -> List[torch.Tensor]:
        """
        Forward pass through the multi-task GNN
        
        Args:
            data: PyG Data or Batch object containing graph data
            
        Returns:
            List of task-specific predictions
        """
        embeddings = self.encoder(data)
        outputs = self.decoder(embeddings, data.task_id)
        return outputs

def train_step(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        task_info: Dict,
        regression_loss_fn: nn.Module = nn.MSELoss(),
        classification_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        clip_grad_norm: Optional[float] = 1.0
) -> float:
    """
    Single training step (one epoch)
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        device: Device to use for training
        task_info: Dictionary containing task information
        regression_loss_fn: Loss function for regression tasks
        classification_loss_fn: Loss function for classification tasks
        clip_grad_norm: Maximum norm for gradient clipping (None to disable)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Get task types from task_info
    task_types = task_info['task_types']
    task_means = task_info.get('task_means', {})
    task_stds = task_info.get('task_stds', {})
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss
        loss = 0.0
        task_losses = {'regression': 0.0, 'classification': 0.0}
        task_counts = {'regression': 0, 'classification': 0}
        
        for i, output in enumerate(outputs):
            task_id = batch.task_id[i].item()
            task_type = task_types[task_id]
            
            # Get the target value
            if task_type == 'regression':
                # For regression, use normalized y value
                y_i = batch.y[i]
                loss_i = regression_loss_fn(output, y_i)
                task_losses['regression'] += loss_i.item()
                task_counts['regression'] += 1
            else:  # classification
                # For classification, use long tensor
                y_i = batch.y[i].long()
                loss_i = classification_loss_fn(output, y_i)
                task_losses['classification'] += loss_i.item()
                task_counts['classification'] += 1
            
            loss += loss_i
        
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping if enabled
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_reg_loss = task_losses['regression'] / max(task_counts['regression'], 1)
        avg_cls_loss = task_losses['classification'] / max(task_counts['classification'], 1)
        progress_bar.set_postfix({
            'loss': loss.item(),
            'reg_loss': avg_reg_loss,
            'cls_loss': avg_cls_loss
        })
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    return avg_loss

def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        task_info: Dict
) -> Dict:
    """
    Evaluate the model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        task_info: Dictionary containing task information
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    task_types = task_info['task_types']
    task_means = task_info.get('task_means', {})
    task_stds = task_info.get('task_stds', {})
    
    # Metrics storage
    metrics = {
        'regression': {
            'mse': {},
            'mae': {},
            'r2': {}
        },
        'classification': {
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1': {}
        }
    }
    
    # Prediction and target storage by task
    predictions = {i: [] for i in range(len(task_types))}
    targets = {i: [] for i in range(len(task_types))}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            
            for i, output in enumerate(outputs):
                task_id = batch.task_id[i].item()
                task_type = task_types[task_id]
                
                if task_type == 'regression':
                    # For regression, get the prediction and denormalize if needed
                    pred = output.item()
                    if task_id in task_means and task_id in task_stds:
                        # Denormalize prediction
                        pred = pred * task_stds[task_id] + task_means[task_id]
                    
                    # Get the target (raw value if available, otherwise denormalize)
                    if hasattr(batch, 'raw_y') and batch.raw_y is not None:
                        target = batch.raw_y[i].item()
                    else:
                        target = batch.y[i].item()
                        if task_id in task_means and task_id in task_stds:
                            # Denormalize target
                            target = target * task_stds[task_id] + task_means[task_id]
                else:  # classification
                    # For classification, get class prediction
                    pred = torch.softmax(output, dim=1).argmax(dim=1).item()
                    target = batch.y[i].long().item()
                
                predictions[task_id].append(pred)
                targets[task_id].append(target)
    
    # Calculate metrics for each task
    for task_id in range(len(task_types)):
        task_type = task_types[task_id]
        task_preds = np.array(predictions[task_id])
        task_targets = np.array(targets[task_id])
        
        if len(task_preds) == 0:
            continue  # Skip if no predictions for this task
        
        if task_type == 'regression':
            # Regression metrics
            mse = np.mean((task_preds - task_targets) ** 2)
            mae = np.mean(np.abs(task_preds - task_targets))
            
            # R² score
            if np.var(task_targets) != 0:
                r2 = 1 - (np.sum((task_targets - task_preds) ** 2) / 
                          np.sum((task_targets - np.mean(task_targets)) ** 2))
            else:
                r2 = 0.0
            
            metrics['regression']['mse'][task_id] = mse
            metrics['regression']['mae'][task_id] = mae
            metrics['regression']['r2'][task_id] = r2
        else:  # classification
            # Classification metrics
            accuracy = np.mean(task_preds == task_targets)
            
            # Handle binary classification metrics
            if len(np.unique(task_targets)) <= 2:
                # Precision, recall, F1 (for positive class)
                true_positives = np.sum((task_preds == 1) & (task_targets == 1))
                false_positives = np.sum((task_preds == 1) & (task_targets == 0))
                false_negatives = np.sum((task_preds == 0) & (task_targets == 1))
                
                precision = true_positives / max(true_positives + false_positives, 1)
                recall = true_positives / max(true_positives + false_negatives, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                
                metrics['classification']['precision'][task_id] = precision
                metrics['classification']['recall'][task_id] = recall
                metrics['classification']['f1'][task_id] = f1
            
            metrics['classification']['accuracy'][task_id] = accuracy
    
    return metrics

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        task_info: Dict,
        device: torch.device,
        n_epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        early_stopping_patience: int = 10,
        clip_grad_norm: float = 1.0
) -> Tuple[nn.Module, Dict]:
    """
    Train the model with early stopping and learning rate scheduling
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        task_info: Dictionary containing task information
        device: Device to use for training
        n_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        scheduler_factor: Factor by which to reduce learning rate
        scheduler_patience: Patience for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        clip_grad_norm: Maximum norm for gradient clipping
        
    Returns:
        Tuple of (trained model, training history)
    """
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True
    )
    
    # Loss functions
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train for one epoch
        train_loss = train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            task_info=task_info,
            regression_loss_fn=regression_loss_fn,
            classification_loss_fn=classification_loss_fn,
            clip_grad_norm=clip_grad_norm
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            task_info=task_info
        )
        
        # Calculate validation loss (average of MSE for regression and 1-accuracy for classification)
        val_reg_losses = list(val_metrics['regression']['mse'].values())
        val_cls_losses = [1 - acc for acc in val_metrics['classification']['accuracy'].values()]
        
        val_loss = (np.mean(val_reg_losses) if val_reg_losses else 0) + \
                  (np.mean(val_cls_losses) if val_cls_losses else 0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check regression metrics
        if val_reg_losses:
            avg_mse = np.mean(list(val_metrics['regression']['mse'].values()))
            avg_r2 = np.mean(list(val_metrics['regression']['r2'].values()))
            print(f"Regression - Avg MSE: {avg_mse:.4f}, Avg R²: {avg_r2:.4f}")
        
        # Check classification metrics
        if val_cls_losses:
            avg_acc = np.mean(list(val_metrics['classification']['accuracy'].values()))
            print(f"Classification - Avg Accuracy: {avg_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history 