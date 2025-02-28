import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, global_max_pool
from typing import Dict, List, Union, Tuple, Optional


class ADMETPredictor(torch.nn.Module):
    """
    Graph Neural Network for ADMET prediction with multi-task capability
    """
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        task_types: List[str] = None,
        use_edge_features: bool = True,
        gnn_type: str = 'GIN',
        pool_type: str = 'mean',
    ):
        """
        Initialize ADMET Predictor
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_channels: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout_rate: Dropout rate
            task_types: List of task types ('classification' or 'regression')
            use_edge_features: Whether to use edge features
            gnn_type: Type of GNN layer ('GCN' or 'GIN')
            pool_type: Type of pooling ('mean', 'add', or 'max')
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.task_types = task_types or []
        self.use_edge_features = use_edge_features
        self.pool_type = pool_type
        
        # Define GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer (input to hidden)
        if gnn_type == 'GCN':
            self.gnn_layers.append(GCNConv(node_features, hidden_channels))
        elif gnn_type == 'GIN':
            # For GIN, we need to create an MLP
            nn1 = nn.Sequential(
                nn.Linear(node_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.gnn_layers.append(GINConv(nn1))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Middle layers (hidden to hidden)
        for _ in range(num_layers - 1):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'GIN':
                nn_i = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.gnn_layers.append(GINConv(nn_i))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Define task-specific heads (one per task)
        self.task_heads = nn.ModuleList()
        for task_type in self.task_types:
            if task_type == 'classification':
                # Binary classification head
                head = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_channels // 2, 1)
                )
            else:  # regression
                # Regression head
                head = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_channels // 2, 1)
                )
            self.task_heads.append(head)
    
    def forward(self, x, edge_index, edge_attr, batch, task_id=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]
            task_id: Task ID (optional)
            
        Returns:
            Prediction(s) for the specified task(s)
        """
        # Apply GNN layers
        for i, gnn in enumerate(self.gnn_layers):
            # For GCN, we can use edge attributes if available
            if self.use_edge_features and isinstance(gnn, GCNConv) and edge_attr is not None:
                x = gnn(x, edge_index, edge_attr)
            else:
                x = gnn(x, edge_index)
                
            if x.size(0)==1 and self.training:
                x = F.relu(x)
            else:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Global pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'add':
            x = global_add_pool(x, batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")
        
        # Apply task-specific head(s)
        if task_id is not None:
            # Single task prediction
            return self.task_heads[task_id](x)
        else:
            # Multi-task prediction
            return [head(x) for head in self.task_heads]
    
    def predict(self, data):
        """
        Make prediction for a single data point
        
        Args:
            data: PyG Data object
            
        Returns:
            Prediction for the task associated with the data
        """
        self.eval()
        with torch.no_grad():
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            task_id = int(data.task_id) if hasattr(data, 'task_id') else None
            output = self.forward(x, edge_index, edge_attr, batch, task_id)
            
            # Process output based on task type
            if task_id is not None and hasattr(data, 'task_type'):
                if data.task_type == 'classification':
                    return torch.sigmoid(output).item()
                else:  # regression
                    return output.item()
            else:
                return output