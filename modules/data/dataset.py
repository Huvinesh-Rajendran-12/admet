import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from tdc.benchmark_group import admet_group
import os
import pickle
import logging
import random
from tqdm import tqdm
from typing import List, Tuple
import concurrent.futures
from modules.data.conversion import smiles_to_graph
from modules.data.augmentation import augment_molecule
from modules.data.config import ADMETConfig

logger = logging.getLogger(__name__)

class ADMETDataset(Dataset):
    """PyTorch Geometric Dataset for ADMET data"""
    
    @property
    def processed_dir(self) -> str:
        """Override processed_dir for PyG compatibility"""
        return os.path.join(self.root, 'processed')
        
    @property
    def raw_dir(self) -> str:
        """Override raw_dir for PyG compatibility"""
        return os.path.join(self.root, 'raw')
    
    def __init__(self, config: ADMETConfig):
        """
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.root = config.root_dir
        self.add_h = config.add_hydrogens
        self.calculate_charges = config.calculate_charges
        self.use_3d = config.use_3d_coordinates
        self.augment = config.augment_data
        self.cache_graphs = config.cache_graphs
        self.cache_file = os.path.join(self.root, config.cache_file)
        self.n_jobs = config.n_jobs
        self.batch_size = config.batch_size
        
        self.data_list = []
        self.task_types = []
        self.task_names = []
        self.task_means = {}
        self.task_stds = {}
        
        # Set up logging
        self._setup_logging(config.log_level)
        
        super().__init__(self.root, None, None)
        
        # Load or process data
        if self.cache_graphs and os.path.exists(self.cache_file):
            self._load_cached_data()
        else:
            self._process_data()
            if self.cache_graphs:
                self._cache_data()
    
    def _setup_logging(self, log_level: str = 'INFO'):
        """Set up logging with appropriate level"""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _load_cached_data(self):
        """Load cached processed data"""
        logger.info(f"Loading cached data from {self.cache_file}")
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.data_list = cached_data['data_list']
                self.task_types = cached_data['task_types']
                self.task_names = cached_data['task_names']
                self.task_means = cached_data['task_means']
                self.task_stds = cached_data['task_stds']
            logger.info(f"Loaded {len(self.data_list)} graphs from cache")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading cached data: {e}. Regenerating dataset.")
            self._process_data()
            if self.cache_graphs:
                self._cache_data()
    
    def _cache_data(self):
        """Cache processed data to disk"""
        logger.info(f"Caching processed data to {self.cache_file}")
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'data_list': self.data_list,
                'task_types': self.task_types,
                'task_names': self.task_names,
                'task_means': self.task_means,
                'task_stds': self.task_stds
            }, f)
        logger.info("Caching complete")
    
    def _process_batch(self, batch, task_id, task_name, task_type, task_mean=None, task_std=None):
        """Process a batch of molecules in parallel"""
        results = []
        for row in batch:
            smiles = row['Drug']
            graph = smiles_to_graph(smiles, self.config)
            
            if graph is not None:
                # Add metadata
                graph.task_id = task_id
                graph.task_name = task_name
                graph.task_type = task_type
                
                # Normalize regression values
                if task_type == 'regression':
                    normalized_y = (row['Y'] - task_mean) / task_std
                    graph.y = torch.tensor([normalized_y], dtype=torch.float)
                    graph.raw_y = torch.tensor([row['Y']], dtype=torch.float)
                else:
                    graph.y = torch.tensor([int(row['Y'])], dtype=torch.long)
                
                results.append(graph)
                
                # Data augmentation for training set
                if self.augment:
                    augmented = augment_molecule(smiles)
                    if augmented is not None:
                        aug_graph = smiles_to_graph(augmented, self.config)
                        
                        if aug_graph is not None:
                            aug_graph.task_id = graph.task_id
                            aug_graph.task_name = graph.task_name
                            aug_graph.task_type = graph.task_type
                            aug_graph.y = graph.y.clone()
                            if task_type == 'regression':
                                aug_graph.raw_y = graph.raw_y.clone()
                            results.append(aug_graph)
                            
        return results
    
    def _determine_task_type(self, values):
        """Determine task type based on target values"""
        # Check if all values are integers or booleans
        is_classification = np.all(np.equal(np.mod(values, 1), 0))
        return 'classification' if is_classification else 'regression'
    
    def _process_data(self):
        """Process ADMET data from TDC with parallel processing"""
        logger.info("Starting to process ADMET dataset")
        
        # Load TDC ADMET Benchmark Group
        group = admet_group(path=self.root)
        datasets = group.dataset_names
        self.task_names = datasets
        
        # Counters for tracking
        total_molecules = 0
        processed_molecules = 0
        
        # Process each dataset
        for i, dataset_name in enumerate(datasets):
            dataset = group.get(dataset_name)
            train_df = dataset['train_val']
            
            logger.info(f"Processing dataset: {dataset_name} ({len(train_df)} molecules)")
            
            # Determine task type from values
            values = train_df['Y'].dropna().values
            task_type = self._determine_task_type(values)
            self.task_types.append(task_type)
            
            # Calculate statistics for regression tasks
            if task_type == 'regression':
                self.task_means[i] = float(np.mean(values))
                self.task_stds[i] = float(np.std(values))
                if self.task_stds[i] < 1e-8:  # Avoid division by zero
                    logger.warning(f"Task {dataset_name} has near-zero standard deviation. Using 1.0 instead.")
                    self.task_stds[i] = 1.0
            
            # Process in batches using parallel processing
            rows = train_df.to_dict('records')
            total_molecules += len(rows)
            
            # Create batches
            batches = [rows[j:j+self.batch_size] for j in range(0, len(rows), self.batch_size)]
            
            dataset_results = []
            with tqdm(total=len(batches), desc=f"Processing {dataset_name}") as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []
                    
                    # Submit batches for processing
                    for batch in batches:
                        future = executor.submit(
                            self._process_batch, 
                            batch, 
                            i, 
                            dataset_name, 
                            task_type,
                            self.task_means.get(i, None),
                            self.task_stds.get(i, None)
                        )
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_results = future.result()
                            dataset_results.extend(batch_results)
                            processed_molecules += len(batch_results)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error in batch processing: {e}")
            
            # Add results to data list
            self.data_list.extend(dataset_results)
            
            logger.info(f"Dataset {dataset_name}: Processed {len(dataset_results)} graphs")
        
        logger.info(f"Total: Processed {processed_molecules}/{total_molecules} molecules")
        logger.info(f"Final dataset size: {len(self.data_list)} graphs")
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def get_task_info(self):
        """Return task information"""
        # Create a dictionary mapping task IDs to their names
        task_ids = {i: name for i, name in enumerate(self.task_names)}
        
        return {
            'task_names': self.task_names,
            'task_types': self.task_types,
            'task_means': self.task_means,
            'task_stds': self.task_stds,
            'task_ids': task_ids
        }
    
    def get_split_indices(self, split_ratio=None, seed=None) -> Tuple[List[int], List[int]]:
        """Get train/test split indices stratified by task"""
        if split_ratio is None:
            split_ratio = self.config.split_ratio
            
        if seed is None:
            seed = self.config.random_seed
            
        random.seed(seed)
        
        # Group by task
        task_indices = {}
        for i, data in enumerate(self.data_list):
            task_id = int(data.task_id)
            if task_id not in task_indices:
                task_indices[task_id] = []
            task_indices[task_id].append(i)
        
        # Split each task
        train_indices = []
        test_indices = []
        
        for task_id, indices in task_indices.items():
            random.shuffle(indices)
            split_point = int(len(indices) * split_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
        
        logger.info(f"Split dataset: {len(train_indices)} training, {len(test_indices)} testing samples")
        return train_indices, test_indices


class LazyADMETDataset(Dataset):
    """
    Memory-efficient version of ADMETDataset that loads molecules on-demand
    """
    
    
    def __init__(self, config: ADMETConfig):
        """
        Args:
            config: Configuration object containing all parameters
        """
        # Initialize with parent class first
        super().__init__(config.root_dir, None, None)
        self._indices = None
        self.config = config
        self.root = config.root_dir
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize other attributes
        self.smiles_list = []
        self.y_list = []
        self.task_ids = []
        self.task_names = []
        self.task_types = []
        self.task_means = {}
        self.task_stds = {}
        
        # Cache of loaded molecules (using LRU cache to limit memory usage)
        self.cache = {}
        self.cache_size = 1000  # Maximum number of molecules to keep in memory
        
        # Load metadata
        self._load_metadata()
        
    def _load_metadata(self):
        """Load metadata without processing molecules"""
        logger.info("Loading ADMET dataset metadata")
        
        # Load TDC ADMET Benchmark Group
        group = admet_group(path=self.root)
        datasets = group.dataset_names
        self.task_names = datasets
        
        # Process each dataset's metadata
        for i, dataset_name in enumerate(datasets):
            dataset = group.get(dataset_name)
            train_df = dataset['train_val']
            
            logger.info(f"Loading metadata for dataset: {dataset_name} ({len(train_df)} molecules)")
            
            # Determine task type from values
            values = train_df['Y'].dropna().values
            is_classification = np.all(np.equal(np.mod(values, 1), 0))
            task_type = 'classification' if is_classification else 'regression'
            self.task_types.append(task_type)
            
            # Calculate statistics for regression tasks
            if task_type == 'regression':
                self.task_means[i] = float(np.mean(values))
                self.task_stds[i] = float(np.std(values))
                if self.task_stds[i] < 1e-8:  # Avoid division by zero
                    logger.warning(f"Task {dataset_name} has near-zero standard deviation. Using 1.0 instead.")
                    self.task_stds[i] = 1.0
            
            # Store metadata for each molecule
            for _, row in train_df.iterrows():
                self.smiles_list.append(row['Drug'])
                self.task_ids.append(i)
                self.y_list.append(row['Y'])
        
        logger.info(f"Loaded metadata for {len(self.smiles_list)} molecules")
        
    def len(self):
        """Required method for PyG Dataset"""
        return len(self.smiles_list)
         
    def get(self, idx):
        """Get molecule graph, processing it if not already in cache (Required for PyG Dataset)"""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Process molecule if not in cache
        smiles = self.smiles_list[idx]
        task_id = self.task_ids[idx]
        task_name = self.task_names[task_id]
        task_type = self.task_types[task_id]
        y_value = self.y_list[idx]
        
        # Convert SMILES to graph
        graph = smiles_to_graph(smiles, self.config)
        
        if graph is None:
            # Create a dummy graph as fallback
            graph = Data(
                x=torch.zeros((1, 32)),  # Default feature size
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 7), dtype=torch.float),
                smiles=smiles
            )
        
        # Add metadata
        graph.task_id = task_id
        graph.task_name = task_name
        graph.task_type = task_type
        
        # Add target value
        if task_type == 'regression':
            normalized_y = (y_value - self.task_means[task_id]) / self.task_stds[task_id]
            graph.y = torch.tensor([normalized_y], dtype=torch.float)
            graph.raw_y = torch.tensor([y_value], dtype=torch.float)
        else:
            graph.y = torch.tensor([int(y_value)], dtype=torch.long)
        
        # Update cache (implement LRU cache behavior)
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = graph
        return graph
    
    def get_task_info(self):
        """Return task information"""
        task_ids = {i: name for i, name in enumerate(self.task_names)}
        
        return {
            'task_names': self.task_names,
            'task_types': self.task_types,
            'task_means': self.task_means,
            'task_stds': self.task_stds,
            'task_ids': task_ids
        }
    
    def get_split_indices(self, split_ratio=None, seed=None):
        """Get train/test split indices stratified by task"""
        if split_ratio is None:
            split_ratio = self.config.split_ratio
            
        if seed is None:
            seed = self.config.random_seed
            
        random.seed(seed)
        
        # Group by task
        task_indices = {}
        for i, task_id in enumerate(self.task_ids):
            if task_id not in task_indices:
                task_indices[task_id] = []
            task_indices[task_id].append(i)
        
        # Split each task
        train_indices = []
        test_indices = []
        
        for task_id, indices in task_indices.items():
            random.shuffle(indices)
            split_point = int(len(indices) * split_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
        
        logger.info(f"Split dataset: {len(train_indices)} training, {len(test_indices)} testing samples")
        return train_indices, test_indices