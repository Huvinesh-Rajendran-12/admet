import torch
import numpy as np
from tdc.benchmark_group import admet_group
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from typing import List, Tuple, Dict, Optional, Union
import os
import pickle

# Enhanced atom feature extraction
def get_atom_features(atom):
    """
    Extract comprehensive atom features
    
    Args:
        atom: RDKit atom object
    
    Returns:
        List of atom features
    """
    # Basic features
    features = [
        atom.GetAtomicNum(),                      # Atomic number
        atom.GetDegree(),                         # Number of bonds
        atom.GetFormalCharge(),                   # Formal charge
        atom.GetChiralTag(),                      # Chirality
        atom.GetTotalNumHs(),                     # Total number of Hs
        atom.GetHybridization(),                  # Hybridization state
        atom.GetIsAromatic() * 1,                 # Aromaticity flag
        atom.IsInRing() * 1,                      # In ring flag
        atom.GetNumRadicalElectrons(),            # Number of radical electrons
        atom.GetExplicitValence(),                # Explicit valence
        atom.GetImplicitValence(),                # Implicit valence
    ]
    
    # One-hot encoding for atom type (common atoms in drug-like molecules)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Na', 'K', 'Li', 'Ca']
    features.extend([1 if atom.GetSymbol() == t else 0 for t in atom_types])
    
    # One-hot encoding for hybridization
    hybridization_types = [Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3, 
                          Chem.rdchem.HybridizationType.SP3D, 
                          Chem.rdchem.HybridizationType.SP3D2]
    features.extend([1 if atom.GetHybridization() == h else 0 for h in hybridization_types])
    
    return features

# Enhanced bond feature extraction
def get_bond_features(bond):
    """
    Extract comprehensive bond features
    
    Args:
        bond: RDKit bond object
    
    Returns:
        List of bond features
    """
    # Bond type as one-hot
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                 Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    features = [1 if bond.GetBondType() == t else 0 for t in bond_types]
    
    # Additional features
    features.extend([
        bond.GetIsConjugated() * 1,  # Is conjugated
        bond.IsInRing() * 1,         # Is in ring
        bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE  # Has stereochemistry
    ])
    
    return features

def smiles_to_graph(smiles, add_h=True, calculate_charges=True, use_3d=False):
    """
    Convert SMILES to graph with enhanced features
    
    Args:
        smiles: SMILES string
        add_h: Whether to add hydrogens
        calculate_charges: Whether to calculate Gasteiger charges
        use_3d: Whether to use 3D coordinates as additional features
        
    Returns:
        PyG Data object or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Preprocessing
        if add_h:
            mol = Chem.AddHs(mol)
        if calculate_charges:
            AllChem.ComputeGasteigerCharges(mol)
        if use_3d:
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
        # Node features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feats = get_bond_features(bond)
            
            # Add edges in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            edge_features.append(bond_feats)
            edge_features.append(bond_feats)  # Same features for both directions
            
        if len(edge_indices) == 0:  # Molecule with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(get_bond_features(None))), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
        # Create data object with all features
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )
        
        return data
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

class ADMETDataset(Dataset):
    """PyTorch Geometric Dataset for ADMET data"""
    
    def __init__(self, 
                 root='data', 
                 transform=None, 
                 pre_transform=None,
                 add_h=True,
                 calculate_charges=True,
                 use_3d=False,
                 augment=False,
                 cache_graphs=True):
        """
        Args:
            root: Root directory
            transform: PyG transform
            pre_transform: PyG pre-transform
            add_h: Whether to add hydrogens
            calculate_charges: Whether to calculate charges
            use_3d: Whether to use 3D coordinates
            augment: Whether to use data augmentation
            cache_graphs: Whether to cache processed graphs
        """
        self.add_h = add_h
        self.calculate_charges = calculate_charges
        self.use_3d = use_3d
        self.augment = augment
        self.cache_graphs = cache_graphs
        self.cache_file = os.path.join(root, 'processed_graphs.pkl')
        self.data_list = []
        self.task_types = []
        self.task_names = []
        self.task_means = {}
        self.task_stds = {}
        
        super().__init__(root, transform, pre_transform)
        
        # Load or process data
        if self.cache_graphs and os.path.exists(self.cache_file):
            self._load_cached_data()
        else:
            self._process_data()
            if self.cache_graphs:
                self._cache_data()
    
    def _load_cached_data(self):
        """Load cached processed data"""
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            self.data_list = cached_data['data_list']
            self.task_types = cached_data['task_types']
            self.task_names = cached_data['task_names']
            self.task_means = cached_data['task_means']
            self.task_stds = cached_data['task_stds']
    
    def _cache_data(self):
        """Cache processed data to disk"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'data_list': self.data_list,
                'task_types': self.task_types,
                'task_names': self.task_names,
                'task_means': self.task_means,
                'task_stds': self.task_stds
            }, f)
    
    def _process_data(self):
        """Process ADMET data from TDC"""
        # Load TDC ADMET Benchmark Group
        group = admet_group(path=self.root)
        datasets = group.dataset_names
        self.task_names = datasets
        
        # Determine task types based on actual data
        self.task_types = []
        self.task_means = {}
        self.task_stds = {}
        
        # Process each dataset
        for i, dataset_name in enumerate(datasets):
            dataset = group.get(dataset_name)
            train_df = dataset['train_val']
            
            # Determine task type from first valid value
            for _, row in train_df.iterrows():
                if isinstance(row['Y'], (int, bool)) or (isinstance(row['Y'], float) and row['Y'].is_integer()):
                    task_type = 'classification'
                else:
                    task_type = 'regression'
                self.task_types.append(task_type)
                break
            
            # Calculate statistics for regression tasks
            if task_type == 'regression':
                values = train_df['Y'].dropna().values
                self.task_means[i] = float(np.mean(values))
                self.task_stds[i] = float(np.std(values))
                if self.task_stds[i] == 0:  # Avoid division by zero
                    self.task_stds[i] = 1.0
            
            # Process each molecule
            for _, row in train_df.iterrows():
                smiles = row['Drug']
                graph = smiles_to_graph(
                    smiles, 
                    add_h=self.add_h,
                    calculate_charges=self.calculate_charges,
                    use_3d=self.use_3d
                )
                
                if graph is not None:
                    # Add metadata
                    graph.task_id = i
                    graph.task_name = dataset_name
                    graph.task_type = task_type
                    
                    # Normalize regression values
                    if task_type == 'regression':
                        normalized_y = (row['Y'] - self.task_means[i]) / self.task_stds[i]
                        graph.y = torch.tensor([normalized_y], dtype=torch.float)
                        graph.raw_y = torch.tensor([row['Y']], dtype=torch.float)
                    else:
                        graph.y = torch.tensor([int(row['Y'])], dtype=torch.long)
                    
                    self.data_list.append(graph)
                    
                    # Data augmentation for training set
                    if self.augment:
                        augmented = self._augment_molecule(smiles)
                        if augmented is not None:
                            aug_graph = smiles_to_graph(
                                augmented,
                                add_h=self.add_h,
                                calculate_charges=self.calculate_charges,
                                use_3d=self.use_3d
                            )
                            
                            if aug_graph is not None:
                                aug_graph.task_id = graph.task_id
                                aug_graph.task_name = graph.task_name
                                aug_graph.task_type = graph.task_type
                                aug_graph.y = graph.y.clone()
                                if task_type == 'regression':
                                    aug_graph.raw_y = graph.raw_y.clone()
                                self.data_list.append(aug_graph)
    
    def _augment_molecule(self, smiles):
        """Simple data augmentation by atom/bond removal if possible"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() <= 5:  # Too small to augment
                return None
                
            # Randomly choose augmentation type
            aug_type = random.choice(['remove_atom', 'remove_bond'])
            
            if aug_type == 'remove_atom':
                # Try to remove a non-essential atom
                for _ in range(5):  # Try 5 times
                    atom_idx = random.randint(0, mol.GetNumAtoms()-1)
                    atom = mol.GetAtomWithIdx(atom_idx)
                    
                    # Skip if removing this atom would break the molecule
                    if atom.GetDegree() > 2 or atom.IsInRing():
                        continue
                        
                    # Create editable mol and remove atom
                    em = Chem.EditableMol(mol)
                    em.RemoveAtom(atom_idx)
                    new_mol = em.GetMol()
                    
                    # Check if valid
                    try:
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles and new_smiles != smiles:
                            return new_smiles
                    except:
                        continue
            
            elif aug_type == 'remove_bond':
                # Try to remove a non-essential bond
                for _ in range(5):  # Try 5 times
                    if mol.GetNumBonds() == 0:
                        break
                        
                    bond_idx = random.randint(0, mol.GetNumBonds()-1)
                    bond = mol.GetBondWithIdx(bond_idx)
                    
                    # Skip if removing this bond would break the molecule
                    if bond.IsInRing():
                        continue
                        
                    # Create editable mol and remove bond
                    em = Chem.EditableMol(mol)
                    em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    new_mol = em.GetMol()
                    
                    # Check if valid
                    try:
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles and new_smiles != smiles:
                            return new_smiles
                    except:
                        continue
            
            return None  # No valid augmentation found
        except:
            return None
    
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
            'task_ids': task_ids  # Added task IDs mapping
        }
    
    def get_split_indices(self, split_ratio=0.8, seed=42):
        """Get train/test split indices stratified by task"""
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
        
        return train_indices, test_indices

def prepare_data(root='data', 
                add_h=True, 
                calculate_charges=True, 
                use_3d=False, 
                augment=False,
                cache_graphs=True):
    """
    Prepare ADMET dataset with enhanced features
    
    Args:
        root: Root directory
        add_h: Whether to add hydrogens
        calculate_charges: Whether to calculate charges
        use_3d: Whether to use 3D coordinates
        augment: Whether to use data augmentation
        cache_graphs: Whether to cache processed graphs
        
    Returns:
        ADMETDataset object
    """
    dataset = ADMETDataset(
        root=root,
        add_h=add_h,
        calculate_charges=calculate_charges,
        use_3d=use_3d,
        augment=augment,
        cache_graphs=cache_graphs
    )
    
    return dataset
