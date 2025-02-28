from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any

@dataclass
class ADMETConfig:
    """Configuration for ADMET data processing"""
    
    # Paths
    root_dir: str = 'data'
    cache_file: str = 'processed_graphs.pkl'
    
    # Feature extraction options
    add_hydrogens: bool = True
    calculate_charges: bool = True
    use_3d_coordinates: bool = False
    
    # Dataset options
    augment_data: bool = False
    cache_graphs: bool = True
    
    # Processing options
    n_jobs: int = 4  # Number of parallel jobs
    batch_size: int = 100  # Batch size for processing
    
    # Atom features
    atom_types: List[str] = field(default_factory=lambda: 
                                ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Na', 'K', 'Li', 'Ca'])
    
    # Molecules to skip (e.g., inorganic compounds, salts)
    skip_elements: List[str] = field(default_factory=lambda: 
                               ['Al', 'Fe', 'Zn', 'Mg', 'Cu', 'Na', 'K', 'Li', 'Ca', 'Ba', 'Sr', 
                                'Cs', 'Rb', 'Be', 'Ra', 'Hg', 'Cd', 'Pb', 'Mn', 'Co', 'Ni', 'Zr',
                                'Cr', 'Mo', 'W', 'V', 'Nb', 'Ta', 'Ti', 'Hf', 'Re', 'Os', 'Ir', 
                                'Pt', 'Pd', 'Rh', 'Ru', 'Au', 'Ag', 'Bi', 'Sb', 'As', 'Sn', 'Pb',
                                'Te', 'Se', 'Si', 'B', 'Ga', 'In', 'Tl', 'Ge', 'P', 'La', 'Ce', 
                                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                                'Yb', 'Lu', 'Y', 'Sc', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'])
    
    # Training settings
    split_ratio: float = 0.8
    random_seed: int = 42
    
    # Logging
    log_level: str = 'INFO'