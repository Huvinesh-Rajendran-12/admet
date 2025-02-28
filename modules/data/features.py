import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def get_atom_features(atom, atom_types=None) -> List[Union[int, float]]:
    """
    Extract comprehensive atom features
    
    Args:
        atom: RDKit atom object
        atom_types: List of atom types to use for one-hot encoding
    
    Returns:
        List of atom features
    """
    if atom_types is None:
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Na', 'K', 'Li', 'Ca']
    
    try:
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
        
        # Add Gasteiger charge if available
        if atom.HasProp('_GasteigerCharge'):
            try:
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                features.append(gasteiger_charge)
            except (ValueError, TypeError):
                features.append(0.0)
        else:
            features.append(0.0)
            
        # One-hot encoding for atom type (common atoms in drug-like molecules)
        features.extend([1 if atom.GetSymbol() == t else 0 for t in atom_types])
        
        # One-hot encoding for hybridization
        hybridization_types = [Chem.rdchem.HybridizationType.SP, 
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3, 
                              Chem.rdchem.HybridizationType.SP3D, 
                              Chem.rdchem.HybridizationType.SP3D2]
        features.extend([1 if atom.GetHybridization() == h else 0 for h in hybridization_types])
        
        return features
    except Exception as e:
        logger.warning(f"Error extracting atom features: {e}")
        # Return zero features as fallback
        default_length = 11 + 1 + len(atom_types) + 5  # Basic + charge + atom types + hybridization
        return [0] * default_length

def get_bond_features(bond) -> List[int]:
    """
    Extract comprehensive bond features
    
    Args:
        bond: RDKit bond object
    
    Returns:
        List of bond features
    """
    # Handle the case when bond is None
    if bond is None:
        # Return default features (all zeros)
        return [0, 0, 0, 0, 0, 0, 0]
    
    try:
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
    except Exception as e:
        logger.warning(f"Error extracting bond features: {e}")
        return [0, 0, 0, 0, 0, 0, 0]  # Default values