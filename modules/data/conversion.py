import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Optional
from modules.data.features import get_atom_features, get_bond_features

logger = logging.getLogger(__name__)

def smiles_to_graph(
    smiles: str, 
    config,
    skip_check: bool = False
) -> Optional[Data]:
    """
    Convert SMILES to graph with enhanced features
    
    Args:
        smiles: SMILES string
        config: Configuration object
        skip_check: Skip checking for problematic SMILES patterns
        
    Returns:
        PyG Data object or None if conversion fails
    """
    try:
        # Skip problematic SMILES patterns (inorganic compounds, salts, etc.)
        if not skip_check and '[' in smiles and any(element in smiles for element in config.skip_elements):
            logger.info(f"Skipping inorganic/metallic compound: {smiles}")
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
            
        # Skip molecules with no atoms
        if mol.GetNumAtoms() == 0:
            logger.warning(f"Molecule has no atoms: {smiles}")
            return None
            
        # Preprocessing
        if config.add_hydrogens:
            try:
                mol = Chem.AddHs(mol)
            except Chem.rdchem.KekulizeException as e:
                logger.warning(f"Kekulization error adding hydrogens to {smiles}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Error adding hydrogens to {smiles}: {e}")
                return None
                
        if config.calculate_charges:
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except ValueError as e:
                logger.warning(f"Value error computing charges for {smiles}: {e}")
                # Continue without charges
            except Exception as e:
                logger.warning(f"Error computing charges for {smiles}: {e}")
                # Continue without charges
                
        if config.use_3d_coordinates:
            try:
                # Generate 3D coordinates
                AllChem.EmbedMolecule(mol, randomSeed=config.random_seed)
                AllChem.MMFFOptimizeMolecule(mol)
            except ValueError as e:
                logger.warning(f"Value error generating 3D coordinates for {smiles}: {e}")
                # Continue without 3D coordinates
            except Exception as e:
                logger.warning(f"Error generating 3D coordinates for {smiles}: {e}")
                # Continue without 3D coordinates
            
        # Node features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom, config.atom_types))
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
    except Chem.rdchem.KekulizeException as e:
        logger.warning(f"Kekulization error processing SMILES {smiles}: {e}")
        return None
    except ValueError as e:
        logger.warning(f"Value error processing SMILES {smiles}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error processing SMILES {smiles}: {e}")
        return None