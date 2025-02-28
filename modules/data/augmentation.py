import random
from rdkit import Chem
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def augment_molecule(smiles: str, max_attempts: int = 5) -> Optional[str]:
    """
    Simple data augmentation by atom/bond removal if possible
    
    Args:
        smiles: SMILES string to augment
        max_attempts: Maximum number of attempts to try augmentation
        
    Returns:
        Augmented SMILES string or None if augmentation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() <= 5:  # Too small to augment
            return None
            
        # Randomly choose augmentation type
        aug_type = random.choice(['remove_atom', 'remove_bond'])
        
        if aug_type == 'remove_atom':
            # Try to remove a non-essential atom
            for _ in range(max_attempts):
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
                except Chem.rdchem.KekulizeException:
                    continue
                except Exception:
                    continue
        
        elif aug_type == 'remove_bond':
            # Try to remove a non-essential bond
            for _ in range(max_attempts):
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
                except Chem.rdchem.KekulizeException:
                    continue
                except Exception:
                    continue
        
        return None  # No valid augmentation found
    except Chem.rdchem.KekulizeException as e:
        logger.debug(f"Kekulization error augmenting {smiles}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error augmenting {smiles}: {e}")
        return None