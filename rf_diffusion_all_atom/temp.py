import shutil
import numpy as np
from Bio import PDB
from copy import deepcopy

def calc_rmsd(xyz1, xyz2, eps=1e-6):
    """
    Calculates RMSD between two sets of atoms (L, 3)
    """
    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute otimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U
    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd, U

def extract_backbone_positions(pdb_file, bb=True):
    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)

    # Parse the structure
    structure = parser.get_structure('protein', pdb_file)

    # Initialize a list to hold backbone coordinates
    coords = []

    # Iterate over all residues in all chains
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip heteroatoms and water molecules
                if PDB.is_aa(residue, standard=True):
                    # Extract backbone atoms N, CA, C
                    try:
                        if bb:
                            n_coord = residue['N'].get_coord()
                            ca_coord = residue['CA'].get_coord()
                            c_coord = residue['C'].get_coord()
                            o_coord = residue['O'].get_coord()
                            
                            # Append the coordinates as a tuple (N, CA, C)
                            coords.append((n_coord, ca_coord, c_coord, o_coord))
                        else:
                            coords.append([atom.get_coord() for atom in residue if atom.element != 'H']) 
                    except KeyError:
                        # In case the residue is missing any backbone atom
                        continue

    return coords

def update_positions(structure, new_positions):
    atom_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.coord = new_positions[atom_index]
                    atom_index += 1
                    
def align_backbone_pos(target_pdb, ori_pdb, aligned_pdb):
    ori_pos = np.array(extract_backbone_positions(ori_pdb)).reshape(-1,3)
    target_pos = np.array(extract_backbone_positions(target_pdb)).reshape(-1,3)    
    rmsd, U = calc_rmsd(deepcopy(target_pos), deepcopy(ori_pos))
    aligned_pos = (ori_pos - ori_pos.mean(0)) @ U + target_pos.mean(0)

    shutil.copyfile(ori_pdb, aligned_pdb)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', aligned_pdb)
    update_positions(structure, aligned_pos)

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(aligned_pdb)
    return aligned_pos