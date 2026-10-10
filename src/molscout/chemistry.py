"""RDKit-backed molecule parsing, descriptors, and fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


@dataclass(frozen=True)
class MoleculeProperties:
    canonical_smiles: str
    molecular_weight: float
    clogp: float
    h_bond_donors: int
    h_bond_acceptors: int
    tpsa: float
    rotatable_bonds: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitize a SMILES string, raising ValueError for invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def validate_smiles(smiles: str) -> dict[str, str | bool]:
    """Return validation status and a canonical SMILES representation."""
    mol = parse_smiles(smiles)
    return {"valid": True, "input_smiles": smiles, "canonical_smiles": Chem.MolToSmiles(mol)}


def compute_properties(smiles: str) -> MoleculeProperties:
    mol = parse_smiles(smiles)
    return MoleculeProperties(
        canonical_smiles=Chem.MolToSmiles(mol),
        molecular_weight=float(Descriptors.MolWt(mol)),
        clogp=float(Crippen.MolLogP(mol)),
        h_bond_donors=int(Lipinski.NumHDonors(mol)),
        h_bond_acceptors=int(Lipinski.NumHAcceptors(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
    )


def morgan_fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a dense binary Morgan fingerprint suitable for scikit-learn/XGBoost."""
    if radius < 0 or n_bits < 1:
        raise ValueError("radius must be >= 0 and n_bits must be >= 1")
    mol = parse_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    result = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result


def fingerprint_matrix(smiles: list[str], *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if not smiles:
        raise ValueError("At least one SMILES string is required")
    return np.stack([morgan_fingerprint(smi, radius=radius, n_bits=n_bits) for smi in smiles])
