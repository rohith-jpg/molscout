"""Transparent nearest-neighbor lookup against a tiny bundled demo catalog."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molscout.chemistry import parse_smiles


@dataclass(frozen=True)
class ReferenceCompound:
    name: str
    smiles: str


# This deliberately small starter set makes the tool runnable offline. It is not
# a comprehensive approved-drug collection and must not be described as ChEMBL.
DEMO_REFERENCE_COMPOUNDS = (
    ReferenceCompound("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ReferenceCompound("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ReferenceCompound("ibuprofen", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
    ReferenceCompound("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ReferenceCompound("metformin", "CN(C)C(=N)NC(=N)N"),
)

_FINGERPRINTS = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def similar_known_drugs(smiles: str, *, limit: int = 3) -> dict[str, object]:
    """Return Tanimoto neighbors from the bundled teaching/demo reference set."""
    if limit < 1 or limit > len(DEMO_REFERENCE_COMPOUNDS):
        raise ValueError(f"limit must be between 1 and {len(DEMO_REFERENCE_COMPOUNDS)}")

    query = _FINGERPRINTS.GetFingerprint(parse_smiles(smiles))
    matches: list[dict[str, str | float]] = []
    for compound in DEMO_REFERENCE_COMPOUNDS:
        reference = Chem.MolFromSmiles(compound.smiles)
        if reference is None:  # pragma: no cover - static catalog guard
            continue
        score = DataStructs.TanimotoSimilarity(
            query, _FINGERPRINTS.GetFingerprint(reference)
        )
        matches.append(
            {
                "name": compound.name,
                "smiles": Chem.MolToSmiles(reference),
                "tanimoto": float(score),
            }
        )

    matches.sort(key=lambda item: (-float(item["tanimoto"]), str(item["name"])))
    return {
        "method": "Morgan radius 2, 2048 bits, Tanimoto similarity",
        "catalog": "molscout_demo_reference_v1",
        "catalog_size": len(DEMO_REFERENCE_COMPOUNDS),
        "catalog_note": (
            "Small offline teaching/demo set; not a comprehensive approved-drug index "
            "and not a source of efficacy or safety evidence."
        ),
        "matches": matches[:limit],
    }
