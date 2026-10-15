"""Load trained baseline artifacts for endpoint-specific ADMET estimates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from molscout.chemistry import morgan_fingerprint, parse_smiles

ENDPOINTS: dict[str, dict[str, str | None]] = {
    "hERG": {"kind": "classification", "metric": "AUROC", "unit": "probability"},
    "AMES": {"kind": "classification", "metric": "AUROC", "unit": "probability"},
    "BBB_Martins": {"kind": "classification", "metric": "AUROC", "unit": "probability"},
    "Solubility_AqSolDB": {
        "kind": "regression",
        "metric": "MAE",
        "unit": "log10(mol/L)",
    },
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "baseline"


def _seed_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("seed-"))
    except ValueError as exc:
        raise ValueError(f"Unexpected model artifact directory: {path.parent}") from exc


@lru_cache(maxsize=32)
def _load_model(path: str, kind: str) -> Any:
    """Load each small model once per app process."""
    from xgboost import XGBClassifier, XGBRegressor

    model = XGBClassifier() if kind == "classification" else XGBRegressor()
    model.load_model(path)
    return model


def predict_admet(
    smiles: str,
    endpoint: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Predict with every saved baseline seed and report their uncalibrated spread.

    Ensemble standard deviation here describes disagreement among independently
    trained baseline models. It is not a calibrated confidence interval.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; choose one of {sorted(ENDPOINTS)}")

    parse_smiles(smiles)  # Validate before loading model artifacts or computing features.
    artifacts = sorted(
        (model_dir / endpoint).glob("seed-*/model.json"),
        key=_seed_from_path,
    )
    if not artifacts:
        raise FileNotFoundError(
            f"No trained models found for {endpoint} under {model_dir}. "
            "Run the TDC baseline benchmark first."
        )

    spec = ENDPOINTS[endpoint]
    kind = spec["kind"]
    values: list[float] = []
    fingerprint = morgan_fingerprint(smiles).reshape(1, -1)
    for artifact in artifacts:
        model = _load_model(str(artifact.resolve()), str(kind))
        if kind == "classification":
            raw = model.predict_proba(fingerprint)[0, 1]
        else:
            raw = model.predict(fingerprint)[0]
        values.append(float(raw))

    array = np.asarray(values, dtype=float)
    return {
        "endpoint": endpoint,
        "kind": kind,
        "metric": spec["metric"],
        "unit": spec["unit"],
        "prediction": float(np.mean(array)),
        "ensemble_std": float(np.std(array)),
        "ensemble_std_meaning": "between-seed model spread; not calibrated uncertainty",
        "seed_predictions": [
            {"seed": _seed_from_path(path), "prediction": value}
            for path, value in zip(artifacts, values, strict=True)
        ],
        "model": "Morgan-r2-2048 + XGBoost",
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": 2048},
        "artifact_paths": [path.as_posix() for path in artifacts],
    }


def explain_prediction(
    smiles: str,
    endpoint: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Map mean XGBoost feature contributions back to Morgan atom environments.

    Explanations are available for classification endpoints only. Contributions
    are on each model's raw-margin (log-odds) scale and are not causal chemistry
    explanations. Hashed-feature collisions are shared evenly across matching
    Morgan environments, then across atoms in each environment.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; choose one of {sorted(ENDPOINTS)}")
    if ENDPOINTS[endpoint]["kind"] != "classification":
        raise ValueError("Atom attribution is currently available for toxicity classifiers only")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(molecule)
    artifacts = sorted(
        (model_dir / endpoint).glob("seed-*/model.json"),
        key=_seed_from_path,
    )
    if not artifacts:
        raise FileNotFoundError(
            f"No trained models found for {endpoint} under {model_dir}. "
            "Run the TDC baseline benchmark first."
        )

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.CollectBitInfoMap()
    generator.GetFingerprint(molecule, additionalOutput=additional)
    bit_info = additional.GetBitInfoMap()
    fingerprint = morgan_fingerprint(canonical_smiles).astype(np.float32).reshape(1, -1)

    from xgboost import DMatrix

    mean_contributions = np.zeros(2049, dtype=np.float64)
    margins: list[float] = []
    for artifact in artifacts:
        model = _load_model(str(artifact.resolve()), "classification")
        contributions = model.get_booster().predict(
            DMatrix(fingerprint), pred_contribs=True
        )[0]
        margins.append(float(model.get_booster().predict(DMatrix(fingerprint), output_margin=True)[0]))
        mean_contributions += contributions
    mean_contributions /= len(artifacts)

    atom_scores = np.zeros(molecule.GetNumAtoms(), dtype=np.float64)
    for bit_index, feature_score in enumerate(mean_contributions[:-1]):
        environments = bit_info.get(bit_index, ())
        if not environments:
            continue
        per_environment = float(feature_score) / len(environments)
        for center, radius in environments:
            bond_ids = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center)
            atom_ids = {center}
            for bond_id in bond_ids:
                bond = molecule.GetBondWithIdx(bond_id)
                atom_ids.add(bond.GetBeginAtomIdx())
                atom_ids.add(bond.GetEndAtomIdx())
            per_atom = per_environment / len(atom_ids)
            for atom_id in atom_ids:
                atom_scores[atom_id] += per_atom

    atoms = [
        {
            "atom_index": index,
            "element": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "attribution": float(atom_scores[index]),
        }
        for index, atom in enumerate(molecule.GetAtoms())
    ]
    atoms.sort(key=lambda item: -abs(item["attribution"]))
    mean_margin = float(np.mean(margins))
    mean_bias = float(np.mean([mean_contributions[-1]]))
    unmapped_feature_indices = [
        bit_index
        for bit_index in range(2048)
        if bit_index not in bit_info and abs(mean_contributions[bit_index]) > 1e-12
    ]
    unmapped_feature_contribution = float(
        np.sum(mean_contributions[unmapped_feature_indices])
    )
    mapped_atom_contribution_sum = float(np.sum(atom_scores))
    positive_atoms = [atom for atom in atoms if atom["attribution"] > 0]
    negative_atoms = [atom for atom in atoms if atom["attribution"] < 0]
    return {
        "endpoint": endpoint,
        "canonical_smiles": canonical_smiles,
        "target_class": 1,
        "attribution_method": "mean XGBoost pred_contribs across saved seeds, mapped through Morgan radius-2 environments",
        "attribution_scale": "raw margin (log-odds), not probability points",
        "positive_direction": "increases the model output toward dataset class 1",
        "collision_handling": "each hashed feature contribution is shared evenly across matching environments and atoms",
        "mean_raw_margin": mean_margin,
        "mean_bias_contribution": mean_bias,
        "unmapped_feature_contribution": unmapped_feature_contribution,
        "unmapped_nonzero_feature_count": len(unmapped_feature_indices),
        "mapped_atom_contribution_sum": mapped_atom_contribution_sum,
        "raw_margin_reconstruction_error": float(
            mean_bias + mapped_atom_contribution_sum
            + unmapped_feature_contribution - mean_margin
        ),
        "atom_attributions": atoms,
        "most_positive_atoms": sorted(
            positive_atoms, key=lambda item: -item["attribution"]
        )[:5],
        "most_negative_atoms": sorted(
            negative_atoms, key=lambda item: item["attribution"]
        )[:5],
        "seed_count": len(artifacts),
        "warning": "Model attribution describes learned fingerprint associations; it does not establish a causal mechanism or experimental toxicity.",
    }

