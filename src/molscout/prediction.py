"""Load trained baseline artifacts for endpoint-specific ADMET estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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


def _seed_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("seed-"))
    except ValueError as exc:
        raise ValueError(f"Unexpected model artifact directory: {path.parent}") from exc


def predict_admet(
    smiles: str,
    endpoint: str,
    *,
    model_dir: Path = Path("outputs/baseline"),
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

    from xgboost import XGBClassifier, XGBRegressor

    spec = ENDPOINTS[endpoint]
    kind = spec["kind"]
    values: list[float] = []
    fingerprint = morgan_fingerprint(smiles).reshape(1, -1)
    for artifact in artifacts:
        model = XGBClassifier() if kind == "classification" else XGBRegressor()
        model.load_model(str(artifact))
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
