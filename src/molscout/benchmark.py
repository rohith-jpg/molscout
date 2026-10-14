"""TDC benchmark definitions and Morgan/XGBoost baseline runner."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from molscout.chemistry import fingerprint_matrix

TASKS: dict[str, dict[str, str]] = {
    "hERG": {"kind": "classification", "metric": "AUROC"},
    "AMES": {"kind": "classification", "metric": "AUROC"},
    "BBB_Martins": {"kind": "classification", "metric": "AUROC"},
    "Solubility_AqSolDB": {"kind": "regression", "metric": "MAE"},
}

BASELINE_PARAMETERS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": 4,
}


def _columns(frame: Any) -> tuple[str, str]:
    """Find TDC molecule and target columns without assuming capitalization."""
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    smiles_col = next(
        (lowered[k] for k in ("drug", "smiles", "canonical_smiles") if k in lowered), None
    )
    target_col = next((lowered[k] for k in ("y", "label", "target") if k in lowered), None)
    if smiles_col is None or target_col is None:
        raise ValueError(f"Could not find molecule/target columns in {list(frame.columns)}")
    return smiles_col, target_col


def _features_labels_and_exclusions(
    frame: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    smiles_col, target_col = _columns(frame)
    smiles = frame[smiles_col].astype(str).tolist()
    y = frame[target_col].to_numpy(dtype=float)
    canonical_smiles: list[str] = []
    labels: list[float] = []
    excluded: list[dict[str, Any]] = []
    from rdkit import Chem

    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            excluded.append({"row": index, "smiles": smi, "reason": "RDKit parse failed"})
            continue
        if not np.isfinite(y[index]):
            raise ValueError(
                f"Missing/non-finite label at row {index}; "
                "refusing to silently drop benchmark examples"
            )
        canonical_smiles.append(Chem.MolToSmiles(mol))
        labels.append(float(y[index]))
    if not canonical_smiles:
        raise ValueError("No valid molecule/label rows were found")
    return fingerprint_matrix(canonical_smiles), np.asarray(labels), excluded


def _features_and_labels(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Create features and fail if any row needs exclusion (useful for train data)."""
    features, labels, excluded = _features_labels_and_exclusions(frame)
    if excluded:
        first = excluded[0]
        raise ValueError(
            f"Invalid SMILES at row {first['row']}; "
            "refusing to silently drop benchmark examples"
        )
    return features, labels


def _make_model(kind: str, seed: int):
    if kind == "classification":
        return XGBClassifier(
            **BASELINE_PARAMETERS,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
        )
    return XGBRegressor(
        **BASELINE_PARAMETERS,
        objective="reg:squarederror",
        random_state=seed,
    )


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize independent runs without hiding the per-seed measurements."""
    aggregate: list[dict[str, Any]] = []
    for task_name, task in TASKS.items():
        task_rows = [row for row in rows if row["dataset"] == task_name]
        values = np.asarray([row["value"] for row in task_rows], dtype=float)
        if not len(values):
            continue
        aggregate.append(
            {
                "dataset": task_name,
                "metric": task["metric"],
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "std_convention": "population standard deviation across supplied seeds",
                "n_runs": int(len(values)),
                "seeds": [int(row["seed"]) for row in task_rows],
            }
        )
    return aggregate


def run_baselines(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Train on TDC train/validation splits and evaluate official test rows."""
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more unique, non-negative random seeds")
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra: pip install -e '.[benchmark]'") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(output_dir / "tdc_cache"))
    rows: list[dict[str, Any]] = []
    for task_name, task in TASKS.items():
        benchmark = group.get(task_name)
        test = benchmark["test"]
        test_x, test_y, excluded_test_rows = _features_labels_and_exclusions(test)
        for seed in seeds:
            train, valid = group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed
            )
            train_x, train_y = _features_and_labels(train)
            # Validation is obtained using TDC's official split utility. The fixed
            # model settings below are intentional for this baseline, so it is not
            # used to choose model hyperparameters in this initial runner.
            _ = valid
            model = _make_model(task["kind"], seed)
            model.fit(train_x, train_y)
            if task["kind"] == "classification":
                prediction = model.predict_proba(test_x)[:, 1]
            else:
                prediction = model.predict(test_x)
            metric_value = (
                float(roc_auc_score(test_y, prediction))
                if task["kind"] == "classification"
                else float(mean_absolute_error(test_y, prediction))
            )
            row = {
                "dataset": task_name,
                "model": "Morgan-r2-2048 + XGBoost",
                "metric": task["metric"],
                "value": metric_value,
                "seed": seed,
                "n_train": int(len(train_y)),
                "n_test": int(len(test)),
                "n_test_scored": int(len(test_y)),
                "n_test_excluded_invalid_smiles": len(excluded_test_rows),
                "excluded_test_rows": excluded_test_rows,
            }
            rows.append(row)
            run_dir = output_dir / task_name / f"seed-{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(run_dir / "model.json")
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(row, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(_aggregate_metrics(rows), handle, indent=2)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("PyTDC", "rdkit", "numpy", "scikit-learn", "xgboost")
        },
        "benchmark_split": (
            "TDC ADMET Benchmark Group scaffold-held-out test; TDC default train/validation split"
        ),
        "invalid_test_rows": (
            "Rows that RDKit could not parse are explicitly excluded from metric calculation "
            "and listed per run; scores therefore use n_test_scored, not always the full TDC test set."
        ),
        "tasks": TASKS,
        "seeds": seeds,
        "model": "Morgan-r2-2048 + XGBoost",
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": 2048},
        "model_parameters": BASELINE_PARAMETERS,
        "metrics_file": "summary.json",
        "aggregate_file": "aggregate.json",
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    result = run_baselines(args.output_dir, args.seeds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

