"""Split-conformal intervals for the AqSolDB regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from molscout.benchmark import (
    _columns,
    _features_and_labels,
    _features_labels_and_exclusions,
    _make_model,
)


def evaluate_split_conformal(
    train: Any,
    calibration: Any,
    test: Any,
    output_dir: Path,
    seed: int = 1,
    confidence_level: float = 0.9,
) -> dict[str, Any]:
    """Fit only on train, calibrate on validation, then measure coverage on test.

    TDC's validation partition is used as the calibration set. The benchmark's
    scaffold-held-out test partition is used only for final measurement. Since
    scaffold splits can shift chemistry distributions, empirical test coverage
    is reported rather than promised by the exchangeability-based guarantee.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    try:
        from mapie.regression import SplitConformalRegressor
    except ImportError as exc:
        raise RuntimeError("Install MAPIE with pip install -e '.[uncertainty]' ") from exc

    train_x, train_y = _features_and_labels(train)
    calibration_x, calibration_y = _features_and_labels(calibration)
    test_x, test_y, excluded_test_rows = _features_labels_and_exclusions(test)
    estimator = _make_model("regression", seed)
    estimator.fit(train_x, train_y)
    conformal = SplitConformalRegressor(
        estimator=estimator, confidence_level=confidence_level, prefit=True
    ).conformalize(calibration_x, calibration_y)
    point_predictions, intervals = conformal.predict_interval(test_x)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    covered = (test_y >= lower) & (test_y <= upper)
    width = upper - lower
    metrics: dict[str, Any] = {
        "dataset": "Solubility_AqSolDB",
        "model": "Morgan-r2-2048 + XGBoost + MAPIE split conformal",
        "seed": seed,
        "confidence_level": confidence_level,
        "n_train": int(len(train_y)),
        "n_calibration": int(len(calibration_y)),
        "n_test_raw": int(len(test)),
        "n_test": int(len(test_y)),
        "n_test_excluded": len(excluded_test_rows),
        "test_mae": float(mean_absolute_error(test_y, point_predictions)),
        "test_empirical_coverage": float(np.mean(covered)),
        "test_mean_interval_width": float(np.mean(width)),
        "coverage_scope": "empirical scaffold-held-out test coverage",
        "coverage_caveat": (
            "Split-conformal finite-sample coverage relies on exchangeability; "
            "scaffold shift can violate that assumption."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_col, _ = _columns(test)
    excluded_indices = {row["row"] for row in excluded_test_rows}
    valid_test_smiles = [
        str(smiles)
        for row_index, smiles in enumerate(test[smiles_col].astype(str))
        if row_index not in excluded_indices
    ]
    pd.DataFrame(
        {
            "smiles": valid_test_smiles,
            "y_true": test_y,
            "y_pred": point_predictions,
            "lower_bound": lower,
            "upper_bound": upper,
            "covered": covered,
            "interval_width": width,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "excluded_test_rows.json").write_text(
        json.dumps(excluded_test_rows, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_tdc_conformal(
    output_dir: Path, seeds: list[int], confidence_level: float = 0.9
) -> list[dict[str, Any]]:
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more unique, non-negative seeds")
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra: pip install -e '.[benchmark]'") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(output_dir / "tdc_cache"))
    benchmark = group.get("Solubility_AqSolDB")
    test = benchmark["test"]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train, calibration = group.get_train_valid_split(
            benchmark="Solubility_AqSolDB", split_type="default", seed=seed
        )
        rows.append(
            evaluate_split_conformal(
                train,
                calibration,
                test,
                output_dir / "Solubility_AqSolDB" / f"seed-{seed}",
                seed,
                confidence_level,
            )
        )
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conformal"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--confidence-level", type=float, default=0.9)
    args = parser.parse_args()
    rows = run_tdc_conformal(args.output_dir, args.seeds, args.confidence_level)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
