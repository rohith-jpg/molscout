"""Train Chemprop ensembles on the same explicit TDC splits as the baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from molscout.benchmark import TASKS, _columns


def _chemprop_frame(frame: Any) -> pd.DataFrame:
    smiles_col, target_col = _columns(frame)
    from rdkit import Chem

    smiles_out: list[str] = []
    targets_out: list[float] = []
    for index, (smiles, target) in enumerate(
        zip(frame[smiles_col].astype(str), frame[target_col], strict=True)
    ):
        molecule = Chem.MolFromSmiles(smiles)
        value = float(target)
        if molecule is None or not np.isfinite(value):
            raise ValueError(f"Invalid molecule or target at row {index}; refusing to drop it")
        smiles_out.append(Chem.MolToSmiles(molecule))
        targets_out.append(value)
    if not smiles_out:
        raise ValueError("TDC returned an empty split")
    return pd.DataFrame({"SMILES": smiles_out, "Y": targets_out})


def build_train_command(
    task_name: str,
    split_paths: tuple[Path, Path, Path],
    output_dir: Path,
    seed: int,
    ensemble_size: int = 5,
) -> list[str]:
    """Build a CLI command using train/validation/test files without resplitting."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown benchmark task: {task_name}")
    if seed < 0 or ensemble_size < 1:
        raise ValueError("seed must be non-negative and ensemble_size must be positive")
    executable = shutil.which("chemprop")
    if executable is None:
        raise RuntimeError("Chemprop CLI not found; install with pip install -e '.[gnn]'")
    kind = TASKS[task_name]["kind"]
    return [
        executable,
        "train",
        "--data-path",
        *(str(path.resolve()) for path in split_paths),
        "--task-type",
        kind,
        "--smiles-columns",
        "SMILES",
        "--target-columns",
        "Y",
        "--ensemble-size",
        str(ensemble_size),
        "--pytorch-seed",
        str(seed),
        "--num-workers",
        "0",
        "--output-dir",
        str(output_dir.resolve()),
    ]


def build_predict_command(
    test_path: Path, model_dir: Path, output_path: Path
) -> list[str]:
    """Build a CLI command to ensemble all checkpoints from one Chemprop run."""
    executable = shutil.which("chemprop")
    if executable is None:
        raise RuntimeError("Chemprop CLI not found; install with pip install -e '.[gnn]'")
    all_checkpoints = sorted(
        [*model_dir.rglob("*.ckpt"), *model_dir.rglob("*.pt")],
        key=lambda path: path.as_posix(),
    )
    best_checkpoints = [path for path in all_checkpoints if path.stem == "best"]
    checkpoints = best_checkpoints or all_checkpoints
    if not checkpoints:
        raise FileNotFoundError(f"No Chemprop checkpoints found under {model_dir}")
    return [
        executable,
        "predict",
        "--test-path",
        str(test_path.resolve()),
        "--model-paths",
        *(str(path.resolve()) for path in checkpoints),
        "--preds-path",
        str(output_path.resolve()),
        "--smiles-columns",
        "SMILES",
        "--num-workers",
        "0",
    ]


def _chemprop_test_frame(frame: Any) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter invalid test SMILES with an audit trail; training rows stay strict."""
    smiles_col, target_col = _columns(frame)
    from rdkit import Chem

    smiles_out: list[str] = []
    targets_out: list[float] = []
    excluded: list[dict[str, Any]] = []
    for index, (smiles, target) in enumerate(
        zip(frame[smiles_col].astype(str), frame[target_col], strict=True)
    ):
        molecule = Chem.MolFromSmiles(smiles)
        value = float(target)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite test target at row {index}; refusing to drop it")
        if molecule is None:
            excluded.append({"row": index, "smiles": smiles, "reason": "RDKit parse failed"})
            continue
        smiles_out.append(Chem.MolToSmiles(molecule))
        targets_out.append(value)
    if not smiles_out:
        raise ValueError("TDC returned no valid test molecules")
    return pd.DataFrame({"SMILES": smiles_out, "Y": targets_out}), excluded


def _prediction_output_path(requested_path: Path) -> Path:
    """Resolve a CLI-added suffix while avoiding its optional individual file."""
    if requested_path.exists():
        return requested_path
    candidates = sorted(
        path
        for path in requested_path.parent.glob(f"{requested_path.stem}*.csv")
        if "individual" not in path.stem.lower()
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Chemprop did not create {requested_path}; found {candidates}"
    )


def evaluate_chemprop_predictions(
    task_name: str,
    test_frame: pd.DataFrame,
    prediction_path: Path,
    output_path: Path,
    *,
    seed: int,
    ensemble_size: int,
    n_train: int,
    n_valid: int,
    n_test_raw: int,
    n_test_excluded: int,
) -> dict[str, Any]:
    """Score the Chemprop ensemble mean on the fixed TDC test rows."""
    frame = pd.read_csv(_prediction_output_path(prediction_path))
    if len(frame) != len(test_frame):
        raise ValueError(
            f"Chemprop returned {len(frame)} rows for {len(test_frame)} valid test molecules"
        )
    prediction_columns = [
        column for column in frame.columns if str(column).startswith("pred_")
    ]
    # Chemprop 2.3 writes the ensemble mean under the task's target column
    # (here `Y`) and per-model predictions to a separate *_individual.csv file.
    # The input test frame keeps the true labels separately, so `Y` in the
    # prediction output is the model prediction in this CLI format.
    if not prediction_columns and "Y" in frame.columns:
        prediction_columns = ["Y"]
    if not prediction_columns:
        raise ValueError(f"Chemprop output has no prediction column: {list(frame.columns)}")
    y_true = test_frame["Y"].to_numpy(dtype=float)
    y_pred = frame[prediction_columns[0]].to_numpy(dtype=float)
    metric_name = TASKS[task_name]["metric"]
    if TASKS[task_name]["kind"] == "classification":
        if len(np.unique(y_true)) < 2:
            raise ValueError(f"Cannot compute AUROC for {task_name}: test has one class")
        score = float(roc_auc_score(y_true, y_pred))
    else:
        score = float(mean_absolute_error(y_true, y_pred))

    scored = test_frame.copy()
    scored["y_pred"] = y_pred
    scored.to_csv(output_path, index=False)
    return {
        "dataset": task_name,
        "model": "Chemprop MPNN ensemble",
        "ensemble_size": ensemble_size,
        "seed": seed,
        "metric": metric_name,
        "value": score,
        "prediction_column": prediction_columns[0],
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test_raw": n_test_raw,
        "n_test": len(test_frame),
        "n_test_excluded": n_test_excluded,
        "status": "scored",
        "predictions_file": output_path.as_posix(),
    }


def run_chemprop_ensemble(
    output_dir: Path,
    task_names: list[str],
    seeds: list[int],
    ensemble_size: int = 5,
) -> list[dict[str, Any]]:
    """Train Chemprop ensembles on TDC train/validation/scaffold-test splits.

    Training and validation are returned by TDC's official split utility. The
    benchmark's fixed scaffold test set is supplied as the third file, so
    Chemprop does not generate a different split.
    """
    if not task_names or any(name not in TASKS for name in task_names):
        raise ValueError("Provide one or more recognized task names")
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more unique, non-negative seeds")
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra: pip install -e '.[benchmark]'") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(output_dir / "tdc_cache"))
    runs: list[dict[str, Any]] = []
    for task_name in task_names:
        benchmark = group.get(task_name)
        test_raw = benchmark["test"]
        test_frame, excluded_test = _chemprop_test_frame(test_raw)
        for seed in seeds:
            train, valid = group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed
            )
            run_dir = output_dir / task_name / f"seed-{seed}"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            split_paths = (data_dir / "train.csv", data_dir / "valid.csv", data_dir / "test.csv")
            _chemprop_frame(train).to_csv(split_paths[0], index=False)
            _chemprop_frame(valid).to_csv(split_paths[1], index=False)
            test_frame.to_csv(split_paths[2], index=False)
            (run_dir / "excluded_test_rows.json").write_text(
                json.dumps(excluded_test, indent=2), encoding="utf-8"
            )

            model_dir = run_dir / "model"
            command = build_train_command(
                task_name, split_paths, model_dir, seed, ensemble_size=ensemble_size
            )
            (run_dir / "command.json").write_text(
                json.dumps(command, indent=2), encoding="utf-8"
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=run_dir,
            )
            (run_dir / "train.log").write_text(
                completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Chemprop failed for {task_name}, seed {seed}; see {run_dir / 'train.log'}"
                )
            requested_predictions = run_dir / "predictions.csv"
            prediction_command = build_predict_command(
                split_paths[2], model_dir, requested_predictions
            )
            predicted = subprocess.run(
                prediction_command,
                check=False,
                capture_output=True,
                text=True,
                cwd=run_dir,
            )
            (run_dir / "predict.log").write_text(
                predicted.stdout + "\n--- STDERR ---\n" + predicted.stderr,
                encoding="utf-8",
            )
            if predicted.returncode:
                raise RuntimeError(
                    f"Chemprop prediction failed for {task_name}, seed {seed}; "
                    f"see {run_dir / 'predict.log'}"
                )
            run = evaluate_chemprop_predictions(
                task_name,
                test_frame,
                requested_predictions,
                run_dir / "test_predictions.csv",
                seed=seed,
                ensemble_size=ensemble_size,
                n_train=len(train),
                n_valid=len(valid),
                n_test_raw=len(test_raw),
                n_test_excluded=len(excluded_test),
            )
            runs.append(run)
            (run_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (output_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chemprop"))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--ensemble-size", type=int, default=5)
    args = parser.parse_args()
    result = run_chemprop_ensemble(
        args.output_dir, args.tasks, args.seeds, args.ensemble_size
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

