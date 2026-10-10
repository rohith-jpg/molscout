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
        test_frame = _chemprop_frame(benchmark["test"])
        for seed in seeds:
            train, valid = group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed
            )
            run_dir = output_dir / task_name / f"seed-{seed}"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            split_paths = (data_dir / "train.csv", data_dir / "valid.csv", data_dir / "test.csv")
            for frame, path in zip((train, valid, test_frame), split_paths, strict=True):
                _chemprop_frame(frame).to_csv(path, index=False)

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
            run = {
                "dataset": task_name,
                "model": "Chemprop MPNN ensemble",
                "ensemble_size": ensemble_size,
                "seed": seed,
                "n_train": len(train),
                "n_valid": len(valid),
                "n_test": len(test_frame),
                "status": "trained",
                "predictions": "not yet exported; use `molscout chemprop-predict` in next milestone",
            }
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
