"""Measure the deterministic local analysis path; no performance is prefilled."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from molscout.analysis import analyze_batch

SAMPLE_SMILES = [
    "CCO",
    "CC(=O)O",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
]


def measure(repeats: int) -> dict[str, object]:
    if repeats < 2:
        raise ValueError("Use at least two timed repetitions")
    for _ in range(3):
        analyze_batch(SAMPLE_SMILES)
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        analyze_batch(SAMPLE_SMILES)
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    median = statistics.median(durations)
    return {
        "scope": "in-process deterministic RDKit analysis; excludes browser/network time",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported by this OS",
        "batch_size": len(SAMPLE_SMILES),
        "warmup_runs": 3,
        "timed_runs": repeats,
        "median_seconds_per_batch": median,
        "p95_seconds_per_batch": ordered[p95_index],
        "median_molecules_per_second": len(SAMPLE_SMILES) / median,
        "samples": SAMPLE_SMILES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.repeats)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
