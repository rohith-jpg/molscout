"""Render a real, bundled-model hERG attribution example for the README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molscout.prediction import explain_prediction  # noqa: E402
from molscout.visualization import molecule_attribution_image  # noqa: E402


def main() -> None:
    smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine
    result = explain_prediction(smiles, "hERG")
    output = ROOT / "docs" / "images" / "caffeine_herg_attribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    molecule_attribution_image(
        result["canonical_smiles"],
        result["atom_attributions"],
        size=(1100, 700),
        legend=(
            "Caffeine · hERG model attribution · red: toward dataset class 1 · "
            "blue: away from class 1"
        ),
    ).save(output)
    print(
        json.dumps(
            {
                "image": output.relative_to(ROOT).as_posix(),
                "endpoint": result["endpoint"],
                "mean_raw_margin": result["mean_raw_margin"],
                "raw_margin_reconstruction_error": result[
                    "raw_margin_reconstruction_error"
                ],
                "seed_count": result["seed_count"],
                "most_positive_atoms": result["most_positive_atoms"][:3],
                "most_negative_atoms": result["most_negative_atoms"][:3],
                "warning": result["warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

