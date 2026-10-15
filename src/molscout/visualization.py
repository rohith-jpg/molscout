"""Molecule drawings for model-attribution views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO

from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from molscout.chemistry import parse_smiles


def molecule_attribution_image(
    smiles: str,
    atom_attributions: Sequence[Mapping[str, object]],
    *,
    size: tuple[int, int] = (560, 360),
    legend: str = "",
) -> Image.Image:
    """Draw positive attributions in red and negative attributions in blue."""
    molecule = parse_smiles(smiles)
    scores = {
        int(atom["atom_index"]): float(atom["attribution"])
        for atom in atom_attributions
    }
    scale = max((abs(value) for value in scores.values()), default=0.0)
    colors: dict[int, tuple[float, float, float]] = {}
    for atom_index, score in scores.items():
        intensity = min(abs(score) / scale, 1.0) if scale else 0.0
        pale = 1.0 - 0.72 * intensity
        colors[atom_index] = (1.0, pale, pale) if score >= 0 else (pale, pale, 1.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        legend=legend,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).copy()

