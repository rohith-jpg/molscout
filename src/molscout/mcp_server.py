"""MCP stdio server exposing implemented chemistry tools."""

from __future__ import annotations

import os
from pathlib import Path

from molscout.analysis import analyze_batch
from molscout.chemistry import compute_properties as _compute_properties
from molscout.chemistry import validate_smiles as _validate_smiles
from molscout.prediction import predict_admet as _predict_admet
from molscout.prediction import explain_prediction as _explain_prediction
from molscout.similarity import similar_known_drugs as _similar_known_drugs

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised by missing-extra installs
    raise RuntimeError("Install the agent extra: pip install -e '.[agent]'") from exc

mcp = FastMCP("MolScout chemistry tools")


@mcp.tool()
def validate_smiles(smiles: str) -> dict[str, str | bool]:
    """Validate a SMILES string and return its canonical representation."""
    return _validate_smiles(smiles)


@mcp.tool()
def compute_properties(smiles: str) -> dict[str, str | float | int]:
    """Calculate reproducible 2D molecular descriptors with RDKit."""
    return _compute_properties(smiles).to_dict()


@mcp.tool()
def similar_known_drugs(smiles: str, limit: int = 3) -> dict[str, object]:
    """Find structural neighbors in MolScout's small offline demo catalog."""
    return _similar_known_drugs(smiles, limit=limit)


@mcp.tool()
def analyze_molecules(smiles_list: list[str]) -> dict[str, object]:
    """Analyze a batch and return ranked, evidence-linked heuristic reports."""
    return analyze_batch(smiles_list)


@mcp.tool()
def predict_admet(smiles: str, endpoint: str) -> dict[str, object]:
    """Predict one supported endpoint using saved Morgan/XGBoost seed models."""
    model_dir = Path(os.environ.get("MOLSCOUT_MODEL_DIR", "outputs/baseline"))
    return _predict_admet(smiles, endpoint, model_dir=model_dir)


@mcp.tool()
def explain_prediction(smiles: str, endpoint: str) -> dict[str, object]:
    """Map toxicity model fingerprint contributions back to molecule atoms."""
    model_dir = Path(os.environ.get("MOLSCOUT_MODEL_DIR", "outputs/baseline"))
    return _explain_prediction(smiles, endpoint, model_dir=model_dir)


if __name__ == "__main__":
    mcp.run()

