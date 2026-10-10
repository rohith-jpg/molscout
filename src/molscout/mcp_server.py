"""MCP stdio server exposing implemented chemistry tools."""

from __future__ import annotations

from molscout.chemistry import compute_properties, validate_smiles

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised by missing-extra installs
    raise RuntimeError("Install the agent extra: pip install -e '.[agent]'") from exc

mcp = FastMCP("MolScout chemistry tools")


@mcp.tool()
def validate_smiles_tool(smiles: str) -> dict[str, str | bool]:
    """Validate a SMILES string and return its canonical representation."""
    return validate_smiles(smiles)


@mcp.tool()
def compute_properties_tool(smiles: str) -> dict[str, str | float | int]:
    """Calculate reproducible 2D molecular descriptors with RDKit."""
    return compute_properties(smiles).to_dict()


if __name__ == "__main__":
    mcp.run()
