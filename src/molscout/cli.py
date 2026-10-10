"""Command-line helpers for early MolScout development."""

from __future__ import annotations

import json

import typer

from molscout.chemistry import compute_properties, validate_smiles

app = typer.Typer(no_args_is_help=True, help="Molecular property triage utilities")


@app.command()
def validate(smiles: str) -> None:
    """Validate one SMILES string and print basic molecular properties."""
    result = {**validate_smiles(smiles), "properties": compute_properties(smiles).to_dict()}
    typer.echo(json.dumps(result, indent=2))
