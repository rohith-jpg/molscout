"""Command-line helpers for early MolScout development."""

from __future__ import annotations

import json

import typer

from molscout.analysis import analyze_batch
from molscout.chemistry import compute_properties, validate_smiles

app = typer.Typer(no_args_is_help=True, help="Molecular property triage utilities")


@app.command()
def validate(smiles: str) -> None:
    """Validate one SMILES string and print basic molecular properties."""
    result = {**validate_smiles(smiles), "properties": compute_properties(smiles).to_dict()}
    typer.echo(json.dumps(result, indent=2))


@app.command()
def analyze(smiles: list[str] = typer.Argument(..., help="One or more SMILES strings")) -> None:
    """Rank a small batch by a transparent heuristic and print cited evidence."""
    try:
        report = analyze_batch(smiles)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, indent=2))
