"""LangGraph orchestration for parallel, evidence-linked molecule triage.

This is a deterministic tool workflow, not an LLM planner. It provides a
reproducible plan, parallel per-molecule execution, heuristic ranking, and
traceable evidence for each generated claim.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from molscout.analysis import analyze_batch
from molscout.prediction import ENDPOINTS, predict_admet


class AgentState(TypedDict, total=False):
    smiles: list[str]
    endpoints: list[str]
    model_dir: str
    parallelism: int
    plan: dict[str, Any]
    molecules: list[dict[str, Any]]
    ranking_method: str


def _analyze_one(
    item: tuple[int, str],
    endpoints: list[str],
    model_dir: Path,
) -> dict[str, Any]:
    index, smiles = item
    report = analyze_batch([smiles])["molecules"][0]
    report["input_index"] = index
    report["evidence"] = [
        {
            **entry,
            "id": f"m{index + 1}-{entry['id'].split('-', 1)[1]}",
        }
        for entry in report["evidence"]
    ]
    for claim in report["claims"]:
        claim["evidence_ids"] = [
            f"m{index + 1}-" + evidence_id.split("-", 1)[1]
            for evidence_id in claim["evidence_ids"]
        ]

    if not report["valid"]:
        return report

    predictions: list[dict[str, Any]] = []
    for endpoint in endpoints:
        evidence_id = f"m{index + 1}-admet-{endpoint}"
        try:
            prediction = predict_admet(
                report["canonical_smiles"], endpoint, model_dir=model_dir
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            prediction = {"error": str(exc), "endpoint": endpoint}
            report["evidence"].append(
                {"id": evidence_id, "tool": "predict_admet", "result": prediction}
            )
            continue
        report["evidence"].append(
            {"id": evidence_id, "tool": "predict_admet", "result": prediction}
        )
        predictions.append(
            {
                "endpoint": endpoint,
                "prediction": prediction["prediction"],
                "ensemble_std": prediction["ensemble_std"],
                "kind": prediction["kind"],
                "unit": prediction["unit"],
                "evidence_id": evidence_id,
            }
        )
        report["claims"].append(
            {
                "text": (
                    f"The {endpoint} baseline output is "
                    f"{prediction['prediction']:.4f} ({prediction['unit']}); "
                    "ensemble spread is descriptive, not calibrated uncertainty."
                ),
                "evidence_ids": [evidence_id],
            }
        )
    report["admet_predictions"] = predictions
    return report


def _plan(state: AgentState) -> dict[str, Any]:
    smiles = state["smiles"]
    endpoints = state.get("endpoints", list(ENDPOINTS))
    unknown = sorted(set(endpoints) - set(ENDPOINTS))
    if unknown:
        raise ValueError(f"Unknown endpoints: {unknown}")
    if not smiles:
        raise ValueError("Provide at least one SMILES string")
    if len(smiles) > 100:
        raise ValueError("A batch may contain at most 100 molecules")
    return {
        "endpoints": endpoints,
        "molecules": len(smiles),
        "tools_per_valid_molecule": [
            "validate_smiles",
            "compute_properties",
            "similar_known_drugs",
            *[f"predict_admet:{endpoint}" for endpoint in endpoints],
        ],
        "parallel_strategy": "independent molecule workflows run in a bounded thread pool",
        "planner": "deterministic; no LLM is called",
    }


def _execute(state: AgentState) -> dict[str, Any]:
    items = list(enumerate(state["smiles"]))
    workers = max(1, min(state.get("parallelism", 4), len(items), 8))
    model_dir = Path(state.get("model_dir", "outputs/baseline"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reports = list(
            pool.map(
                lambda item: _analyze_one(item, state["endpoints"], model_dir),
                items,
            )
        )
    return {"molecules": reports}


def _rank_and_cite(state: AgentState) -> dict[str, Any]:
    reports = state["molecules"]
    for report in reports:
        report["claims"] = [
            claim for claim in report["claims"]
            if not claim["text"].startswith("Ranked #")
        ]
    ranked = sorted(
        (report for report in reports if report["valid"]),
        key=lambda report: (
            report["properties"]["rule_of_five_violations"],
            report["input_index"],
        ),
    )
    for rank, report in enumerate(ranked, start=1):
        report["rank"] = rank
        evidence_id = f"m{report['input_index'] + 1}-properties"
        report["claims"].append(
            {
                "text": (
                    f"Ranked #{rank} by ascending Lipinski threshold violation count; "
                    "ties retain input order. This is a transparent heuristic."
                ),
                "evidence_ids": [evidence_id],
            }
        )
    return {
        "molecules": reports,
        "ranking_method": (
            "Ascending Lipinski rule-of-five threshold violation count; "
            "ties retain input order. ADMET predictions do not affect rank."
        ),
    }


def run_agent_workflow(
    smiles: list[str],
    *,
    endpoints: list[str] | None = None,
    model_dir: Path = Path("outputs/baseline"),
    parallelism: int = 4,
) -> dict[str, Any]:
    """Run a planned, parallel chemistry-tool workflow and return cited results."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan)
    graph.add_node("execute_tools", _execute)
    graph.add_node("rank_and_cite", _rank_and_cite)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_tools")
    graph.add_edge("execute_tools", "rank_and_cite")
    graph.add_edge("rank_and_cite", END)
    result = graph.compile().invoke(
        {
            "smiles": smiles,
            "endpoints": endpoints or list(ENDPOINTS),
            "model_dir": str(model_dir),
            "parallelism": parallelism,
        }
    )
    return {
        "plan": result["plan"],
        "ranking_method": result["ranking_method"],
        "molecules": result["molecules"],
        "disclaimer": (
            "Research and education aid only. Model outputs and heuristic ranking "
            "are not experimental findings or medical advice."
        ),
    }
