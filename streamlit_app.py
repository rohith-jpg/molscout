"""Streamlit Community Cloud entrypoint for MolScout's offline demo."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molscout.analysis import analyze_batch  # noqa: E402

st.set_page_config(page_title="MolScout | Molecule triage", page_icon="🧪", layout="wide")
st.title("🧪 MolScout")
st.subheader("Evidence-backed molecule triage")
st.write(
    "Paste SMILES to check whether each structure parses, review calculated 2D "
    "descriptors, and find its closest match in a small offline reference set."
)
st.info(
    "Demo scope: this version uses deterministic RDKit descriptors and a "
    "five-compound teaching catalog. It does **not** run trained ADMET models "
    "or establish whether a molecule is safe, effective, or a drug candidate."
)

with st.sidebar:
    st.header("What this demo does")
    st.markdown(
        "1. Parses each SMILES with RDKit.\n"
        "2. Calculates molecular weight, cLogP, H-bond counts, TPSA, and rotatable bonds.\n"
        "3. Counts Lipinski threshold violations as a simple heuristic.\n"
        "4. Finds nearest entries using Morgan fingerprints and Tanimoto similarity.\n"
        "5. Links report claims to their supporting tool results."
    )
    st.caption("No network lookups or model API keys are used in this demo.")

example = "CCO\nCC(=O)O\nCn1c(=O)c2c(ncn2C)n(C)c1=O\nnot-a-smiles"
demo_path = ROOT / "data" / "demo_run.json"
if st.query_params.get("demo") == "1" and "molscout_result" not in st.session_state:
    st.session_state["molscout_result"] = json.loads(demo_path.read_text(encoding="utf-8"))

if st.button("Replay saved no-cost example"):
    st.session_state["molscout_result"] = json.loads(demo_path.read_text(encoding="utf-8"))

smiles_text = st.text_area(
    "One molecule per line (SMILES)",
    value=example,
    height=160,
    help="Maximum 100 inputs per analysis. Invalid rows are shown instead of discarded.",
)

if st.button("Analyze molecules", type="primary", width="stretch"):
    smiles_list = [line.strip() for line in smiles_text.splitlines() if line.strip()]
    if not smiles_list:
        st.error("Enter at least one SMILES string.")
    elif len(smiles_list) > 100:
        st.error("This demo accepts up to 100 molecules per batch.")
    else:
        try:
            started = time.perf_counter()
            st.session_state["molscout_result"] = analyze_batch(smiles_list)
            st.session_state["molscout_elapsed"] = time.perf_counter() - started
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))


def render_molecule(report: dict[str, Any]) -> None:
    rank = report.get("rank")
    structure = report.get("canonical_smiles", report["input_smiles"])
    heading = f"#{rank} · {structure}" if rank is not None else f"Invalid · {structure}"
    with st.expander(heading, expanded=rank == 1):
        if not report["valid"]:
            st.error("This input could not be parsed as a valid SMILES.")
        else:
            props = report["properties"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Molecular weight", f"{props['molecular_weight']:.1f}")
            metric_cols[1].metric("cLogP", f"{props['clogp']:.2f}")
            metric_cols[2].metric(
                "H-bond donors / acceptors",
                f"{props['h_bond_donors']} / {props['h_bond_acceptors']}",
            )
            metric_cols[3].metric(
                "Rule-of-five violations", str(props["rule_of_five_violations"])
            )
            st.caption(
                f"TPSA: {props['tpsa']:.1f} Å² · Rotatable bonds: "
                f"{props['rotatable_bonds']} · Rank uses violation count only."
            )
            matches = report["evidence"][-1]["result"]["matches"]
            st.markdown("**Closest entries in the demo catalog**")
            if matches:
                st.dataframe(
                    [
                        {"Name": item["name"], "Tanimoto similarity": item["tanimoto"]}
                        for item in matches
                    ],
                    hide_index=True,
                    width="stretch",
                )
        st.markdown("**Claims and evidence IDs**")
        st.dataframe(
            [
                {"Claim": claim["text"], "Evidence": ", ".join(claim["evidence_ids"])}
                for claim in report["claims"]
            ],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Raw tool results"):
            st.code(json.dumps(report["evidence"], indent=2), language="json")


result = st.session_state.get("molscout_result")
if result:
    molecules = result["molecules"]
    valid_count = sum(molecule["valid"] for molecule in molecules)
    invalid_count = len(molecules) - valid_count
    summary = st.columns(3)
    summary[0].metric("Inputs", len(molecules))
    summary[1].metric("Parsed", valid_count)
    summary[2].metric("Invalid", invalid_count)
    elapsed = st.session_state.get("molscout_elapsed")
    if elapsed is not None:
        st.caption(f"Analysis completed in {elapsed:.3f} seconds on this app instance.")
    st.caption(f"Ranking method: {result['ranking_method']}")
    for molecule in molecules:
        render_molecule(molecule)
    st.caption(result["disclaimer"])

benchmark_path = ROOT / "docs" / "benchmark" / "baseline_results.json"
if benchmark_path.exists():
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    st.subheader("Offline model benchmark")
    st.caption(
        "These are measured five-seed Morgan + XGBoost results on TDC scaffold splits. "
        "The live molecule demo above does not use these trained models."
    )
    leaderboard = benchmark["leaderboard_reference"]
    rows = []
    for result_row in benchmark["aggregate"]:
        reference = leaderboard[result_row["dataset"]]
        rows.append(
            {
                "Dataset": result_row["dataset"],
                "Metric": result_row["metric"],
                "MolScout mean ± SD": (
                    f"{result_row['mean']:.3f} ± {result_row['std']:.3f}"
                ),
                "TDC leaderboard reference": (
                    f"{reference['best_model']}: "
                    f"{reference['mean']:.3f} ± {reference['std']:.3f}"
                ),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "TDC values are published reference results, not reproduced runs. "
        "Two unparseable AqSolDB rows were excluded from MolScout's MAE; see the "
        "benchmark report in the GitHub repository for details."
    )

