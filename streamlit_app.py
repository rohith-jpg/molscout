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
from molscout.prediction import ENDPOINTS, explain_prediction, predict_admet  # noqa: E402
from molscout.visualization import molecule_attribution_image  # noqa: E402

st.set_page_config(page_title="MolScout | Molecule triage", page_icon="🧪", layout="wide")
st.title("🧪 MolScout")
st.subheader("Evidence-backed molecule triage")
st.write(
    "Paste SMILES to check whether each structure parses, review calculated 2D "
    "descriptors, inspect four trained ADMET baseline scores, and find its closest "
    "match in a small offline reference set."
)
st.info(
    "Demo scope: ADMET scores come from five-seed Morgan + XGBoost models trained "
    "on TDC scaffold splits. Seed-to-seed spread is not calibrated uncertainty. "
    "These retrospective benchmark models do not establish whether a molecule is "
    "safe, effective, or a drug candidate."
)

with st.sidebar:
    st.header("What this demo does")
    st.markdown(
        "1. Parses each SMILES with RDKit.\n"
        "2. Calculates molecular weight, cLogP, H-bond counts, TPSA, and rotatable bonds.\n"
        "3. Counts Lipinski threshold violations as a simple heuristic.\n"
        "4. Finds nearest entries using Morgan fingerprints and Tanimoto similarity.\n"
        "5. Scores four ADMET endpoints with five trained baseline seeds each.\n"
        "6. Links report claims to their supporting tool results."
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
            result = analyze_batch(smiles_list)
            for molecule in result["molecules"]:
                if not molecule["valid"]:
                    continue
                predictions = []
                for endpoint in ENDPOINTS:
                    prediction = predict_admet(
                        molecule["canonical_smiles"], endpoint
                    )
                    evidence_id = (
                        f"m{molecule['input_index'] + 1}-admet-{endpoint}"
                    )
                    molecule["evidence"].append(
                        {
                            "id": evidence_id,
                            "tool": "predict_admet",
                            "result": prediction,
                        }
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
                    molecule["claims"].append(
                        {
                            "text": (
                                f"The {endpoint} baseline output is "
                                f"{prediction['prediction']:.4f} ({prediction['unit']}); "
                                "the five-seed spread is descriptive, not calibrated "
                                "uncertainty."
                            ),
                            "evidence_ids": [evidence_id],
                        }
                    )
                molecule["admet_predictions"] = predictions
            st.session_state["molscout_result"] = result
            st.session_state["molscout_elapsed"] = time.perf_counter() - started
            st.session_state["molscout_explanations"] = {}
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
            predictions = report.get("admet_predictions", [])
            if predictions:
                st.markdown("**Five-seed ADMET baseline outputs**")
                st.dataframe(
                    [
                        {
                            "Endpoint": item["endpoint"],
                            (
                                "P(class 1)" if item["kind"] == "classification"
                                else "Predicted log10(mol/L)"
                            ): f"{item['prediction']:.4f}",
                            "Seed spread (SD)": f"{item['ensemble_std']:.4f}",
                            "Evidence": item["evidence_id"],
                        }
                        for item in predictions
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Classification values are model scores for dataset class 1. "
                    "Seed spread is not a probability interval or calibrated confidence. "
                    "Scores do not determine the heuristic rank."
                )
                for endpoint in ("hERG", "AMES"):
                    if not any(item["endpoint"] == endpoint for item in predictions):
                        continue
                    evidence_id = f"m{report['input_index'] + 1}-explain-{endpoint}"
                    explanation_key = f"{report['input_index']}:{endpoint}"
                    if st.button(
                        f"Explain {endpoint} score with atom attribution",
                        key=f"explain-{explanation_key}",
                    ):
                        try:
                            explanation = explain_prediction(
                                report["canonical_smiles"], endpoint
                            )
                            report["evidence"] = [
                                item for item in report["evidence"]
                                if item["id"] != evidence_id
                            ]
                            report["evidence"].append(
                                {
                                    "id": evidence_id,
                                    "tool": "explain_prediction",
                                    "result": explanation,
                                }
                            )
                            report["claims"] = [
                                claim for claim in report["claims"]
                                if evidence_id not in claim["evidence_ids"]
                            ]
                            top_atoms = explanation["most_positive_atoms"][:3]
                            atom_summary = (
                                ", ".join(
                                    f"{atom['element']} at atom {atom['atom_index']}"
                                    for atom in top_atoms
                                )
                                if top_atoms
                                else "no positive mapped atom contribution"
                            )
                            report["claims"].append(
                                {
                                    "text": (
                                        f"The strongest positive {endpoint} fingerprint "
                                        f"attribution maps to {atom_summary}; this is an "
                                        "association learned by the model, not a causal "
                                        "toxicity mechanism."
                                    ),
                                    "evidence_ids": [evidence_id],
                                }
                            )
                            st.session_state.setdefault("molscout_explanations", {})[
                                explanation_key
                            ] = explanation
                        except (ValueError, FileNotFoundError) as exc:
                            st.error(str(exc))

                    explanation = st.session_state.get("molscout_explanations", {}).get(
                        explanation_key
                    )
                    if explanation:
                        st.image(
                            molecule_attribution_image(
                                explanation["canonical_smiles"],
                                explanation["atom_attributions"],
                                size=(560, 360),
                                legend=f"{endpoint} model atom attribution",
                            ),
                            caption=(
                                f"{endpoint} XGBoost contribution by atom: red pushes "
                                "toward dataset class 1; blue pushes away."
                            ),
                        )
                        st.dataframe(
                            [
                                {
                                    "Atom": atom["atom_index"],
                                    "Element": atom["element"],
                                    "Mean raw-margin contribution": f"{atom['attribution']:+.4f}",
                                }
                                for atom in explanation["atom_attributions"][:10]
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(explanation["warning"])
                        st.caption(
                            "The picture projects hashed fingerprint-feature effects onto "
                            f"atom environments. Unmapped nonzero feature contribution: "
                            f"{explanation['unmapped_feature_contribution']:+.4f} raw-margin units."
                        )
            similarity_result = next(
                item["result"]
                for item in report["evidence"]
                if item["tool"] == "similar_known_drugs"
            )
            matches = similarity_result["matches"]
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
        "The same saved baseline models power the live molecule scores above."
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

conformal_path = ROOT / "docs" / "benchmark" / "conformal_results.json"
if conformal_path.exists():
    conformal = json.loads(conformal_path.read_text(encoding="utf-8"))
    st.subheader("Offline conformal interval evaluation")
    st.caption(
        "Five 90% split-conformal runs on TDC AqSolDB. This reports measured coverage "
        "on the scaffold-held-out test set; the live molecule scores do not yet include "
        "calibrated intervals."
    )
    aggregate = conformal["aggregate"]
    st.dataframe(
        [
            {
                "Metric": "Test MAE (log10 mol/L)",
                "Mean ± population SD": (
                    f"{aggregate['test_mae']['mean']:.3f} ± "
                    f"{aggregate['test_mae']['population_sd']:.3f}"
                ),
            },
            {
                "Metric": "Empirical test coverage (nominal 90%)",
                "Mean ± population SD": (
                    f"{100 * aggregate['test_empirical_coverage']['mean']:.1f}% ± "
                    f"{100 * aggregate['test_empirical_coverage']['population_sd']:.1f}%"
                ),
            },
            {
                "Metric": "Mean interval width (log10 mol/L)",
                "Mean ± population SD": (
                    f"{aggregate['test_mean_interval_width']['mean']:.3f} ± "
                    f"{aggregate['test_mean_interval_width']['population_sd']:.3f}"
                ),
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Two invalid AqSolDB test rows were excluded per seed. Coverage is empirical; "
        "scaffold shift can break the exchangeability assumption. See "
        "docs/benchmark/conformal_results.json for seed-level details."
    )

chemprop_path = ROOT / "docs" / "benchmark" / "chemprop_hERG_pilot.json"
if chemprop_path.exists():
    chemprop_pilot = json.loads(chemprop_path.read_text(encoding="utf-8"))
    st.subheader("Offline Chemprop hERG pilot")
    st.caption(
        "One seed, five CPU-trained models, and 132 official scaffold-held-out test "
        "molecules. This is a pilot result, not evidence of general model superiority."
    )
    st.dataframe(
        [
            {"Model": "Chemprop MPNN ensemble (5 models)", "AUROC": f"{chemprop_pilot['chemprop_auroc']:.3f}"},
            {"Model": "Morgan + XGBoost (same seed and test rows)", "AUROC": f"{chemprop_pilot['same_seed_morgan_xgboost_auroc']:.3f}"},
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Seed {chemprop_pilot['seed']}; n={chemprop_pilot['n_test']} test rows; "
        "see docs/benchmark/chemprop_hERG_pilot.json for exact values and versions."
    )




