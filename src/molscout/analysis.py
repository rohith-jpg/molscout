"""Evidence-linked batch triage using implemented deterministic chemistry tools."""

from __future__ import annotations

from typing import Any

from molscout.chemistry import compute_properties, validate_smiles
from molscout.similarity import similar_known_drugs

DISCLAIMER = (
    "Exploratory cheminformatics only. Rule-of-five counts and fingerprint similarity "
    "are heuristics; this report contains no trained ADMET prediction or experimental result."
)


def analyze_batch(smiles_list: list[str], *, neighbors: int = 3) -> dict[str, Any]:
    """Analyze molecules and link each prose claim to a concrete tool result."""
    if not smiles_list:
        raise ValueError("Provide at least one SMILES string")
    if len(smiles_list) > 100:
        raise ValueError("A batch may contain at most 100 molecules")

    reports: list[dict[str, Any]] = []
    for input_index, smiles in enumerate(smiles_list):
        evidence: list[dict[str, Any]] = []
        validation_id = f"m{input_index + 1}-validation"
        try:
            validation = validate_smiles(smiles)
        except (TypeError, ValueError) as exc:
            evidence.append(
                {
                    "id": validation_id,
                    "tool": "validate_smiles",
                    "result": {"valid": False, "error": str(exc)},
                }
            )
            reports.append(
                {
                    "input_index": input_index,
                    "input_smiles": smiles,
                    "valid": False,
                    "rank": None,
                    "claims": [
                        {
                            "text": "The input could not be parsed as a valid SMILES string.",
                            "evidence_ids": [validation_id],
                        }
                    ],
                    "evidence": evidence,
                }
            )
            continue

        evidence.append(
            {"id": validation_id, "tool": "validate_smiles", "result": validation}
        )
        canonical = str(validation["canonical_smiles"])
        properties = compute_properties(canonical).to_dict()
        properties_id = f"m{input_index + 1}-properties"
        evidence.append(
            {
                "id": properties_id,
                "tool": "compute_properties",
                "result": properties,
            }
        )

        neighbors_result = similar_known_drugs(canonical, limit=neighbors)
        neighbors_id = f"m{input_index + 1}-similarity"
        evidence.append(
            {
                "id": neighbors_id,
                "tool": "similar_known_drugs",
                "result": neighbors_result,
            }
        )

        claims = [
            {
                "text": f"The input canonicalizes to {canonical}.",
                "evidence_ids": [validation_id],
            },
            {
                "text": (
                    "The computed descriptors have "
                    f"{properties['rule_of_five_violations']} Lipinski rule-of-five "
                    "threshold violation(s); this is a heuristic count."
                ),
                "evidence_ids": [properties_id],
            },
        ]
        top_match = neighbors_result["matches"][0]
        if top_match:
            claims.append(
                {
                    "text": (
                        "The nearest entry in the small bundled demo reference set is "
                        f"{top_match['name']} (Tanimoto {top_match['tanimoto']:.3f}); "
                        "similarity does not imply shared activity or safety."
                    ),
                    "evidence_ids": [neighbors_id],
                }
            )

        reports.append(
            {
                "input_index": input_index,
                "input_smiles": smiles,
                "canonical_smiles": canonical,
                "valid": True,
                "properties": properties,
                "claims": claims,
                "evidence": evidence,
            }
        )

    ranked = sorted(
        (report for report in reports if report["valid"]),
        key=lambda report: (
            report["properties"]["rule_of_five_violations"],
            report["input_index"],
        ),
    )
    for rank, report in enumerate(ranked, start=1):
        report["rank"] = rank
        property_evidence_id = f"m{report['input_index'] + 1}-properties"
        report["claims"].append(
            {
                "text": (
                    f"Ranked #{rank} using {report['properties']['rule_of_five_violations']} "
                    "rule-of-five threshold violation(s); ties keep input order."
                ),
                "evidence_ids": [property_evidence_id],
            }
        )

    return {
        "ranking_method": (
            "Ascending count of Lipinski rule-of-five threshold violations; ties retain input order."
        ),
        "disclaimer": DISCLAIMER,
        "molecules": reports,
    }
