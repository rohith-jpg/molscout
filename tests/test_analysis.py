from molscout.analysis import analyze_batch


def test_batch_report_links_claims_to_tool_evidence_and_keeps_invalid_inputs():
    report = analyze_batch(["CCO", "not-a-smiles"])
    ethanol, invalid = report["molecules"]

    assert ethanol["valid"] is True
    assert ethanol["rank"] == 1
    evidence_ids = {item["id"] for item in ethanol["evidence"]}
    assert evidence_ids == {"m1-validation", "m1-properties", "m1-similarity"}
    assert all(set(claim["evidence_ids"]) <= evidence_ids for claim in ethanol["claims"])
    assert invalid["valid"] is False
    assert invalid["rank"] is None
    assert invalid["claims"][0]["evidence_ids"] == ["m2-validation"]


def test_batch_ranks_only_by_explicit_rule_of_five_heuristic():
    report = analyze_batch(["CCCCCCCCCCCCCCCCCCCC", "CCO"])
    ranked = sorted(
        (item for item in report["molecules"] if item["valid"]),
        key=lambda item: item["rank"],
    )
    assert ranked[0]["canonical_smiles"] == "CCO"
    assert "Lipinski rule-of-five" in report["ranking_method"]
    assert "no trained ADMET prediction" in report["disclaimer"]


def test_batch_rejects_empty_or_oversized_input():
    import pytest

    with pytest.raises(ValueError):
        analyze_batch([])
    with pytest.raises(ValueError):
        analyze_batch(["CCO"] * 101)
