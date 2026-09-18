import pytest

import scripts.analyze_proof_failures as apf


def row(*, proof=True, answer=False, executed="(: $prf (Wrong thing) $tv)"):
    raw_proof = "['(: proof (Wrong thing) (STV 1.0 1.0))']" if proof else "[]"
    return {
        "case": {"case_id": "X01", "name": "case", "question": "What is right?"},
        "proof_found": proof,
        "answer_correct": answer,
        "answer_reason": "0/1 slots [any]: none",
        "matched_entities": [],
        "end_to_end": {"query": {
            "original_query": "(: $prf (Right thing) $tv)",
            "executed_query": executed,
            "query_status": "weakly_aligned",
            "fallback_used": True,
            "proof": raw_proof,
            "senf": {"candidate_score_breakdown": [
                {"query": executed, "total": 12, "senf_components": {"transport": -1}}
            ]},
        }},
    }


GOLD = {"entities": [["right"]], "verdict": None}


def test_classifies_missing_answer_evidence_without_calling_query_off_target():
    result = apf.classify_failure(row(), GOLD)

    assert result["primary_failure"] == "proof_answer_evidence_miss"
    assert "executed_query_answer_symbol_miss" in result["evidence_flags"]
    assert result["senf"]["candidate_rank"] == 1


def test_valid_query_need_not_name_the_answer():
    result = apf.classify_failure(
        row(executed="(: $prf (Produces process $answer) $tv)"),
        {"entities": [["product"]], "verdict": None},
    )

    assert result["primary_failure"] == "proof_answer_evidence_miss"
    assert not result["primary_failure"].startswith("off_target_query")


def test_malformed_query_takes_priority():
    result = apf.classify_failure(
        row(executed="(: $prf (Right thing}) $tv)"), GOLD
    )

    assert result["primary_failure"] == "malformed_executed_query"
    assert "malformed_executed_query" in result["evidence_flags"]


def test_zero_strength_proof_does_not_supply_answer_evidence():
    result_row = row(executed="(: $prf (Right thing) $tv)")
    result_row["end_to_end"]["query"]["proof"] = (
        "['(: proof (Right thing) (STV 0.0 1.0))']"
    )

    result = apf.classify_failure(result_row, GOLD)

    assert result["primary_failure"] == "proof_answer_evidence_miss"
    assert "nonpositive_proof" in result["evidence_flags"]


def test_expected_non_proof_is_classified_as_unexpected_proof():
    result = apf.classify_failure(
        row(), {"entities": [], "verdict": None, "expected_proof": False}
    )

    assert result["primary_failure"] == "unexpected_proof"
    assert "proof_answer_evidence_miss" not in result["evidence_flags"]


def test_analysis_defaults_to_proof_bearing_failures():
    report = {
        "suite": "fixture",
        "run_id": "run",
        "parsers": {"senf": [row(), row(proof=False)]},
    }
    gold = {"cases": {"X01": GOLD}}

    result = apf.analyze_report(report, gold, "senf")

    assert result["summary"]["analyzed_failures"] == 1
    assert result["summary"]["proof_bearing_incorrect"] == 1
    assert result["summary"]["no_proof_incorrect"] == 1


def test_analysis_can_include_no_proof_failures():
    report = {
        "suite": "fixture",
        "run_id": "run",
        "parsers": {"senf": [row(), row(proof=False)]},
    }

    result = apf.analyze_report(
        report, {"cases": {"X01": GOLD}}, "senf", include_no_proof=True
    )

    assert result["summary"]["analyzed_failures"] == 2
    assert result["failures"][1]["primary_failure"] == "no_proof"


def test_markdown_identifies_scope_and_source_hash():
    analysis = apf.analyze_report(
        {"suite": "fixture", "run_id": "run", "parsers": {"senf": [row()]}},
        {"cases": {"X01": GOLD}},
        "senf",
    )

    markdown = apf.render_markdown(analysis, "abc123")

    assert "deterministic symptom taxonomy" in markdown
    assert "Source report SHA-256: `abc123`" in markdown
    assert "| X01 | proof_answer_evidence_miss |" in markdown


def test_rejects_malformed_gold_entity_slots():
    with pytest.raises(ValueError, match="malformed entity slots"):
        apf.classify_failure(row(), {"entities": ["right"]})
