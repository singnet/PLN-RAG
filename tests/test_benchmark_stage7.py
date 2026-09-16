import json
from pathlib import Path

import pytest

import benchmark_stage7 as bs7


SUITE = Path("data/benchmarks/stage7_counterfactual_v1.json")


def test_stage7_suite_has_positive_and_negative_cases():
    payload = bs7.load_suite(SUITE)
    categories = {case["category"] for case in payload["cases"]}

    assert categories == {"authorized_positive", "safety_negative"}
    assert len(payload["cases"]) >= 10


def test_stage7_suite_case_ids_are_unique(tmp_path):
    payload = json.loads(SUITE.read_text(encoding="utf-8"))
    payload["cases"].append(dict(payload["cases"][0]))
    duplicate = tmp_path / "stage7_duplicate_suite.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        bs7.load_suite(duplicate)


def test_fixture_backend_returns_independent_results():
    case = bs7.load_suite(SUITE)["cases"][0]
    backend = bs7.FixtureGenerationBackend(case)

    statement = backend.generate(
        sentences=[case["input_text"]], context=[], pln_spec=None, live=None
    )
    query = backend.generate(
        sentences=[case["user_query"]], context=[], pln_spec=None, live=None
    )

    assert statement.statements == case["statement_atoms"]
    assert query.queries == case["query_atoms"]
    assert backend.calls == ["statement", "query"]


def test_summary_separates_capability_and_safety():
    summary = bs7.summarize([
        {
            "category": "authorized_positive",
            "correct": True,
            "proof_found": True,
            "execution_valid": True,
            "elapsed_seconds": 1.0,
            "semantic_checks": [{"passed": True}],
        },
        {
            "category": "safety_negative",
            "correct": False,
            "proof_found": True,
            "execution_valid": True,
            "elapsed_seconds": 3.0,
            "semantic_checks": [{"passed": False}],
        },
    ])

    assert summary["authorized_positive_correct"] == 1
    assert summary["safety_false_positives"] == 1
    assert summary["semantic_checks_passed"] == 1
    assert summary["average_latency_seconds"] == 2.0


def test_stage7_checks_use_the_executed_plan():
    case = {
        "stage7_expectation": {
            "branch_id": "rain",
            "validity_interval_id": "near",
            "branch_probability": 0.4,
        }
    }
    result = {
        "senf": {
            "executed_temporal_plan": {
                "branch_id": "dry",
                "validity_interval_id": "near",
                "decisions": [{"allowed": True, "branch_probability": 0.4}],
            },
            "candidate_temporal_plans": [{
                "branch_id": "rain",
                "validity_interval_id": "near",
                "decisions": [{"allowed": True, "branch_probability": 0.4}],
            }],
        }
    }

    checks = {item["name"]: item for item in bs7._stage7_checks(case, result)}

    assert checks["branch_id"]["passed"] is False
    assert checks["validity_interval_id"]["passed"] is True
    assert checks["branch_probability"]["passed"] is True


def test_suite_rejects_string_proof_expectations(tmp_path):
    payload = json.loads(SUITE.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_proof"] = "false"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be boolean"):
        bs7.load_suite(invalid)


def test_suite_requires_case_name(tmp_path):
    payload = json.loads(SUITE.read_text(encoding="utf-8"))
    del payload["cases"][0]["name"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields: name"):
        bs7.load_suite(invalid)
