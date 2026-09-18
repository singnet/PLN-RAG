import copy
import itertools

import pytest

import scripts.compare_query_policies as cqp


def result(case_id, query, *, correct=False, proof="[]", tried=1):
    return {
        "case": {"case_id": case_id, "question": "What is the answer?", "input_text": "Text."},
        "parse_only": {"queries": ["q"]},
        "end_to_end": {
            "ingest": [{"status": "success", "atoms": ["atom"], "rejected_count": 0}],
            "query": {
                "original_query": "original",
                "executed_query": query,
                "query_status": "well_aligned",
                "proof": proof,
                "candidate_count": 3,
                "candidate_count_tried": tried,
                "attempted_candidate_indices": [0],
                "successful_candidate_index": 0 if proof != "[]" else None,
            },
        },
        "proof_found": proof != "[]",
        "answer_correct": correct,
        "answer_score": 1.0 if correct else 0.0,
        "answer_reason": "fixture",
    }


def report():
    payload = {
        "suite": "fixture",
        "run_id": "run",
        "valid": True,
        "validity_warnings": [],
        "mode": "isolated",
        "generation_tape": {"mode": "replay", "sha256": "tape", "parser_runs": []},
        "parsers": {},
        "arm_metadata": {},
    }
    settings = cqp.EXPECTED_POLICY_SETTINGS
    for policy, arm in cqp.POLICY_ARMS.items():
        query = "original" if policy == "original_only" else "ranked"
        payload["parsers"][arm] = [
            result("X01", query, correct=policy == "original_only", proof="[]")
        ]
        payload["parsers"][arm][0]["end_to_end"]["query"]["execution_policy"] = policy
        payload["arm_metadata"][arm] = {
            "parser": "canonical_senf_pln",
            "senf_counterfactual_enabled": True,
            "query_execution_policy": policy,
            "settings": settings,
        }
        payload["generation_tape"]["parser_runs"].append({
            "parser": arm,
            "case_id": "X01",
            "calls": 1,
            "context_mismatches": 0,
            "repeated_calls": 0,
            "unconsumed_calls": {},
        })
    return payload


def test_compares_paired_policy_results():
    comparison = cqp.compare_report(report(), expected_tape_sha256="tape")

    assert comparison["changed_case_count"] == 1
    assert comparison["summary"]["original_only"]["automatic_correct"] == 1
    assert comparison["automatic_discordance"] == {
        "original_only_automatic_correct": 1
    }


def test_rejects_ingest_drift():
    payload = report()
    arm = cqp.POLICY_ARMS["original_only"]
    payload["parsers"][arm][0]["end_to_end"]["ingest"][0]["atoms"] = ["different"]

    with pytest.raises(ValueError, match="Ingest output differs"):
        cqp.compare_report(payload)


def test_rejects_setting_drift():
    payload = report()
    arm = cqp.POLICY_ARMS["original_only"]
    payload["arm_metadata"][arm]["settings"] = {"answer_generation_enabled": "true"}

    with pytest.raises(ValueError, match="frozen policy settings"):
        cqp.compare_report(payload)


def test_frozen_settings_include_planner_modalities_and_diagnostics():
    assert {
        "senf_weave_forget_fine_costs",
        "senf_weave_coarse_identity",
        "senf_weave_max_conflict_cost",
        "senf_weave_max_transport_cost",
        "senf_weave_require_context_match",
        "senf_matched_soft_mass_weight",
        "senf_global_residual_ratio_weight",
        "senf_alignment_confidence_weight",
        "senf_diagnostics_max_weaves",
        "senf_diagnostics_max_items",
        "senf_diagnostics_max_evidence",
    } <= set(cqp.EXPECTED_POLICY_SETTINGS)


def test_rejects_systematic_setting_drift():
    payload = report()
    for arm in cqp.POLICY_ARMS.values():
        payload["arm_metadata"][arm]["settings"] = {
            **cqp.EXPECTED_POLICY_SETTINGS,
            "answer_generation_enabled": "true",
        }

    with pytest.raises(ValueError, match="frozen policy settings"):
        cqp.compare_report(payload)


def test_adjudication_packet_is_blinded_and_only_includes_changes():
    payload = report()
    comparison = cqp.compare_report(payload)

    packet, key = cqp.adjudication_packet(comparison, payload, "source")

    assert len(packet["cases"]) == 1
    assert {item["label"] for item in packet["cases"][0]["outputs"]} == {"A", "B", "C"}
    assert "policy" not in packet["cases"][0]["outputs"][0]
    assert set(key["cases"]["X01"].values()) == set(cqp.POLICY_ARMS)


def test_blinding_uses_all_policy_permutations():
    orders = {
        cqp._blinded_policy_order("source", f"X{index:03d}")
        for index in range(100)
    }

    assert orders == set(itertools.permutations(cqp.POLICY_ARMS))


def test_rejects_case_set_drift():
    payload = copy.deepcopy(report())
    arm = cqp.POLICY_ARMS["ranked_first_only"]
    payload["parsers"][arm] = []

    with pytest.raises(ValueError, match="different case sets"):
        cqp.compare_report(payload)


def test_rejects_replay_drift():
    payload = report()
    payload["generation_tape"]["parser_runs"][0]["context_mismatches"] = 1

    with pytest.raises(ValueError, match="replay diagnostics report drift"):
        cqp.compare_report(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("valid", False, "not marked valid"),
        ("mode", "shared", "requires isolated mode"),
    ],
)
def test_rejects_invalid_report_modes(field, value, message):
    payload = report()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        cqp.compare_report(payload)


def test_rejects_capture_tape_and_incomplete_replay_coverage():
    payload = report()
    payload["generation_tape"]["mode"] = "capture"
    with pytest.raises(ValueError, match="requires generation replay"):
        cqp.compare_report(payload)

    payload = report()
    payload["generation_tape"]["parser_runs"].pop()
    with pytest.raises(ValueError, match="do not cover every arm/case"):
        cqp.compare_report(payload)


def test_rejects_policy_telemetry_errors_and_ingest_failure():
    arm = cqp.POLICY_ARMS["original_only"]

    payload = report()
    payload["parsers"][arm][0]["end_to_end"]["query"]["execution_policy"] = "ranked_first_only"
    with pytest.raises(ValueError, match="telemetry mismatch"):
        cqp.compare_report(payload)

    payload = report()
    payload["parsers"][arm][0]["error"] = "failed"
    with pytest.raises(ValueError, match="Invalid or ungraded result"):
        cqp.compare_report(payload)

    payload = report()
    for policy_arm in cqp.POLICY_ARMS.values():
        payload["parsers"][policy_arm][0]["end_to_end"]["ingest"][0]["status"] = "error"
    with pytest.raises(ValueError, match="Ingestion failed"):
        cqp.compare_report(payload)


def test_manual_labels_report_usable_accuracy_and_grader_confusion():
    comparison = cqp.compare_report(report())
    comparison["source_report_sha256"] = "source"
    comparison["adjudication_packet_sha256"] = "packet"
    labels = {
        "source_report_sha256": "source",
        "adjudication_packet_sha256": "packet",
        "status": "fixture",
        "cases": {"X01": {
            "ranked_first_proof": "incorrect",
            "ranked_first_only": "qualified",
            "original_only": "adequate",
        }},
    }

    result = cqp.apply_manual_labels(comparison, labels)

    assert result["summary"]["original_only"]["strict_correct"] == 1
    assert result["summary"]["ranked_first_only"]["usable_correct"] == 1
    assert result["summary"]["original_only"]["automatic_confusion"][
        "automatic_true_manual_usable"
    ] == 1


def test_manual_agreement_reports_exact_and_binary_counts():
    first = {"source_report_sha256": "source", "adjudication_packet_sha256": "packet", "cases": {"X01": {
        "ranked_first_proof": "adequate",
        "ranked_first_only": "qualified",
        "original_only": "incorrect",
    }}}
    second = {"source_report_sha256": "source", "adjudication_packet_sha256": "packet", "cases": {"X01": {
        "ranked_first_proof": "qualified",
        "ranked_first_only": "qualified",
        "original_only": "insufficient",
    }}}

    result = cqp.manual_agreement([first, second])

    assert result["ranked_first_proof"]["exact_label_agreement"] == 0
    assert result["ranked_first_proof"]["usable_binary_agreement"] == 1
    assert result["original_only"]["usable_binary_agreement"] == 1
