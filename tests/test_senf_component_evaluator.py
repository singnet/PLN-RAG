import copy
import hashlib
import json
from pathlib import Path

import pytest
import scripts.evaluate_senf_components as evaluator

from scripts.evaluate_senf_components import (
    DEFAULT_GOLD,
    GoldSchemaError,
    evaluate,
    load_gold,
    main,
    render_markdown,
    validate_gold,
)


def test_versioned_gold_is_valid_and_has_semantic_component_coverage():
    gold = load_gold(DEFAULT_GOLD)

    assert gold["benchmark_version"] == 1
    assert {case["component"] for case in gold["cases"]} == {"spans", "identity", "exemplar", "weave"}
    identity = [case["gold"] for case in gold["cases"] if case["component"] == "identity"]
    weave = [case for case in gold["cases"] if case["component"] == "weave"]
    assert any(item["abstain_pairs"] for item in identity)
    assert any(item["id_minus"] for item in identity)
    assert any(item["gold"]["adapters"] for item in weave)
    assert any(len(item["documents"]) > 1 and item["gold"]["top_k_guards"] for item in weave)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda gold: gold.update(schema_version="future"), "schema_version"),
        (lambda gold: gold["cases"][0]["gold"]["mentions"][0].__setitem__(3, 1), "surface does not equal"),
        (lambda gold: gold["cases"][2]["gold"]["merge_pairs"].append(["missing:m0", "ip2:m0"]), "known"),
        (lambda gold: gold["cases"][2]["gold"]["merge_pairs"].append(["ip1:m99", "ip2:m0"]), "known"),
        (lambda gold: gold["cases"][0].update(unexpected=True), "unknown field"),
        (lambda gold: gold["cases"][2]["gold"]["non_merge_pairs"].append(["ip1:m0", "ip2:m0"]), "partition"),
        (lambda gold: gold["cases"][10]["gold"]["top_k_guards"].__setitem__(0, "not a guard"), "invalid format"),
        (lambda gold: gold["cases"][9]["gold"]["assignments"].append(["ex1:m0", "generic_game"]), "duplicate mention"),
        (lambda gold: gold["cases"][10]["gold"].update(adapters=[]), "allow requires"),
        (lambda gold: gold["cases"][10]["gold"]["adapters"][0].update(source_predicate="Wrong"), "source predicate"),
        (lambda gold: gold["cases"][10]["gold"]["mappings"]["frame"].__setitem__(0, "missing:f0->q1:f0"), "known source/query"),
    ],
)
def test_malformed_gold_fails_closed(mutation, match):
    gold = copy.deepcopy(load_gold(DEFAULT_GOLD))
    mutation(gold)

    with pytest.raises(GoldSchemaError, match=match):
        validate_gold(gold)


def test_evaluator_is_deterministic_and_reports_safety_metrics():
    report = evaluate(load_gold(DEFAULT_GOLD))
    metrics = report["metrics"]

    assert report == evaluate(load_gold(DEFAULT_GOLD))
    assert metrics["determinism"]["rate"] == 1.0
    assert metrics["false_merge_rate"] == 0.0
    assert metrics["false_bridge_rate"] == 0.0
    assert metrics["bridge_adapter"]["f1"] == 1.0
    assert metrics["residual"]["convergence_rate"] == 1.0
    assert metrics["spans"]["f1"] == 1.0
    assert metrics["top_k"]["hit_rate"] == 1.0
    assert metrics["top_k"]["wrong_guard_count"] == 1
    assert metrics["typed_mappings_micro"]["f1"] == 1.0
    assert set(metrics) >= {f"mapping_{name}" for name in ("frame", "entity", "role", "exemplar", "time", "location")}


def test_unannotated_identity_emission_is_scored_as_false_positive(monkeypatch):
    baseline = evaluate(load_gold(DEFAULT_GOLD))["metrics"]["id_plus"]
    original = evaluator._run_case

    def with_extra_edge(case, top_k):
        result = original(case, top_k)
        if case["id"] == "identity_definite_positive":
            result["id_plus"].append("id1:m0|id1:m1")
        return result

    monkeypatch.setattr(evaluator, "_run_case", with_extra_edge)
    metric = evaluate(load_gold(DEFAULT_GOLD))["metrics"]["id_plus"]
    assert metric["fp"] == baseline["fp"] + 1
    assert metric["precision"] < baseline["precision"]


def test_false_merge_rate_covers_all_explicit_non_merge_pairs(monkeypatch):
    gold = load_gold(DEFAULT_GOLD)
    denominator = sum(
        len(case["gold"]["non_merge_pairs"])
        for case in gold["cases"]
        if case["component"] == "identity"
    )
    original = evaluator._run_case

    def with_definite_false_merge(case, top_k):
        result = original(case, top_k)
        if case["id"] == "identity_pronoun_positive":
            result["merge_pairs"].append("ip1:m0|ip1:m1")
        return result

    monkeypatch.setattr(evaluator, "_run_case", with_definite_false_merge)
    assert evaluate(gold)["metrics"]["false_merge_rate"] == round(1 / denominator, 6)


def test_false_bridge_rate_is_bounded_case_level(monkeypatch):
    original = evaluator._run_case

    def with_multiple_denied_adapters(case, top_k):
        result = original(case, top_k)
        if case["id"] == "weave_bridge_deny_inverse":
            result["adapters"].extend([{"atom": "first"}, {"atom": "second"}])
        return result

    monkeypatch.setattr(evaluator, "_run_case", with_multiple_denied_adapters)
    metrics = evaluate(load_gold(DEFAULT_GOLD))["metrics"]
    assert metrics["false_bridge_rate"] == 1.0
    assert metrics["bridge_adapter"]["fp"] == 2


def test_wrong_guards_and_adapters_are_not_oracle_filtered(monkeypatch):
    baseline = evaluate(load_gold(DEFAULT_GOLD))["metrics"]
    original = evaluator._run_case

    def with_wrong_predictions(case, top_k):
        result = original(case, top_k)
        if case["id"] == "weave_typed_location_bridge_allow":
            result["top_k_guards"].append("wrong->q1")
            result["adapters"].append({
                "source_frame": "s1:f0", "query_frame": "q1:f0",
                "source_predicate": "Wrong", "query_predicate": "LocatedIn",
                "source_arguments": ["camera", "lab"], "query_arguments": ["camera", "lab"],
                "atom": "(: wrong (Wrong camera lab) (STV 1 1))",
            })
        return result

    monkeypatch.setattr(evaluator, "_run_case", with_wrong_predictions)
    metrics = evaluate(load_gold(DEFAULT_GOLD))["metrics"]
    assert metrics["top_k"]["hit_rate"] == 1.0
    assert metrics["top_k"]["wrong_guard_count"] == baseline["top_k"]["wrong_guard_count"] + 1
    assert metrics["top_k"]["extra_prediction_count"] == baseline["top_k"]["extra_prediction_count"] + 1
    assert metrics["bridge_adapter"]["fp"] == baseline["bridge_adapter"]["fp"] + 1


def test_json_and_markdown_cli_outputs(tmp_path):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    assert main(["--format", "json", "--output", str(json_path)]) == 0
    assert main(["--format", "markdown", "--output", str(md_path)]) == 0
    assert json.loads(json_path.read_text())["schema_version"] == "senf-components-report/v2"
    assert "# SENF Component Benchmark v1" in md_path.read_text()
    assert "False-merge rate" in render_markdown(evaluate(load_gold(DEFAULT_GOLD)))


def test_tracked_reports_match_the_evaluator():
    report = evaluate(load_gold(DEFAULT_GOLD), hashlib.sha256(DEFAULT_GOLD.read_bytes()).hexdigest())
    docs = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"

    json_report = copy.deepcopy(report)
    json_report["reproducibility"]["command"] = "python scripts/evaluate_senf_components.py --gold data/benchmarks/senf_components_v1.json --format json --output docs/benchmarks/senf_components_v1.json"
    markdown_report = copy.deepcopy(report)
    markdown_report["reproducibility"]["command"] = "python scripts/evaluate_senf_components.py --gold data/benchmarks/senf_components_v1.json --format markdown --output docs/benchmarks/senf_components_v1.md"
    tracked_json = json.loads((docs / "senf_components_v1.json").read_text())
    for candidate in (tracked_json, json_report):
        reproducibility = candidate["reproducibility"]
        assert reproducibility["git_revision"]
        assert reproducibility["environment"]["python"]
        assert reproducibility["environment"]["platform"]
        assert reproducibility["environment"]["dependencies"]
        reproducibility["git_revision"] = "<variable>"
        reproducibility["environment"] = {"recorded": "<variable>"}
    assert tracked_json == json_report

    tracked_markdown_report = copy.deepcopy(markdown_report)
    tracked_metadata = json.loads((docs / "senf_components_v1.json").read_text())["reproducibility"]
    tracked_markdown_report["reproducibility"]["git_revision"] = tracked_metadata["git_revision"]
    tracked_markdown_report["reproducibility"]["environment"] = tracked_metadata["environment"]
    assert (docs / "senf_components_v1.md").read_text() == render_markdown(tracked_markdown_report)


def test_cli_rejects_malformed_gold(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version": "senf-components-gold/v1"}')

    with pytest.raises(SystemExit) as exc:
        main(["--gold", str(malformed)])

    assert exc.value.code == 2
