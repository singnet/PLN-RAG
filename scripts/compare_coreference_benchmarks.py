from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TIMING_FIELDS = ("parse_only_seconds", "ingest_seconds", "query_seconds", "total_seconds")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _indexed_rows(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for parser_name, rows in (report.get("parsers") or {}).items():
        for row in rows:
            case = row.get("case") or {}
            case_id = str(row.get("case_id") or case.get("case_id") or case.get("id") or case.get("name"))
            key = (str(parser_name), case_id)
            if key in indexed:
                raise ValueError(f"duplicate parser/case ID: {parser_name}/{case_id}")
            indexed[key] = row
    return indexed


def _numeric_delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(right) - float(left), 6)


def _coref_latency(row: dict[str, Any]) -> float | None:
    values = []
    ingest = row.get("end_to_end", {}).get("ingest", [])
    for item in ingest:
        coref = item.get("coreference") or {}
        value = coref.get("duration_seconds")
        if coref.get("status") != "disabled" and isinstance(value, (int, float)):
            values.append(float(value))
    if values:
        return sum(values)
    if ingest and all((item.get("coreference") or {}).get("status") == "disabled" for item in ingest):
        return 0.0
    return None


def _coref_activity(row: dict[str, Any]) -> tuple[bool, int]:
    items = row.get("end_to_end", {}).get("ingest", [])
    changed = any(bool((item.get("coreference") or {}).get("changed")) for item in items)
    replacements = sum(
        int((item.get("coreference") or {}).get("replacement_count") or 0)
        for item in items
    )
    return changed, replacements


def _validate_reports(left: dict[str, Any], right: dict[str, Any]) -> str:
    if not left.get("valid") or not right.get("valid"):
        raise ValueError("both benchmark reports must be valid")
    left_pair = left.get("pair_id")
    right_pair = right.get("pair_id")
    if not left_pair or not right_pair or left_pair != right_pair:
        raise ValueError("benchmark reports must have the same non-empty pair ID")
    left_backend = (left.get("coref") or {}).get("backend")
    right_backend = (right.get("coref") or {}).get("backend")
    if not left_backend or not right_backend or left_backend == right_backend:
        raise ValueError("benchmark reports must use different coreference backends")
    if left.get("config") != right.get("config"):
        raise ValueError("benchmark configurations do not match")
    if left.get("suite_metadata") != right.get("suite_metadata"):
        raise ValueError("benchmark suite metadata does not match")
    if left.get("git") != right.get("git"):
        raise ValueError("benchmark source revisions do not match")
    left_cassette = left.get("lm_cassette") or {"mode": "off"}
    right_cassette = right.get("lm_cassette") or {"mode": "off"}
    if left_cassette.get("mode") != right_cassette.get("mode"):
        raise ValueError("benchmark cassette modes do not match")
    if left_cassette.get("mode") != "off" and left_cassette.get("path") != right_cassette.get("path"):
        raise ValueError("benchmark cassette paths do not match")
    for cassette in (left_cassette, right_cassette):
        if cassette.get("mode") == "replay" and (
            not cassette.get("replay_valid")
            or cassette.get("misses")
            or cassette.get("unconsumed")
        ):
            raise ValueError("benchmark cassette replay is incomplete or invalid")
    return str(left_pair)


def compare_reports(
    left: dict[str, Any],
    right: dict[str, Any],
    labels: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    pair_id = _validate_reports(left, right)
    left_rows = _indexed_rows(left)
    right_rows = _indexed_rows(right)
    if set(left_rows) != set(right_rows):
        missing_right = sorted(set(left_rows) - set(right_rows))
        missing_left = sorted(set(right_rows) - set(left_rows))
        raise ValueError(f"case IDs do not match (missing_right={missing_right}, missing_left={missing_left})")

    transitions = {"false_to_true": 0, "true_to_false": 0, "unchanged_true": 0, "unchanged_false": 0}
    query_changes = 0
    rows = []
    timing_samples: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    timing_samples["coreference_seconds"] = []
    rewritten_cases = 0
    query_changes_without_rewrite = 0
    quality_transitions = {
        "fail_to_pass": 0,
        "pass_to_fail": 0,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
    }
    left_quality = {}
    right_quality = {}
    if labels is not None:
        try:
            from scripts.evaluate_coreference_benchmark import evaluate_report
        except ModuleNotFoundError:
            from evaluate_coreference_benchmark import evaluate_report

        for report, destination in ((left, left_quality), (right, right_quality)):
            evaluation = evaluate_report(report, labels, score_threshold)
            for parser_name, parser_data in evaluation["parsers"].items():
                for case in parser_data["cases"]:
                    destination[(parser_name, case["case_id"])] = case
    for key in sorted(left_rows):
        before = left_rows[key]
        after = right_rows[key]
        if not before.get("case_hash") or before.get("case_hash") != after.get("case_hash"):
            raise ValueError(f"case hash does not match for {key[0]}/{key[1]}")
        before_proof = bool(before.get("proof_found"))
        after_proof = bool(after.get("proof_found"))
        transition = (
            "false_to_true" if not before_proof and after_proof else
            "true_to_false" if before_proof and not after_proof else
            "unchanged_true" if before_proof else "unchanged_false"
        )
        transitions[transition] += 1
        before_query = before.get("end_to_end", {}).get("query", {})
        after_query = after.get("end_to_end", {}).get("query", {})
        changed_fields = [
            field for field in ("original_query", "executed_query", "pln_query")
            if before_query.get(field) != after_query.get(field)
        ]
        query_changes += bool(changed_fields)
        coreference_changed, replacement_count = _coref_activity(after)
        rewritten_cases += int(coreference_changed)
        query_changes_without_rewrite += int(bool(changed_fields) and not coreference_changed)
        deltas = {}
        for field in TIMING_FIELDS:
            delta = _numeric_delta(before.get("timing", {}).get(field), after.get("timing", {}).get(field))
            deltas[field] = delta
            if delta is not None:
                timing_samples[field].append(delta)
        coref_delta = _numeric_delta(_coref_latency(before), _coref_latency(after))
        deltas["coreference_seconds"] = coref_delta
        if coref_delta is not None:
            timing_samples["coreference_seconds"].append(coref_delta)
        row = {
            "parser": key[0],
            "case_id": key[1],
            "case_hash": before["case_hash"],
            "proof_before": before_proof,
            "proof_after": after_proof,
            "proof_transition": transition,
            "query_changed": bool(changed_fields),
            "query_changed_fields": changed_fields,
            "coreference_changed": coreference_changed,
            "coreference_replacements": replacement_count,
            "query_before": {
                field: before_query.get(field)
                for field in ("original_query", "executed_query", "pln_query")
            },
            "query_after": {
                field: after_query.get(field)
                for field in ("original_query", "executed_query", "pln_query")
            },
            "timing_delta_seconds": deltas,
        }
        if labels is not None:
            before_quality = left_quality[key]
            after_quality = right_quality[key]
            before_pass = bool(before_quality["passed"])
            after_pass = bool(after_quality["passed"])
            quality_transition = (
                "fail_to_pass" if not before_pass and after_pass else
                "pass_to_fail" if before_pass and not after_pass else
                "unchanged_pass" if before_pass else "unchanged_fail"
            )
            quality_transitions[quality_transition] += 1
            row["quality_before"] = before_quality
            row["quality_after"] = after_quality
            row["quality_transition"] = quality_transition
        rows.append(row)

    timing_summary = {
        field: {
            "samples": len(values),
            "mean_delta": round(sum(values) / len(values), 6) if values else None,
            "median_delta": round(statistics.median(values), 6) if values else None,
        }
        for field, values in timing_samples.items()
    }
    warnings = []
    mode = (right.get("config") or {}).get("mode")
    if mode != "isolated":
        warnings.append("cumulative_mode_allows_cross_case_leakage")
    if mode == "isolated" and query_changes_without_rewrite:
        warnings.append("query_changed_without_coreference_rewrite")
    cassette_modes = {
        (report.get("lm_cassette") or {}).get("mode", "off")
        for report in (left, right)
    }
    if len(cassette_modes) != 1 or cassette_modes == {"off"}:
        warnings.append("llm_not_deterministically_controlled")
    timing_warnings = []
    cassette_mode = next(iter(cassette_modes)) if len(cassette_modes) == 1 else None
    if cassette_mode == "capture" and any(
        (report.get("lm_cassette") or {}).get("live_calls", 0)
        for report in (left, right)
    ):
        timing_warnings.append("capture_timing_mixes_live_and_replayed_lm_calls")
    elif cassette_mode == "replay":
        timing_warnings.append("replay_timing_excludes_live_lm_latency")
    return {
        "schema_version": 1,
        "pair_id": pair_id,
        "left": {"run_id": left.get("run_id"), "coref": left.get("coref")},
        "right": {"run_id": right.get("run_id"), "coref": right.get("coref")},
        "matched_cases": len(rows),
        "proof_transitions": transitions,
        "quality_transitions": quality_transitions if labels is not None else None,
        "query_changes": query_changes,
        "coreference_changed_cases": rewritten_cases,
        "query_changes_without_rewrite": query_changes_without_rewrite,
        "causal_interpretation_valid": not warnings,
        "warnings": warnings,
        "provider_inclusive_timing_valid": not timing_warnings,
        "timing_warnings": timing_warnings,
        "timing_deltas_seconds": timing_summary,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two paired coreference benchmark reports.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--output")
    parser.add_argument("--labels", help="Optional proof-quality label sidecar")
    parser.add_argument("--score-threshold", type=float)
    args = parser.parse_args()
    try:
        labels = _load(Path(args.labels)) if args.labels else None
        comparison = compare_reports(
            _load(Path(args.left)),
            _load(Path(args.right)),
            labels,
            args.score_threshold,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(comparison, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
