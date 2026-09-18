"""Classify proof-bearing benchmark failures from an existing report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_grading import extract_proof_traces, grade_case, parse_trace, split_top_level
from core.symbol_normalization import canonical_symbol


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _query_atom(query: str) -> str:
    text = " ".join(str(query).split())
    if not text.startswith("(:") or not text.endswith(")"):
        return ""
    parts = split_top_level(text[2:-1].strip())
    return parts[1] if len(parts) == 3 and parts[1].startswith("(") else ""


def _symbols(atom: str) -> list[str]:
    values: list[str] = []
    for token in atom.replace("(", " ").replace(")", " ").split():
        if token.startswith(("$", "?")):
            continue
        value = canonical_symbol(token)
        if value and value not in values:
            values.append(value)
    return values


def _gold_symbols(gold: dict[str, Any]) -> set[str]:
    return {
        canonical_symbol(alias)
        for slot in gold.get("entities") or []
        for alias in slot
        if alias
    }


def _validate_inputs(row: dict[str, Any], gold: dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise ValueError("Benchmark result row must be an object")
    case = row.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
        raise ValueError("Benchmark result row must contain a string case.case_id")
    end_to_end = row.get("end_to_end")
    if not isinstance(end_to_end, dict) or not isinstance(end_to_end.get("query"), dict):
        raise ValueError(f"Case {case['case_id']} must contain end_to_end.query")
    entities = gold.get("entities")
    if not isinstance(entities, list) or not all(
        isinstance(slot, list)
        and bool(slot)
        and all(isinstance(alias, str) and alias for alias in slot)
        for slot in entities
    ):
        raise ValueError(f"Gold case {case['case_id']} has malformed entity slots")


def _malformed_query(query: str) -> bool:
    if any(char in query for char in "{}[]"):
        return True
    depth = 0
    in_string = False
    escaped = False
    for char in query:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string and char == "(":
            depth += 1
        elif not in_string and char == ")":
            depth -= 1
            if depth < 0:
                return True
    return in_string or depth != 0 or not _query_atom(query)


def _selected_score(diagnostics: dict[str, Any], executed_query: str) -> dict[str, Any]:
    candidates = diagnostics.get("candidate_score_breakdown") or []
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, dict) and candidate.get("query") == executed_query:
            return {
                "candidate_rank": index,
                "candidate_total_score": candidate.get("total"),
                "candidate_rejected": candidate.get("rejected"),
                "candidate_senf_components": (
                    candidate.get("senf_components")
                    if isinstance(candidate.get("senf_components"), dict)
                    else {}
                ),
            }
    return {
        "candidate_rank": None,
        "candidate_total_score": None,
        "candidate_rejected": None,
        "candidate_senf_components": {},
    }


def classify_failure(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    _validate_inputs(row, gold)
    case = row.get("case") or {}
    query = (row.get("end_to_end") or {}).get("query") or {}
    original_query = str(query.get("original_query") or "")
    executed_query = str(query.get("executed_query") or "")
    original_symbols = set(_symbols(_query_atom(original_query)))
    executed_symbols = set(_symbols(_query_atom(executed_query)))
    wanted = _gold_symbols(gold)

    conclusions = [
        conclusion
        for conclusion in (
            parse_trace(trace) for trace in extract_proof_traces(row)
        )
        if conclusion is not None
    ]
    grading = grade_case(gold, row)
    reason = str(grading.get("answer_reason") or "")

    flags: list[str] = []
    if query.get("query_status") == "weakly_aligned":
        flags.append("weak_alignment")
    if query.get("fallback_used"):
        flags.append("fallback_selected")
    if original_query != executed_query:
        flags.append("executed_query_changed")
    if _malformed_query(executed_query):
        flags.append("malformed_executed_query")
    if wanted and not original_symbols.intersection(wanted):
        flags.append("original_query_answer_symbol_miss")
    if wanted and not executed_symbols.intersection(wanted):
        flags.append("executed_query_answer_symbol_miss")
    unexpected_proof = gold.get("expected_proof") is False and bool(row.get("proof_found"))
    if unexpected_proof:
        flags.append("unexpected_proof")
    elif not grading.get("matched_entities"):
        flags.append("proof_answer_evidence_miss")
    if "verdict " in reason and ": unmet" in reason:
        flags.append("verdict_mismatch")
    if any(conclusion.strength == 0.0 for conclusion in conclusions):
        flags.append("nonpositive_proof")

    if not row.get("proof_found"):
        primary = "no_proof"
    elif "unexpected_proof" in flags:
        primary = "unexpected_proof"
    elif "malformed_executed_query" in flags:
        primary = "malformed_executed_query"
    elif "proof_answer_evidence_miss" in flags:
        primary = "proof_answer_evidence_miss"
    elif "verdict_mismatch" in flags or "nonpositive_proof" in flags:
        primary = "polarity_or_truth_value"
    else:
        primary = "unclassified"

    diagnostics = query.get("senf") if isinstance(query.get("senf"), dict) else {}
    return {
        "case_id": case.get("case_id"),
        "name": case.get("name"),
        "question": case.get("question") or case.get("user_query"),
        "primary_failure": primary,
        "evidence_flags": flags,
        "query_status": query.get("query_status"),
        "fallback_used": bool(query.get("fallback_used")),
        "original_query": original_query,
        "executed_query": executed_query,
        "proof_conclusions": [conclusion.atom for conclusion in conclusions],
        "proof_truth_values": [
            {"strength": conclusion.strength, "confidence": conclusion.confidence}
            for conclusion in conclusions
        ],
        "gold_entities": gold.get("entities") or [],
        "gold_verdict": gold.get("verdict"),
        "matched_entities": grading.get("matched_entities") or [],
        "answer_reason": reason,
        "senf": {
            "merge_count": diagnostics.get("merge_count"),
            "weave_total_cost": diagnostics.get("weave_total_cost"),
            "weave_distortion": diagnostics.get("weave_distortion"),
            "bridge_atom_count": diagnostics.get("bridge_atom_count"),
            **_selected_score(diagnostics, executed_query),
        },
    }


def analyze_report(
    report: dict[str, Any],
    gold_payload: dict[str, Any],
    parser: str,
    *,
    include_no_proof: bool = False,
) -> dict[str, Any]:
    parsers = report.get("parsers")
    if not isinstance(parsers, dict) or not isinstance(parsers.get(parser), list):
        raise ValueError(f"Benchmark report has no parser arm {parser!r}")
    gold_cases = gold_payload.get("cases")
    if not isinstance(gold_cases, dict):
        raise ValueError("Gold file must contain a cases object")

    for row in parsers[parser]:
        if not isinstance(row, dict):
            raise ValueError(f"Parser arm {parser!r} contains a non-object result row")
    incorrect = [row for row in parsers[parser] if row.get("answer_correct") is False]
    selected = [
        row for row in incorrect if include_no_proof or row.get("proof_found") is True
    ]
    failures = []
    for row in selected:
        case_id = str((row.get("case") or {}).get("case_id") or "")
        gold = gold_cases.get(case_id)
        if not isinstance(gold, dict):
            raise ValueError(f"Missing gold case {case_id!r}")
        failures.append(classify_failure(row, gold))

    primary_counts = Counter(item["primary_failure"] for item in failures)
    flag_counts = Counter(flag for item in failures for flag in item["evidence_flags"])
    return {
        "analysis_schema_version": 1,
        "suite": report.get("suite"),
        "run_id": report.get("run_id"),
        "parser": parser,
        "scope": (
            "incorrect answer-graded cases with proofs"
            if not include_no_proof
            else "all incorrect answer-graded cases"
        ),
        "summary": {
            "analyzed_failures": len(failures),
            "proof_bearing_incorrect": sum(
                row.get("answer_correct") is False and row.get("proof_found") is True
                for row in parsers[parser]
            ),
            "no_proof_incorrect": sum(
                row.get("answer_correct") is False and not row.get("proof_found")
                for row in parsers[parser]
            ),
            "primary_failure_counts": dict(sorted(primary_counts.items())),
            "evidence_flag_counts": dict(sorted(flag_counts.items())),
        },
        "failures": failures,
    }


def render_markdown(analysis: dict[str, Any], report_sha256: str) -> str:
    summary = analysis["summary"]
    lines = [
        "# Proof-Bearing Failure Taxonomy",
        "",
        f"Suite: `{analysis.get('suite')}`  ",
        f"Run: `{analysis.get('run_id')}`  ",
        f"Parser: `{analysis.get('parser')}`  ",
        f"Source report SHA-256: `{report_sha256}`",
        "",
        "This is a deterministic symptom taxonomy, not a root-cause annotation. "
        "Proof evidence uses the benchmark grader. Query answer-symbol misses are neutral "
        "diagnostics: a valid query need not contain the answer it is intended to derive.",
        "",
        f"Proof-bearing incorrect cases: **{summary['proof_bearing_incorrect']}**  ",
        f"Incorrect cases without proofs: **{summary['no_proof_incorrect']}**",
        "",
        "| Case | Primary symptom | Status | Candidate | Evidence | Executed query | Proof conclusion |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in analysis["failures"]:
        candidate = item["senf"].get("candidate_rank")
        evidence = ", ".join(item["evidence_flags"])
        conclusions = "; ".join(item["proof_conclusions"])
        values = [
            item["case_id"],
            item["primary_failure"],
            item["query_status"],
            candidate if candidate is not None else "-",
            evidence,
            item["executed_query"],
            conclusions,
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(["", "## Counts", ""])
    for name, count in summary["primary_failure_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Evidence Flags", ""])
    for name, count in summary["evidence_flag_counts"].items():
        lines.append(f"- `{name}`: {count}")
    return "\n".join(lines) + "\n"


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("report", type=Path)
    cli.add_argument("--gold-file", type=Path, required=True)
    cli.add_argument("--parser", default="canonical_senf_pln_stage7")
    cli.add_argument("--include-no-proof", action="store_true")
    cli.add_argument("--json-output", type=Path)
    cli.add_argument("--markdown-output", type=Path)
    args = cli.parse_args()

    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    gold = _load_object(args.gold_file)
    analysis = analyze_report(
        report, gold, args.parser, include_no_proof=args.include_no_proof
    )
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    analysis["source_report"] = args.report.name
    analysis["source_report_sha256"] = report_hash

    rendered_json = json.dumps(analysis, indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(rendered_json, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(
            render_markdown(analysis, report_hash), encoding="utf-8"
        )
    if not args.json_output and not args.markdown_output:
        print(rendered_json, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
