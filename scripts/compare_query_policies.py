"""Compare paired query-execution-policy arms and prepare blind adjudication."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_grading import extract_proof_traces, parse_trace


POLICY_ARMS = {
    "ranked_first_proof": "canonical_senf_pln_stage7_ranked_first_proof",
    "ranked_first_only": "canonical_senf_pln_stage7_ranked_first_only",
    "original_only": "canonical_senf_pln_stage7_original_only",
}
ANSWER_LABELS = ("adequate", "qualified", "insufficient", "incorrect")
EXPECTED_POLICY_SETTINGS = {
    "answer_generation_enabled": "false",
    "source_lookup_max_atoms": "0",
    "query_candidate_max_tries": "5",
    "chunk_size": "512",
    "chunk_overlap": "64",
    "context_top_k": "10",
    "parser_batch_sentences": "4",
    "parser_batch_max_chars": "2000",
    "chaining_timeout": "30",
    "chaining_max_steps": "100",
    "senf_identity_threshold": "0.75",
    "senf_context_top_k": "10",
    "senf_session_max_frames": "200",
    "senf_use_vector_context": "false",
    "senf_exemplar_enabled": "true",
    "senf_emit_bridge_atoms": "false",
    "senf_weave_top_k": "3",
    "senf_query_max_priors": "16",
    "senf_query_max_source_frames": "128",
    "senf_query_max_mentions": "256",
    "senf_query_max_candidate_work": "32",
    "senf_weave_per_source_k": "3",
    "senf_weave_beam_width": "32",
    "senf_weave_max_frames": "64",
    "senf_weave_max_pair_candidates": "256",
    "senf_weave_max_exemplar_alternatives": "4",
    "senf_weave_max_cost": "2.0",
    "senf_weave_engine": "beam",
    "senf_weave_global_candidate_cap": "512",
    "senf_weave_max_cells": "65536",
    "senf_weave_max_seeds": "32",
    "senf_weave_max_iterations": "200",
    "senf_weave_sinkhorn_tolerance": "0.0000001",
    "senf_weave_sinkhorn_regularization": "0.25",
    "senf_weave_forget_fine_costs": "false",
    "senf_weave_coarse_identity": "false",
    "senf_weave_max_conflict_cost": "1000000.0",
    "senf_weave_max_transport_cost": "1000000.0",
    "senf_weave_require_context_match": "false",
    "senf_source_grounding_weight": "3",
    "senf_role_compat_weight": "2",
    "senf_distortion_weight": "0",
    "senf_identity_support_weight": "2",
    "senf_exemplar_coherence_weight": "2",
    "senf_conflict_weight": "3",
    "senf_transport_cost_weight": "2",
    "senf_matched_soft_mass_weight": "0",
    "senf_global_residual_ratio_weight": "0",
    "senf_alignment_confidence_weight": "0",
    "senf_diagnostics_max_weaves": "3",
    "senf_diagnostics_max_items": "128",
    "senf_diagnostics_max_evidence": "16",
    "senf_branch_max_nodes": "64",
    "senf_branch_max_depth": "16",
    "senf_branch_max_theory_statements": "128",
    "senf_temporal_decay_rate": "0.01",
}


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark report must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _rows_by_id(rows: Any, arm: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"Missing result list for {arm}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Arm {arm} contains a non-object result")
        case_id = str((row.get("case") or {}).get("case_id") or "")
        if not case_id or case_id in indexed:
            raise ValueError(f"Arm {arm} has a missing or duplicate case_id {case_id!r}")
        indexed[case_id] = row
    return indexed


def _ingest_signature(row: dict[str, Any]) -> list[dict[str, Any]]:
    ingest = (row.get("end_to_end") or {}).get("ingest") or []
    return [
        {
            "status": item.get("status"),
            "atoms": item.get("atoms"),
            "rejected_count": item.get("rejected_count", 0),
        }
        for item in ingest
        if isinstance(item, dict)
    ]


def _proof_conclusions(row: dict[str, Any]) -> list[str]:
    return [
        conclusion.atom
        for conclusion in (
            parse_trace(trace) for trace in extract_proof_traces(row)
        )
        if conclusion is not None
    ]


def _outcome(row: dict[str, Any]) -> dict[str, Any]:
    query = (row.get("end_to_end") or {}).get("query") or {}
    return {
        "proof_found": bool(row.get("proof_found")),
        "answer_correct": row.get("answer_correct"),
        "answer_score": row.get("answer_score"),
        "answer_reason": row.get("answer_reason"),
        "original_query": query.get("original_query"),
        "executed_query": query.get("executed_query"),
        "query_status": query.get("query_status"),
        "proof": query.get("proof"),
        "proof_conclusions": _proof_conclusions(row),
        "candidate_count": query.get("candidate_count"),
        "candidate_count_tried": query.get("candidate_count_tried"),
        "attempted_candidate_indices": query.get("attempted_candidate_indices"),
        "successful_candidate_index": query.get("successful_candidate_index"),
    }


def compare_report(
    report: dict[str, Any],
    *,
    expected_tape_sha256: str | None = None,
) -> dict[str, Any]:
    parsers = report.get("parsers")
    metadata = report.get("arm_metadata")
    if report.get("valid") is not True or report.get("validity_warnings"):
        raise ValueError("Benchmark report is not marked valid")
    if report.get("mode") != "isolated":
        raise ValueError("Query-policy comparison requires isolated mode")
    if not isinstance(parsers, dict) or not isinstance(metadata, dict):
        raise ValueError("Report must contain parsers and arm_metadata objects")
    tape = report.get("generation_tape") or {}
    tape_hash = tape.get("sha256")
    if tape.get("mode") != "replay":
        raise ValueError("Query-policy comparison requires generation replay")
    if expected_tape_sha256 and tape_hash != expected_tape_sha256:
        raise ValueError(
            f"Generation tape hash mismatch: expected {expected_tape_sha256}, got {tape_hash}"
        )

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    reference_settings: dict[str, Any] | None = None
    for policy, arm in POLICY_ARMS.items():
        arm_metadata = metadata.get(arm)
        if not isinstance(arm_metadata, dict):
            raise ValueError(f"Missing arm metadata for {arm}")
        if arm_metadata.get("parser") != "canonical_senf_pln":
            raise ValueError(f"Arm {arm} does not use canonical_senf_pln")
        if arm_metadata.get("senf_counterfactual_enabled") is not True:
            raise ValueError(f"Arm {arm} does not enable Stage 7")
        if arm_metadata.get("query_execution_policy") != policy:
            raise ValueError(f"Arm {arm} has the wrong execution policy")
        settings = arm_metadata.get("settings") or {}
        if settings != EXPECTED_POLICY_SETTINGS:
            raise ValueError(f"Arm {arm} does not use the frozen policy settings")
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("Query-policy arm settings differ")

        indexed[policy] = _rows_by_id(parsers.get(arm), arm)
        case_ids = set(indexed[policy])
        if reference_ids is None:
            reference_ids = case_ids
        elif case_ids != reference_ids:
            raise ValueError("Query-policy arms have different case sets")

    expected_runs = {
        (arm, case_id)
        for arm in POLICY_ARMS.values()
        for case_id in (reference_ids or set())
    }
    parser_runs = tape.get("parser_runs")
    if not isinstance(parser_runs, list):
        raise ValueError("Generation replay diagnostics are missing")
    actual_runs = set()
    for run in parser_runs:
        if not isinstance(run, dict):
            raise ValueError("Generation replay diagnostics are malformed")
        actual_runs.add((run.get("parser"), run.get("case_id")))
        if (
            int(run.get("calls", 0) or 0) <= 0
            or int(run.get("context_mismatches", 0) or 0) != 0
            or int(run.get("repeated_calls", 0) or 0) != 0
            or run.get("unconsumed_calls")
        ):
            raise ValueError("Generation replay diagnostics report drift")
    if actual_runs != expected_runs:
        raise ValueError("Generation replay diagnostics do not cover every arm/case")

    cases = []
    ingest_rejection_cases: list[str] = []
    discordance = Counter()
    for case_id in sorted(reference_ids or set()):
        rows = {policy: indexed[policy][case_id] for policy in POLICY_ARMS}
        for policy, row in rows.items():
            if row.get("error") or row.get("answer_correct") is None:
                raise ValueError(f"Invalid or ungraded result for {case_id}/{policy}")
            query = (row.get("end_to_end") or {}).get("query") or {}
            if query.get("execution_policy") != policy:
                raise ValueError(f"Execution-policy telemetry mismatch for {case_id}/{policy}")
        parse_only = [row.get("parse_only") for row in rows.values()]
        ingest = [_ingest_signature(row) for row in rows.values()]
        if any(value != parse_only[0] for value in parse_only[1:]):
            raise ValueError(f"Parse-only output differs across arms for {case_id}")
        if any(value != ingest[0] for value in ingest[1:]):
            raise ValueError(f"Ingest output differs across arms for {case_id}")
        if not ingest[0] or any(item.get("status") != "success" for item in ingest[0]):
            raise ValueError(f"Ingestion failed for {case_id}")
        if any(item.get("rejected_count") for item in ingest[0]):
            ingest_rejection_cases.append(case_id)

        outcomes = {policy: _outcome(row) for policy, row in rows.items()}
        current = outcomes["ranked_first_proof"]
        original = outcomes["original_only"]
        current_correct = current["answer_correct"] is True
        original_correct = original["answer_correct"] is True
        if current_correct and original_correct:
            discordance["both_automatic_correct"] += 1
        elif current_correct:
            discordance["current_only_automatic_correct"] += 1
        elif original_correct:
            discordance["original_only_automatic_correct"] += 1
        else:
            discordance["neither_automatic_correct"] += 1

        changed = len({
            (outcome["executed_query"], outcome["proof"])
            for outcome in outcomes.values()
        }) > 1
        cases.append({
            "case_id": case_id,
            "question": (rows["ranked_first_proof"].get("case") or {}).get("question"),
            "outputs_changed": changed,
            "outcomes": outcomes,
        })

    summaries = {}
    for policy in POLICY_ARMS:
        outcomes = [case["outcomes"][policy] for case in cases]
        summaries[policy] = {
            "cases": len(outcomes),
            "proof_found": sum(item["proof_found"] for item in outcomes),
            "automatic_correct": sum(item["answer_correct"] is True for item in outcomes),
            "average_candidates_tried": round(
                sum(int(item["candidate_count_tried"] or 0) for item in outcomes)
                / len(outcomes),
                4,
            ) if outcomes else 0.0,
        }
    return {
        "comparison_schema_version": 1,
        "suite": report.get("suite"),
        "run_id": report.get("run_id"),
        "generation_tape_sha256": tape_hash,
        "policies": POLICY_ARMS,
        "settings": reference_settings or {},
        "summary": summaries,
        "automatic_discordance": dict(sorted(discordance.items())),
        "changed_case_count": sum(case["outputs_changed"] for case in cases),
        "ingest_rejection_cases": ingest_rejection_cases,
        "cases": cases,
    }


def adjudication_packet(
    comparison: dict[str, Any], report: dict[str, Any], source_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser_rows = {
        policy: _rows_by_id(report["parsers"][arm], arm)
        for policy, arm in POLICY_ARMS.items()
    }
    packet_cases = []
    key_cases: dict[str, dict[str, str]] = {}
    for case in comparison["cases"]:
        if not case["outputs_changed"]:
            continue
        case_id = case["case_id"]
        # Deterministic blinding keeps regenerated packets byte-stable.
        ordered = _blinded_policy_order(source_sha256, case_id)
        labels = ["A", "B", "C"]
        key_cases[case_id] = dict(zip(labels, ordered))
        base_case = parser_rows["ranked_first_proof"][case_id].get("case") or {}
        outputs = []
        for label, policy in zip(labels, ordered):
            outcome = case["outcomes"][policy]
            outputs.append({
                "label": label,
                "executed_query": outcome["executed_query"],
                "proof": outcome["proof"],
                "proof_conclusions": outcome["proof_conclusions"],
                "adjudication": {
                    "query_fidelity": None,
                    "source_supported": None,
                    "preserves_polarity_and_hedging": None,
                    "answer_label": None,
                    "confidence": None,
                    "rationale": "",
                },
            })
        packet_cases.append({
            "case_id": case_id,
            "question": base_case.get("question") or base_case.get("user_query"),
            "input_text": base_case.get("input_text") or "\n".join(base_case.get("texts") or []),
            "outputs": outputs,
        })
    packet = {
        "adjudication_schema_version": 1,
        "status": "unadjudicated",
        "source_report_sha256": source_sha256,
        "labels": {
            "answer_label": list(ANSWER_LABELS),
            "query_fidelity": ["valid", "partial", "invalid"],
            "source_supported": ["yes", "no", "unclear"],
            "preserves_polarity_and_hedging": ["yes", "no", "not_applicable"],
            "confidence": ["high", "medium", "low"],
        },
        "instructions": (
            "Judge each output against only the case text and question. Adequate directly and "
            "sufficiently answers; qualified is useful and truthful but incomplete; insufficient "
            "is related but does not answer; incorrect is unsupported, contradictory, malformed, "
            "or absent. Do not inspect the separate policy key before adjudication."
        ),
        "cases": packet_cases,
    }
    key = {
        "adjudication_key_schema_version": 1,
        "source_report_sha256": source_sha256,
        "cases": key_cases,
    }
    return packet, key


def _blinded_policy_order(source_sha256: str, case_id: str) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{source_sha256}:{case_id}".encode()).digest()
    return list(itertools.permutations(POLICY_ARMS))[digest[0] % 6]


def _content_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_manual_labels(
    comparison: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    if labels.get("source_report_sha256") != comparison.get("source_report_sha256"):
        raise ValueError("Manual labels reference a different source report")
    if labels.get("adjudication_packet_sha256") != comparison.get(
        "adjudication_packet_sha256"
    ):
        raise ValueError("Manual labels reference a different adjudication packet")
    label_cases = labels.get("cases")
    if not isinstance(label_cases, dict):
        raise ValueError("Manual labels must contain a cases object")
    expected_ids = {case["case_id"] for case in comparison["cases"]}
    if set(label_cases) != expected_ids:
        raise ValueError("Manual labels do not cover the comparison case set")

    counts: dict[str, Counter] = {policy: Counter() for policy in POLICY_ARMS}
    blind_counts: dict[str, Counter] = {policy: Counter() for policy in POLICY_ARMS}
    confusion = {
        policy: {
            "automatic_true_manual_usable": 0,
            "automatic_true_manual_not_usable": 0,
            "automatic_false_manual_usable": 0,
            "automatic_false_manual_not_usable": 0,
        }
        for policy in POLICY_ARMS
    }
    blind_confusion = {
        policy: {name: 0 for name in values}
        for policy, values in confusion.items()
    }
    transitions = Counter()
    by_id = {case["case_id"]: case for case in comparison["cases"]}
    for case_id, policy_labels in label_cases.items():
        if not isinstance(policy_labels, dict):
            raise ValueError(f"Manual labels for {case_id} must be an object")
        for policy in POLICY_ARMS:
            label = policy_labels.get(policy)
            if label not in ANSWER_LABELS:
                raise ValueError(f"Invalid manual label for {case_id}/{policy}: {label!r}")
            counts[policy][label] += 1
            automatic = by_id[case_id]["outcomes"][policy]["answer_correct"] is True
            usable = label in {"adequate", "qualified"}
            confusion[policy][
                f"automatic_{'true' if automatic else 'false'}_manual_"
                f"{'usable' if usable else 'not_usable'}"
            ] += 1
            if by_id[case_id]["outputs_changed"]:
                blind_counts[policy][label] += 1
                blind_confusion[policy][
                    f"automatic_{'true' if automatic else 'false'}_manual_"
                    f"{'usable' if usable else 'not_usable'}"
                ] += 1
        transitions[
            f"{policy_labels['ranked_first_proof']} -> {policy_labels['original_only']}"
        ] += 1

    summary = {}
    blind_summary = {}
    for policy, policy_counts in counts.items():
        summary[policy] = {
            **{label: policy_counts[label] for label in ANSWER_LABELS},
            "strict_correct": policy_counts["adequate"],
            "usable_correct": policy_counts["adequate"] + policy_counts["qualified"],
            "automatic_confusion": confusion[policy],
        }
        changed_counts = blind_counts[policy]
        blind_summary[policy] = {
            **{label: changed_counts[label] for label in ANSWER_LABELS},
            "strict_correct": changed_counts["adequate"],
            "usable_correct": changed_counts["adequate"] + changed_counts["qualified"],
            "automatic_confusion": blind_confusion[policy],
        }
    return {
        "status": labels.get("status"),
        "notes": labels.get("notes"),
        "summary": summary,
        "blind_changed_summary": blind_summary,
        "current_to_original_transitions": dict(sorted(transitions.items())),
    }


def manual_agreement(
    label_sets: list[dict[str, Any]], case_ids: set[str] | None = None
) -> dict[str, Any]:
    if len(label_sets) < 2:
        return {}
    if len(label_sets) != 2:
        raise ValueError("Manual agreement currently requires exactly two reviewers")
    first, second = label_sets[:2]
    if first.get("source_report_sha256") != second.get("source_report_sha256"):
        raise ValueError("Manual reviewers reference different source reports")
    if first.get("adjudication_packet_sha256") != second.get(
        "adjudication_packet_sha256"
    ):
        raise ValueError("Manual reviewers reference different adjudication packets")
    if set(first["cases"]) != set(second["cases"]):
        raise ValueError("Manual reviewers cover different case sets")
    result = {}
    for policy in POLICY_ARMS:
        exact = 0
        usable = 0
        compared_ids = sorted(case_ids or set(first["cases"]))
        for case_id in compared_ids:
            left = first["cases"][case_id][policy]
            right = second["cases"][case_id][policy]
            if left not in ANSWER_LABELS or right not in ANSWER_LABELS:
                raise ValueError("Manual agreement received an invalid label")
            exact += left == right
            usable += (left in {"adequate", "qualified"}) == (
                right in {"adequate", "qualified"}
            )
        result[policy] = {
            "exact_label_agreement": exact,
            "usable_binary_agreement": usable,
            "cases": len(compared_ids),
        }
    return result


def render_markdown(comparison: dict[str, Any], source_sha256: str) -> str:
    lines = [
        "# Query Execution Policy Comparison",
        "",
        f"Source report SHA-256: `{source_sha256}`  ",
        f"Generation tape SHA-256: `{comparison.get('generation_tape_sha256')}`",
        "",
        "| Policy | Proofs | Automatic correct | Avg candidates tried |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy, summary in comparison["summary"].items():
        lines.append(
            f"| `{policy}` | {summary['proof_found']}/{summary['cases']} | "
            f"{summary['automatic_correct']}/{summary['cases']} | "
            f"{summary['average_candidates_tried']:.4f} |"
        )
    manual_reviews = comparison.get("manual_adjudications") or []
    if manual_reviews:
        lines.extend([
            "",
            "## Manual Adjudication",
            "",
        ])
        for review in manual_reviews:
            review_summary = review["blind_changed_summary"]
            lines.extend([
                f"Status: `{review.get('status')}`",
                "Scope: 20 changed cases from the hash-bound blind packet. "
                "The JSON also records five unchanged inherited labels, which are excluded here.",
                "",
                "| Policy | Adequate | Qualified | Insufficient | Incorrect | Strict | Usable |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for policy, summary in review_summary.items():
                lines.append(
                    f"| `{policy}` | {summary['adequate']} | {summary['qualified']} | "
                    f"{summary['insufficient']} | {summary['incorrect']} | "
                    f"{summary['strict_correct']}/{sum(summary[label] for label in ANSWER_LABELS)} | "
                    f"{summary['usable_correct']}/{sum(summary[label] for label in ANSWER_LABELS)} |"
                )
            lines.extend([
                "",
                "| Policy | Grader TP | Grader FP | Grader FN | Grader TN |",
                "| --- | ---: | ---: | ---: | ---: |",
            ])
            for policy, summary in review_summary.items():
                confusion = summary["automatic_confusion"]
                lines.append(
                    f"| `{policy}` | "
                    f"{confusion['automatic_true_manual_usable']} | "
                    f"{confusion['automatic_true_manual_not_usable']} | "
                    f"{confusion['automatic_false_manual_usable']} | "
                    f"{confusion['automatic_false_manual_not_usable']} |"
                )
            lines.append("")
        agreement = comparison.get("manual_agreement") or {}
        if agreement:
            lines.extend([
                "| Policy | Exact label agreement | Usable/not-usable agreement |",
                "| --- | ---: | ---: |",
            ])
            for policy, values in agreement.items():
                lines.append(
                    f"| `{policy}` | {values['exact_label_agreement']}/{values['cases']} | "
                    f"{values['usable_binary_agreement']}/{values['cases']} |"
                )
    lines.extend([
        "",
        f"Changed outputs requiring blind adjudication: **{comparison['changed_case_count']}**",
        "",
        "| Case | Changed | Current query | Original-only query |",
        "| --- | --- | --- | --- |",
    ])
    for case in comparison["cases"]:
        if not case["outputs_changed"]:
            continue
        current = str(case["outcomes"]["ranked_first_proof"]["executed_query"] or "")
        original = str(case["outcomes"]["original_only"]["executed_query"] or "")
        current = current.replace("|", "\\|")
        original = original.replace("|", "\\|")
        lines.append(
            f"| {case['case_id']} | yes | {current} | {original} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("report", type=Path)
    cli.add_argument("--expected-tape-sha256")
    cli.add_argument("--json-output", type=Path, required=True)
    cli.add_argument("--markdown-output", type=Path, required=True)
    cli.add_argument("--adjudication-output", type=Path, required=True)
    cli.add_argument("--adjudication-key-output", type=Path, required=True)
    cli.add_argument("--manual-labels", type=Path, action="append", default=[])
    args = cli.parse_args()

    report, source_hash = _load(args.report)
    comparison = compare_report(
        report, expected_tape_sha256=args.expected_tape_sha256
    )
    comparison["source_report"] = args.report.name
    comparison["source_report_sha256"] = source_hash
    packet, key = adjudication_packet(comparison, report, source_hash)
    comparison["adjudication_packet_sha256"] = _content_sha256(packet)
    if args.manual_labels:
        raw_labels = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.manual_labels
        ]
        comparison["manual_adjudications"] = [
            apply_manual_labels(comparison, labels) for labels in raw_labels
        ]
        changed_ids = {
            case["case_id"] for case in comparison["cases"] if case["outputs_changed"]
        }
        comparison["manual_agreement"] = manual_agreement(raw_labels, changed_ids)
    args.json_output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(
        render_markdown(comparison, source_hash), encoding="utf-8"
    )
    args.adjudication_output.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    args.adjudication_key_output.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
