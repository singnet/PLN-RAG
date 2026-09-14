from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 0.8
_TOKEN_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalized_tokens(value: Any) -> list[str]:
    """Normalize proof atoms and labels across CamelCase, snake_case, and hyphens."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = _DIGIT_BOUNDARY.sub(" ", _TOKEN_BOUNDARY.sub(r"\1 \2", text)).lower()
    return [token for token in _NON_ALNUM.split(text) if token]


def _contains_alias(tokens: list[str], alias: str) -> bool:
    wanted = _normalized_tokens(alias)
    if not wanted or len(wanted) > len(tokens):
        return False
    width = len(wanted)
    return any(tokens[index : index + width] == wanted for index in range(len(tokens) - width + 1))


def _match_groups(tokens: list[str], groups: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    matched = []
    missing = []
    for group in groups:
        group_id = str(group["id"])
        alias = next((item for item in group.get("aliases", []) if _contains_alias(tokens, item)), None)
        if alias is None:
            missing.append(group_id)
        else:
            matched.append({"id": group_id, "alias": alias})
    return matched, missing


def _proof_value(row: dict[str, Any]) -> Any:
    query = (row.get("end_to_end") or {}).get("query") or {}
    return query.get("proof", row.get("proof"))


def _proof_found(proof: Any) -> bool:
    if proof is None or proof is False:
        return False
    if isinstance(proof, str):
        return bool(proof.strip()) and proof.strip() != "[]"
    return bool(proof)


def _provenance_value(row: dict[str, Any]) -> Any:
    if "proof_provenance" in row:
        return row["proof_provenance"]
    query = (row.get("end_to_end") or {}).get("query") or {}
    return query.get("proof_provenance")


def _has_grounded_provenance(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("current_case_grounded") is True:
            return True
        return any(_has_grounded_provenance(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_grounded_provenance(item) for item in value)
    return False


def _proof_items(proof: Any) -> list[Any]:
    if isinstance(proof, list):
        return proof
    if isinstance(proof, str) and proof.strip().startswith("["):
        try:
            parsed = ast.literal_eval(proof)
        except (SyntaxError, ValueError):
            return [proof]
        if isinstance(parsed, list):
            return parsed
    return [proof]


def _grounded_proof_items(proof: Any, provenance: Any) -> list[Any]:
    proofs = _proof_items(proof)
    if isinstance(provenance, list):
        return [
            item
            for index, item in enumerate(proofs)
            if index < len(provenance)
            and _has_grounded_provenance(provenance[index])
        ]
    return proofs if _has_grounded_provenance(provenance) else []


def score_row(
    row: dict[str, Any], label: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("score threshold must be between 0 and 1")

    proof = _proof_value(row)
    provenance = _provenance_value(row)
    proof_found = _proof_found(proof)
    # Score only traces grounded in the current case. Metadata and foreign proof
    # text cannot contribute entities or semantics to a passing result.
    grounded_proofs = _grounded_proof_items(proof, provenance)
    tokens = _normalized_tokens(grounded_proofs)
    all_proof_tokens = _normalized_tokens(proof)

    entity_groups = label.get("required_entity_groups") or []
    semantic_groups = label.get("required_semantic_groups") or []
    forbidden_groups = label.get("forbidden_groups") or []
    matched_entities, missing_entities = _match_groups(tokens, entity_groups)
    matched_semantics, missing_semantics = _match_groups(tokens, semantic_groups)
    matched_forbidden, _ = _match_groups(all_proof_tokens, forbidden_groups)

    entity_coverage = len(matched_entities) / len(entity_groups) if entity_groups else 1.0
    semantic_coverage = len(matched_semantics) / len(semantic_groups) if semantic_groups else 1.0
    score = round((entity_coverage + semantic_coverage) / 2, 6)
    minimum_entities = int(label.get("minimum_entity_groups", len(entity_groups)))
    minimum_semantics = int(label.get("minimum_semantic_groups", len(semantic_groups)))
    grounded = _has_grounded_provenance(provenance)

    reasons = []
    if not proof_found:
        reasons.append("proof_not_found")
    if len(matched_entities) < minimum_entities:
        reasons.append("insufficient_entity_groups")
    if len(matched_semantics) < minimum_semantics:
        reasons.append("insufficient_semantic_groups")
    if not grounded:
        reasons.append("provenance_not_current_case_grounded")
    if matched_forbidden:
        reasons.append("forbidden_group_hit")
    if score < threshold:
        reasons.append("score_below_threshold")

    return {
        "case_id": str(label["case_id"]),
        "case_hash": str(label["case_hash"]),
        "passed": not reasons,
        "proof_found": proof_found,
        "current_case_grounded": grounded,
        "grounded_proof_count": len(grounded_proofs),
        "score": score,
        "score_threshold": threshold,
        "coverage": {
            "entity": round(entity_coverage, 6),
            "semantic": round(semantic_coverage, 6),
        },
        "minimum_groups": {"entity": minimum_entities, "semantic": minimum_semantics},
        "matched_entity_groups": matched_entities,
        "missing_entity_groups": missing_entities,
        "matched_semantic_groups": matched_semantics,
        "missing_semantic_groups": missing_semantics,
        "matched_forbidden_groups": matched_forbidden,
        "failure_reasons": reasons,
    }


def _suite_ids(report: dict[str, Any]) -> set[str]:
    values = [report.get("suite_name")]
    for field in ("suite", "suite_metadata"):
        value = report.get(field)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            values.extend((value.get("suite_id"), value.get("suite"), value.get("name")))
    return {str(value) for value in values if value}


def _labels_by_id(labels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for label in labels.get("cases") or []:
        case_id = str(label.get("case_id") or "")
        if not case_id or case_id in indexed:
            raise ValueError(f"invalid or duplicate label case ID: {case_id!r}")
        if not label.get("case_hash"):
            raise ValueError(f"label case hash missing for {case_id}")
        indexed[case_id] = label
    if not indexed:
        raise ValueError("labels contain no cases")
    return indexed


def _validate_report(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes = [report.get("mode"), (report.get("config") or {}).get("mode")]
    if not any(mode is not None for mode in modes) or any(
        mode is not None and mode != "isolated" for mode in modes
    ):
        raise ValueError("benchmark report must use isolated mode")

    expected_suite = str((labels.get("benchmark_reference") or {}).get("suite_id") or "")
    if not expected_suite or expected_suite not in _suite_ids(report):
        raise ValueError(f"benchmark report suite does not match labels ({expected_suite})")

    parsers = report.get("parsers")
    if not isinstance(parsers, dict) or not parsers:
        raise ValueError("benchmark report contains no parsers")
    return _labels_by_id(labels)


def evaluate_report(
    report: dict[str, Any], labels: dict[str, Any], threshold: float | None = None
) -> dict[str, Any]:
    indexed_labels = _validate_report(report, labels)
    selected_threshold = (
        float(labels.get("default_score_threshold", DEFAULT_THRESHOLD))
        if threshold is None
        else float(threshold)
    )
    if not 0 <= selected_threshold <= 1:
        raise ValueError("score threshold must be between 0 and 1")

    parser_results = {}
    total_cases = 0
    total_passed = 0
    for parser_name, rows in report["parsers"].items():
        seen = set()
        cases = []
        for row in rows:
            embedded = row.get("case") or {}
            case_id = str(
                row.get("case_id")
                or embedded.get("case_id")
                or embedded.get("id")
                or embedded.get("name")
                or ""
            )
            if case_id not in indexed_labels:
                continue
            if case_id in seen:
                raise ValueError(f"duplicate parser/case ID: {parser_name}/{case_id}")
            seen.add(case_id)
            expected_hash = str(indexed_labels[case_id]["case_hash"])
            actual_hash = str(row.get("case_hash") or "")
            if actual_hash != expected_hash:
                raise ValueError(f"label/case hash mismatch for {parser_name}/{case_id}")
            cases.append(score_row(row, indexed_labels[case_id], selected_threshold))

        passed = sum(1 for case in cases if case["passed"])
        average = round(sum(case["score"] for case in cases) / len(cases), 6) if cases else None
        parser_results[str(parser_name)] = {
            "summary": {
                "cases_evaluated": len(cases),
                "passed": passed,
                "failed": len(cases) - passed,
                "pass_rate": round(passed / len(cases), 6) if cases else None,
                "average_score": average,
            },
            "cases": cases,
        }
        total_cases += len(cases)
        total_passed += passed

    if total_cases == 0:
        raise ValueError("benchmark report contains no cases present in labels")
    return {
        "schema_version": 1,
        "evaluation": "labels_only_proof_quality",
        "label_set": labels.get("label_set"),
        "benchmark_suite": (labels.get("benchmark_reference") or {}).get("suite_id"),
        "benchmark_run_id": report.get("run_id"),
        "mode": "isolated",
        "score_threshold": selected_threshold,
        "summary": {
            "parsers": len(parser_results),
            "cases_evaluated": total_cases,
            "passed": total_passed,
            "failed": total_cases - total_passed,
            "pass_rate": round(total_passed / total_cases, 6),
        },
        "parsers": parser_results,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score isolated benchmark proofs with Stress7 labels.")
    parser.add_argument("report", help="Isolated parser benchmark report JSON")
    parser.add_argument("--labels", required=True, help="Proof-quality labels sidecar JSON")
    parser.add_argument("--threshold", type=float, help="Override the labels score threshold")
    parser.add_argument("--output", help="Write evaluation JSON instead of printing it")
    args = parser.parse_args(argv)
    try:
        result = evaluate_report(
            _load_json(Path(args.report)), _load_json(Path(args.labels)), args.threshold
        )
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
