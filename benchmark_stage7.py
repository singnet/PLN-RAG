"""Deterministic three-arm benchmark for temporal/counterfactual SENF reasoning."""

import argparse
import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from config import get_settings
from core.parser import ParseResult
from core.service import PLNRAGService
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


ARMS = {
    "canonical_pln": (CanonicalPLNParser, False),
    "canonical_senf_pln_stage6": (CanonicalSENFPLNParser, False),
    "canonical_senf_pln_stage7": (CanonicalSENFPLNParser, True),
}
SETTINGS_OVERRIDES = {
    "CONCEPTNET_ENABLED": "false",
    "CONCEPTNET_INDEX_ON_STARTUP": "false",
    "CONCEPTNET_AUTOLOAD": "false",
    "CONCEPTNET_AUTO_REBUILD_ON_CHANGE": "false",
    "ANSWER_GENERATION_ENABLED": "false",
    "SOURCE_LOOKUP_MAX_ATOMS": "0",
    "QUERY_FALLBACK_ENABLED": "true",
    "QUERY_CANDIDATE_MAX_TRIES": "5",
    "CHUNK_SIZE": "512",
    "CHUNK_OVERLAP": "64",
    "CONTEXT_TOP_K": "10",
    "PARSER_BATCH_SENTENCES": "4",
    "PARSER_BATCH_MAX_CHARS": "2000",
    "CHAINING_TIMEOUT": "30",
    "CHAINING_MAX_STEPS": "100",
    "SENF_IDENTITY_THRESHOLD": "0.75",
    "SENF_CONTEXT_TOP_K": "10",
    "SENF_SESSION_MAX_FRAMES": "200",
    "SENF_USE_VECTOR_CONTEXT": "false",
    "SENF_EXEMPLAR_ENABLED": "true",
    "SENF_EMIT_BRIDGE_ATOMS": "false",
    "SENF_WEAVE_TOP_K": "3",
    "SENF_QUERY_MAX_PRIORS": "16",
    "SENF_QUERY_MAX_SOURCE_FRAMES": "128",
    "SENF_QUERY_MAX_MENTIONS": "256",
    "SENF_QUERY_MAX_CANDIDATE_WORK": "32",
    "SENF_WEAVE_PER_SOURCE_K": "3",
    "SENF_WEAVE_BEAM_WIDTH": "32",
    "SENF_WEAVE_MAX_FRAMES": "64",
    "SENF_WEAVE_MAX_PAIR_CANDIDATES": "256",
    "SENF_WEAVE_MAX_EXEMPLAR_ALTERNATIVES": "4",
    "SENF_WEAVE_MAX_COST": "2.0",
    "SENF_WEAVE_ENGINE": "beam",
    "SENF_WEAVE_GLOBAL_CANDIDATE_CAP": "512",
    "SENF_WEAVE_MAX_CELLS": "65536",
    "SENF_WEAVE_MAX_SEEDS": "32",
    "SENF_WEAVE_MAX_ITERATIONS": "200",
    "SENF_WEAVE_SINKHORN_TOLERANCE": "0.0000001",
    "SENF_WEAVE_SINKHORN_REGULARIZATION": "0.25",
    "SENF_WEAVE_FORGET_FINE_COSTS": "false",
    "SENF_WEAVE_COARSE_IDENTITY": "false",
    "SENF_WEAVE_MAX_CONFLICT_COST": "1000000.0",
    "SENF_WEAVE_MAX_TRANSPORT_COST": "1000000.0",
    "SENF_WEAVE_REQUIRE_CONTEXT_MATCH": "false",
    "SENF_SOURCE_GROUNDING_WEIGHT": "3",
    "SENF_ROLE_COMPAT_WEIGHT": "2",
    "SENF_DISTORTION_WEIGHT": "0",
    "SENF_IDENTITY_SUPPORT_WEIGHT": "2",
    "SENF_EXEMPLAR_COHERENCE_WEIGHT": "2",
    "SENF_CONFLICT_WEIGHT": "3",
    "SENF_TRANSPORT_COST_WEIGHT": "2",
    "SENF_MATCHED_SOFT_MASS_WEIGHT": "0",
    "SENF_GLOBAL_RESIDUAL_RATIO_WEIGHT": "0",
    "SENF_ALIGNMENT_CONFIDENCE_WEIGHT": "0",
    "SENF_DIAGNOSTICS_MAX_WEAVES": "3",
    "SENF_DIAGNOSTICS_MAX_ITEMS": "128",
    "SENF_DIAGNOSTICS_MAX_EVIDENCE": "16",
    "SENF_BRANCH_MAX_NODES": "64",
    "SENF_BRANCH_MAX_DEPTH": "16",
    "SENF_BRANCH_MAX_THEORY_STATEMENTS": "128",
    "SENF_TEMPORAL_DECAY_RATE": "0.01",
}


class FixtureGenerationBackend:
    """Return reviewed generated atoms while exercising the production parser path."""

    fail_closed = True

    def __init__(self, case: dict[str, Any]):
        self._query = " ".join(str(case["user_query"]).split())
        self._statements = list(case["statement_atoms"])
        self._queries = list(case["query_atoms"])
        self.calls: list[str] = []

    def generate(self, *, sentences, context, pln_spec, live) -> ParseResult:
        text = " ".join(" ".join(str(item).split()) for item in sentences)
        is_query = text == self._query
        self.calls.append("query" if is_query else "statement")
        if is_query:
            return ParseResult(queries=list(self._queries))
        return ParseResult(statements=list(self._statements))


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Stage 7 suite must contain a non-empty cases list")
    seen: set[str] = set()
    required = {
        "case_id",
        "name",
        "category",
        "input_text",
        "user_query",
        "statement_atoms",
        "query_atoms",
        "expected_proof",
    }
    for case in cases:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Stage 7 case is missing fields: {', '.join(missing)}")
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate Stage 7 case_id: {case_id}")
        seen.add(case_id)
        if case["category"] not in {"authorized_positive", "safety_negative"}:
            raise ValueError(f"Invalid Stage 7 category for {case_id}")
        if not isinstance(case["expected_proof"], bool):
            raise ValueError(f"Stage 7 expected_proof must be boolean for {case_id}")
        if not all(
            isinstance(case[field], str) and case[field].strip()
            for field in ("name", "input_text", "user_query")
        ):
            raise ValueError(f"Stage 7 text fields must be non-empty for {case_id}")
        if not all(
            isinstance(case[field], list)
            and case[field]
            and all(isinstance(atom, str) and atom.strip() for atom in case[field])
            for field in ("statement_atoms", "query_atoms")
        ):
            raise ValueError(f"Stage 7 atom fields must be non-empty string lists for {case_id}")
        expected_category = (
            "authorized_positive" if case["expected_proof"] else "safety_negative"
        )
        if case["category"] != expected_category:
            raise ValueError(f"Stage 7 category/proof mismatch for {case_id}")
        expectation = case.get("stage7_expectation")
        if expectation is not None and not isinstance(expectation, dict):
            raise ValueError(f"Stage 7 expectation must be an object for {case_id}")
    return payload


def _configure_arm(arm: str, case_id: str, run_id: str) -> Path:
    _, enabled = ARMS[arm]
    slug = f"stage7_{arm}_{case_id}_{run_id}".replace("-", "_")
    atomspace = Path("/tmp") / f"{slug}.metta"
    os.environ.update({
        **SETTINGS_OVERRIDES,
        "QDRANT_COLLECTION": slug,
        "ATOMSPACE_PATH": str(atomspace),
        "SENF_COUNTERFACTUAL_ENABLED": "true" if enabled else "false",
    })
    get_settings.cache_clear()
    return atomspace


def _proof_found(proof: str) -> bool:
    return bool(proof and proof != "[]")


def _executed_plan(diagnostics: dict[str, Any]) -> dict[str, Any]:
    plan = diagnostics.get("executed_temporal_plan")
    return plan if isinstance(plan, dict) else {}


def _stage7_checks(case: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    expected = case.get("stage7_expectation") or {}
    diagnostics = result.get("senf") or {}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual: Any, wanted: Any) -> None:
        checks.append({"name": name, "passed": passed, "actual": actual, "expected": wanted})

    minimum = expected.get("minimum_theory_statements")
    if minimum is not None:
        actual = int(diagnostics.get("counterfactual_theory_statement_count", 0) or 0)
        add("minimum_theory_statements", actual >= int(minimum), actual, minimum)

    if expected.get("stage7_rejection"):
        rejection = diagnostics.get("stage7_rejection")
        rejected = result.get("query_status") == "no_query" and bool(rejection)
        add("stage7_rejection", rejected, rejection, "structured rejection")

    plan = _executed_plan(diagnostics)
    decisions = [item for item in plan.get("decisions", []) if isinstance(item, dict)]
    for field in ("branch_id", "validity_interval_id"):
        if field in expected:
            add(field, plan.get(field) == expected[field], plan.get(field), expected[field])
    for field in ("branch_probability", "temporal_decay"):
        if field not in expected:
            continue
        actual_values = [item.get(field) for item in decisions if item.get("allowed")]
        wanted = float(expected[field])
        matched = any(
            isinstance(value, (int, float)) and math.isclose(value, wanted, abs_tol=1e-8)
            for value in actual_values
        )
        add(field, matched, actual_values, wanted)

    denial = expected.get("denial_reason")
    if denial:
        reasons = [item.get("reason") for item in decisions if not item.get("allowed")]
        add("denial_reason", denial in reasons, reasons, denial)
    return checks


async def _run_case(arm: str, case: dict[str, Any], run_id: str) -> dict[str, Any]:
    atomspace = _configure_arm(arm, str(case["case_id"]), run_id)
    parser_class, enabled = ARMS[arm]
    backend = FixtureGenerationBackend(case)
    service = None
    started = time.perf_counter()
    try:
        parser = parser_class()
        parser.set_generation_backend(backend)
        service = PLNRAGService(parser)
        service.reset("all")
        ingest = await service.ingest_batch([str(case["input_text"])])
        response = await service.reason(str(case["user_query"]))
        found = _proof_found(response.proof)
        ingested_atom_count = sum(len(item.atoms) for item in ingest)
        rejected_atom_count = sum(item.rejected_count for item in ingest)
        ingest_ok = (
            bool(ingest)
            and all(item.status == "success" for item in ingest)
            and rejected_atom_count == 0
            and ingested_atom_count == len(case["statement_atoms"])
        )
        expected_rejection = bool(
            enabled and (case.get("stage7_expectation") or {}).get("stage7_rejection")
        )
        query_executed = bool(response.executed_query) and response.query_status not in {
            "error", "no_query"
        }
        structured_rejection = bool((response.senf or {}).get("stage7_rejection"))
        execution_valid = ingest_ok and (
            response.query_status == "no_query" and structured_rejection
            if expected_rejection else query_executed
        )
        result = {
            "case_id": case["case_id"],
            "name": case["name"],
            "category": case["category"],
            "expected_proof": bool(case["expected_proof"]),
            "proof_found": found,
            "correct": found == bool(case["expected_proof"]) and execution_valid,
            "execution_valid": execution_valid,
            "query_status": response.query_status,
            "original_query": response.original_query,
            "executed_query": response.executed_query,
            "proof": response.proof,
            "senf": response.senf or {},
            "ingest_status": [item.status for item in ingest],
            "expected_ingested_atom_count": len(case["statement_atoms"]),
            "ingested_atom_count": ingested_atom_count,
            "rejected_atom_count": rejected_atom_count,
            "generation_calls": list(backend.calls),
            "counterfactual_enabled": enabled,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
        result["semantic_checks"] = (
            _stage7_checks(case, result) if arm == "canonical_senf_pln_stage7" else []
        )
        return result
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "name": case["name"],
            "category": case["category"],
            "expected_proof": bool(case["expected_proof"]),
            "proof_found": False,
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "generation_calls": list(backend.calls),
            "counterfactual_enabled": enabled,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "semantic_checks": [],
        }
    finally:
        if service is not None:
            service.reset("all")
        if atomspace.exists():
            atomspace.unlink()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [item for item in results if item["category"] == "authorized_positive"]
    negatives = [item for item in results if item["category"] == "safety_negative"]
    checks = [check for item in results for check in item.get("semantic_checks", [])]
    return {
        "cases": len(results),
        "correct": sum(bool(item.get("correct")) for item in results),
        "proof_found": sum(bool(item.get("proof_found")) for item in results),
        "authorized_positive_correct": sum(bool(item.get("correct")) for item in positives),
        "authorized_positive_cases": len(positives),
        "safety_negative_correct": sum(bool(item.get("correct")) for item in negatives),
        "safety_negative_cases": len(negatives),
        "safety_false_positives": sum(bool(item.get("proof_found")) for item in negatives),
        "semantic_checks_passed": sum(bool(item.get("passed")) for item in checks),
        "semantic_checks_total": len(checks),
        "errors": sum(1 for item in results if item.get("error")),
        "invalid_executions": sum(
            1 for item in results if not item.get("execution_valid", False)
        ),
        "average_latency_seconds": round(
            sum(float(item.get("elapsed_seconds", 0.0)) for item in results) / len(results), 4
        ) if results else 0.0,
    }


async def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--suite-file",
        default="data/benchmarks/stage7_counterfactual_v1.json",
    )
    cli.add_argument("--output-dir", default="data/benchmarks")
    cli.add_argument("--progress", action="store_true")
    args = cli.parse_args()

    suite_path = Path(args.suite_file)
    suite = load_suite(suite_path)
    run_id = uuid.uuid4().hex[:8]
    environment_keys = set(SETTINGS_OVERRIDES) | {
        "QDRANT_COLLECTION", "ATOMSPACE_PATH", "SENF_COUNTERFACTUAL_ENABLED"
    }
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    arm_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for arm in ARMS:
            results = []
            for index, case in enumerate(suite["cases"], start=1):
                if args.progress:
                    print(f"[{arm}] {index}/{len(suite['cases'])} {case['case_id']}", flush=True)
                results.append(await _run_case(arm, case, run_id))
            arm_results[arm] = results
            summaries[arm] = summarize(results)
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    stage6 = summaries["canonical_senf_pln_stage6"]
    stage7 = summaries["canonical_senf_pln_stage7"]
    payload = {
        "benchmark_schema_version": 1,
        "run_id": run_id,
        "suite": suite.get("suite_id", suite_path.stem),
        "suite_version": suite.get("version"),
        "suite_file": str(suite_path),
        "generation_policy": "reviewed deterministic MeTTa fixtures",
        "scope": "production parser post-processing, storage, planning, and reasoning; LLM generation is fixed",
        "settings": {
            key.lower(): value for key, value in SETTINGS_OVERRIDES.items()
        },
        "arms": {
            name: {
                "parser": parser.__name__,
                "senf_counterfactual_enabled": enabled,
            }
            for name, (parser, enabled) in ARMS.items()
        },
        "results": arm_results,
        "summary": summaries,
        "stage7_delta_vs_stage6": {
            "correct": stage7["correct"] - stage6["correct"],
            "authorized_positive_correct": (
                stage7["authorized_positive_correct"]
                - stage6["authorized_positive_correct"]
            ),
            "safety_false_positives": (
                stage7["safety_false_positives"]
                - stage6["safety_false_positives"]
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"stage7_comparison_{run_id}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summaries, "delta": payload["stage7_delta_vs_stage6"]}, indent=2))

    stage7_checks_ok = stage7["semantic_checks_passed"] == stage7["semantic_checks_total"]
    stage7_results_ok = stage7["correct"] == stage7["cases"]
    return 0 if stage7["errors"] == 0 and stage7_checks_ok and stage7_results_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
