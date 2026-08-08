import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import benchmark_grading as bg
from config import get_settings
from core.service import PLNRAGService


SMOKE_CASES = [
    {
        "name": "dog-animal-yesno",
        "category": "isa",
        "texts": ["Dogs are animals.", "Fido is a dog."],
        "question": "Is Fido an animal?",
        "expected_proof": True,
    },
    {
        "name": "human-mortal-yesno",
        "category": "isa",
        "texts": ["Humans are mortal.", "Socrates is a human."],
        "question": "Is Socrates mortal?",
        "expected_proof": True,
    },
    {
        "name": "fish-smart-yesno",
        "category": "rule",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Is Kebede smart?",
        "expected_proof": True,
    },
    {
        "name": "teacher-educates-yesno",
        "category": "rule",
        "texts": ["Teachers educate students.", "Marta is a teacher."],
        "question": "Does Marta educate students?",
        "expected_proof": True,
    },
    {
        "name": "programmer-solves-yesno",
        "category": "rule",
        "texts": ["Programmers solve problems.", "Bekele is a programmer."],
        "question": "Does Bekele solve problems?",
        "expected_proof": True,
    },
    {
        "name": "who-is-smart-open",
        "category": "open",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Who is smart?",
        "expected_proof": True,
    },
    {
        "name": "what-does-kebede-eat-open",
        "category": "open",
        "texts": ["Kebede eats fish."],
        "question": "What does Kebede eat?",
        "expected_proof": True,
    },
    {
        "name": "hasa-dog-nose",
        "category": "relation",
        "texts": ["Dogs have noses.", "Fido is a dog."],
        "question": "Does Fido have a nose?",
        "expected_proof": True,
    },
    {
        "name": "usedfor-soap-cleaning",
        "category": "relation",
        "texts": ["Soap is used for cleaning."],
        "question": "Is soap used for cleaning?",
        "expected_proof": True,
    },
    {
        "name": "capableof-dog-drink-water",
        "category": "relation",
        "texts": ["Dogs are capable of drinking water."],
        "question": "Is dog capable of drinking water?",
        "expected_proof": True,
    },
    {
        "name": "partof-automobile-horn-car",
        "category": "relation",
        "texts": ["An automobile horn is part of a car."],
        "question": "Is automobile horn part of car?",
        "expected_proof": True,
    },
    {
        "name": "plural-dolphins-mammals",
        "category": "normalization",
        "texts": ["Dolphins are mammals."],
        "question": "Are dolphins mammals?",
        "expected_proof": True,
    },
    {
        "name": "negative-missing-fido",
        "category": "negative",
        "texts": ["Dogs are animals."],
        "question": "Is Fido an animal?",
        "expected_proof": False,
    },
    {
        "name": "negative-unrelated-query",
        "category": "negative",
        "texts": ["Cats are animals."],
        "question": "Is soap used for cleaning?",
        "expected_proof": False,
    },
]

SMOKE_QUICK_CASE_NAMES = {
    "dog-animal-yesno",
    "fish-smart-yesno",
    "what-does-kebede-eat-open",
    "usedfor-soap-cleaning",
    "negative-missing-fido",
}

CASE_FILES = {
    "entailmentbank": Path("data/benchmarks/cases/entailmentbank_actual_curated.json"),
    "abstracts": Path("data/benchmarks/cases/abstract_curated_candidates.json"),
}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}

ACTIVE_PARSERS = ("nl2pln", "canonical_pln")
AVAILABLE_PARSERS = (
    "nl2pln",
    "canonical_pln",
    "langextract",
    "canonical_langextract",
    "canonical_pln_1686527",
    "canonical_pln_d8d39afd",
)


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _normalize_loaded_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    # Support both legacy benchmark schema (premises/texts + question)
    # and stress-suite schema (input_text + user_query).
    if "question" not in normalized and "user_query" in normalized:
        normalized["question"] = str(normalized["user_query"])
    normalized.setdefault(
        "name",
        normalized.get("id")
        or normalized.get("case_id")
        or _slugify(normalized.get("question") or "case"),
    )
    if "texts" not in normalized:
        if "input_text" in normalized:
            normalized["texts"] = [str(normalized.get("input_text") or "")]
        else:
            premises = normalized.get("premises", [])
            if not isinstance(premises, list):
                raise ValueError(f"Case {normalized['name']} has invalid premises field")
            normalized["texts"] = [str(item) for item in premises]
    normalized.setdefault("category", "external")
    # expected_proof is optional for stress suites.
    if "expected_proof" not in normalized:
        normalized["expected_proof"] = None
    elif normalized["expected_proof"] is not None:
        normalized["expected_proof"] = _is_truthy(normalized["expected_proof"])
    return normalized


def _load_case_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"] = [_normalize_loaded_case(case) for case in payload.get("cases", [])]
    return payload


def _select_cases_from_file(path: Path, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_case_file(path)
    cases = payload["cases"][:3] if quick else payload["cases"]
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    # Normalize suite metadata keys so output stays consistent.
    if "suite" not in metadata:
        metadata["suite"] = metadata.get("suite_id") or path.stem
    metadata.setdefault("status", "external")
    metadata.setdefault("source_type", "custom")
    metadata.setdefault("intended_use", "end_to_end_usefulness")
    metadata.setdefault("suite_path", str(path))
    return metadata, cases


def _resolve_gold(
    explicit: str | None, suite_metadata: dict[str, Any], suite_label: str
) -> tuple[dict[str, Any], str | None]:
    """Locate the gold file: explicit path, else a sibling `<suite>_gold.json`.

    A missing sibling is normal; a missing explicit `--gold-file` is an error, since the caller
    asked for grading and would otherwise get a silently ungraded run.
    """
    if explicit:
        path = Path(explicit)
    else:
        suite_path = suite_metadata.get("suite_path")
        directory = Path(suite_path).parent if suite_path else Path("data/benchmarks")
        path = directory / f"{suite_label}_gold.json"
        if not path.exists():
            return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"{path}: expected a 'cases' object")
    return cases, str(path)


def _apply_gold_expectations(cases: list[dict[str, Any]], gold_cases: dict[str, Any]) -> None:
    """Fill in `expected_proof` from gold where the suite does not state one."""
    for case in cases:
        if case.get("expected_proof") is not None:
            continue
        gold = gold_cases.get(case.get("case_id") or case.get("id") or case.get("name"))
        if gold and gold.get("expected_proof") is not None:
            case["expected_proof"] = bool(gold["expected_proof"])


def _apply_grading(results: list[dict[str, Any]], gold_cases: dict[str, Any]) -> None:
    """Merge answer grades into each result in place, before summarizing.

    A benchmark costs ~28 minutes per parser, so a gold or artifact defect is recorded on the
    case rather than raised and losing the run.
    """
    if not gold_cases:
        return
    for result in results:
        try:
            row = bg.grade_results(gold_cases, [result])[0]
        except bg.ArtifactSchemaError as exc:
            result["answer_correct"] = None
            result["answer_score"] = None
            result["answer_reason"] = f"ungradable: {exc}"
            result["verdict_gradable"] = False
            result["matched_entities"] = []
            continue
        for field in ("answer_correct", "answer_score", "answer_reason",
                      "verdict_gradable", "matched_entities"):
            result[field] = row[field]


def _select_cases(suite: str, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if suite == "smoke":
        cases = [
            case for case in SMOKE_CASES
            if not quick or case["name"] in SMOKE_QUICK_CASE_NAMES
        ]
        return (
            {
                "suite": "smoke",
                "status": "built_in",
                "source_type": "internal_smoke",
                "intended_use": "sanity_check",
            },
            cases,
        )

    if suite == "combined":
        entailmentbank = _load_case_file(CASE_FILES["entailmentbank"])
        abstracts = _load_case_file(CASE_FILES["abstracts"])
        cases = entailmentbank["cases"] + abstracts["cases"]
        if quick:
            cases = cases[:6]
        return (
            {
                "suite": "combined",
                "status": "composed",
                "source_type": "multi_suite",
                "members": [entailmentbank["suite"], abstracts["suite"]],
                "intended_use": "end_to_end_usefulness",
            },
            cases,
        )

    if suite not in CASE_FILES:
        raise ValueError(f"Unsupported suite '{suite}'")

    payload = _load_case_file(CASE_FILES[suite])
    cases = payload["cases"][:3] if quick else payload["cases"]
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    return metadata, cases


def _get_parser_factory(name: str):
    if name == "nl2pln":
        from parsers.nl2pln_parser import NL2PLNParser

        return NL2PLNParser
    if name == "canonical_pln":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return CanonicalPLNParser
    if name == "langextract":
        from parsers.langextract_pln_parser import LangExtractPLNParser

        return LangExtractPLNParser
    if name == "canonical_langextract":
        from parsers.canonical_langextract_parser import CanonicalLangExtractParser

        return CanonicalLangExtractParser
    if name == "canonical_pln_1686527":
        from parsers.canonical_pln_1686527_parser import CanonicalPLN1686527Parser

        return CanonicalPLN1686527Parser
    if name == "canonical_pln_d8d39afd":
        from parsers.canonical_pln_d8d39afd_parser import CanonicalPLND8D39AFDParser

        return CanonicalPLND8D39AFDParser
    if name == "manhin":
        from parsers.manhin_parser import ManhinParser

        return ManhinParser
    raise ValueError(f"Unsupported parser '{name}'")


def _preflight_llm() -> tuple[bool, str]:
    """Make one real LLM call before spending an hour on a run that cannot work.

    Deliberately goes through dspy with the same model and key the parsers use,
    rather than probing a hardcoded endpoint: a relay, a custom base_url or a
    permissions-scoped key all behave differently on a synthetic probe than on
    the actual completion path.
    """
    cfg = get_settings()
    try:
        import dspy

        lm = dspy.LM(cfg.openai_model, api_key=cfg.openai_api_key, cache=False)
        reply = lm("Reply with the single word: ok", max_tokens=5)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not reply:
        return False, "LLM returned an empty response"
    return True, ""


def _preflight_reasoner() -> tuple[bool, str]:
    """Check backward chaining still works before blaming a parser for zero proofs.

    Reasoner.query catches every exception and returns [], so a broken chainer
    yields a clean exit and proof_found=0 for every case. That is not
    distinguishable from a hard suite by the summary alone, since a low proof
    count is a legitimate isolated-mode result. Uses a throwaway chainer so the
    persistent atomspace is untouched.
    """
    try:
        from pettachainer import PeTTaChainer

        chainer = PeTTaChainer()
        for atom in (
            "(: edge_ab (Edge A B) (STV 1.0 1.0))",
            "(: edge_bc (Edge B C) (STV 1.0 1.0))",
            "(: edge_to_path (Implication (Premises (Edge $x $y)) "
            "(Conclusions (Path $x $y))) (CTV (STV 1.0 1.0) (STV 0.0 1.0)))",
            "(: path_step (Implication (Premises (Path $x $y) (Edge $y $z)) "
            "(Conclusions (Path $x $z))) (CTV (STV 1.0 1.0) (STV 0.0 1.0)))",
        ):
            chainer.add_atom(atom)
        seeds = chainer.select_facts(["(Edge A B)", "(Edge B C)"])
        chainer.forward_chain(seeds, steps=50)
        proofs = chainer.query("(: $prf (Path A C) $tv)", steps=10, timeout_sec=60)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not proofs:
        return False, "no proof returned for a known-derivable fact (Path A C)"
    return True, ""


def _run_parse(parser: object, text: str, context: list[str], is_query: bool):
    if is_query and hasattr(parser, "parse_query"):
        return parser.parse_query(text, context)
    return parser.parse(text, context)


def _proof_found(proof: str) -> bool:
    return bool(proof and proof != "[]")


def _configure_case_environment(parser_name: str, case_name: str, run_id: str):
    slug = f"bench_{parser_name}_{case_name}_{run_id}".replace("-", "_")
    os.environ["QDRANT_COLLECTION"] = slug
    os.environ["ATOMSPACE_PATH"] = f"data/atomspace/{slug}.metta"
    os.environ["CONCEPTNET_ENABLED"] = "false"
    os.environ["CONCEPTNET_INDEX_ON_STARTUP"] = "false"
    os.environ["CONCEPTNET_AUTOLOAD"] = "false"
    os.environ["CONCEPTNET_AUTO_REBUILD_ON_CHANGE"] = "false"
    get_settings.cache_clear()
    return slug, Path(os.environ["ATOMSPACE_PATH"])


def _configure_suite_environment(parser_name: str, suite_name: str, run_id: str):
    slug = f"bench_{parser_name}_{suite_name}_{run_id}".replace("-", "_")
    os.environ["QDRANT_COLLECTION"] = slug
    os.environ["ATOMSPACE_PATH"] = f"data/atomspace/{slug}.metta"
    os.environ["CONCEPTNET_ENABLED"] = "false"
    os.environ["CONCEPTNET_INDEX_ON_STARTUP"] = "false"
    os.environ["CONCEPTNET_AUTOLOAD"] = "false"
    os.environ["CONCEPTNET_AUTO_REBUILD_ON_CHANGE"] = "false"
    get_settings.cache_clear()
    return slug, Path(os.environ["ATOMSPACE_PATH"])


def _compact_ingest_results(results) -> list[dict]:
    return [
        {
            "text": item.text,
            "status": item.status,
            "atoms": item.atoms,
            "error": item.error,
            "rejected_count": getattr(item, "rejected_count", 0),
            "rejected_samples": getattr(item, "rejected_samples", []),
        }
        for item in results
    ]


async def _benchmark_case(parser_name: str, case: dict, run_id: str) -> dict:
    collection, atomspace_path = _configure_case_environment(parser_name, case["name"], run_id)
    parser_factory = _get_parser_factory(parser_name)

    init_started = time.perf_counter()
    parser = parser_factory()
    parser_init_seconds = time.perf_counter() - init_started

    parse_started = time.perf_counter()
    statement_parse = [_run_parse(parser, text, [], is_query=False) for text in case["texts"]]
    query_parse = _run_parse(parser, case["question"], [], is_query=True)
    parse_seconds = time.perf_counter() - parse_started

    service = PLNRAGService(parser)
    service.reset("all")

    try:
        ingest_started = time.perf_counter()
        ingest_results = await service.ingest_batch(case["texts"])
        ingest_seconds = time.perf_counter() - ingest_started

        query_started = time.perf_counter()
        query_response = await service.reason(case["question"])
        query_seconds = time.perf_counter() - query_started

        total_seconds = parser_init_seconds + parse_seconds + ingest_seconds + query_seconds
        found = _proof_found(query_response.proof)
        expected = case.get("expected_proof")
        correct = None if expected is None else (found == expected)

        return {
            "case": case,
            "collection": collection,
            "atomspace_path": str(atomspace_path),
            "timing": {
                "parser_init_seconds": round(parser_init_seconds, 4),
                "parse_only_seconds": round(parse_seconds, 4),
                "ingest_seconds": round(ingest_seconds, 4),
                "query_seconds": round(query_seconds, 4),
                "total_seconds": round(total_seconds, 4),
            },
            "parse_only": {
                "statements": [result.statements for result in statement_parse],
                "statement_counts": [len(result.statements) for result in statement_parse],
                "query_statements": query_parse.statements,
                "query_statements_count": len(query_parse.statements),
                "queries": query_parse.queries,
                "query_count": len(query_parse.queries),
            },
            "end_to_end": {
                "ingest": _compact_ingest_results(ingest_results),
                "query": {
                    "query": query_response.query,
                    "pln_query": query_response.pln_query,
                    "original_query": query_response.original_query,
                    "executed_query": query_response.executed_query,
                    "fallback_used": query_response.fallback_used,
                    "query_status": query_response.query_status,
                    "proof": query_response.proof,
                    "sources": query_response.sources,
                    "answer": query_response.answer,
                    "candidate_count": query_response.candidate_count,
                    "candidate_count_tried": query_response.candidate_count_tried,
                    "executed_candidate_index": query_response.executed_candidate_index,
                    "retry_used": query_response.retry_used,
                    "context_retrieval_seconds": query_response.context_retrieval_seconds,
                    "parse_query_seconds": query_response.parse_query_seconds,
                    "reasoning_seconds": query_response.reasoning_seconds,
                    "source_lookup_seconds": query_response.source_lookup_seconds,
                    "answer_generation_seconds": query_response.answer_generation_seconds,
                },
            },
            "proof_found": found,
            "correct": correct,
        }
    finally:
        service.reset("all")
        if atomspace_path.exists():
            atomspace_path.unlink()


def _knowledge_state(service: PLNRAGService) -> dict[str, int]:
    info = service.health()
    return {
        "atomspace_size": int(info.get("atomspace_size", 0)),
        "background_atomspace_size": int(info.get("background_atomspace_size", 0)),
        "vectordb_count": int(info.get("vectordb_count", 0)),
    }


async def _benchmark_case_with_service(
    parser_name: str,
    case: dict,
    run_id: str,
    collection: str,
    atomspace_path: Path,
    parser: object,
    service: PLNRAGService,
) -> dict:
    parse_started = time.perf_counter()
    statement_parse = [_run_parse(parser, text, [], is_query=False) for text in case["texts"]]
    query_parse = _run_parse(parser, case["question"], [], is_query=True)
    parse_seconds = time.perf_counter() - parse_started

    state_before = _knowledge_state(service)

    ingest_started = time.perf_counter()
    ingest_results = await service.ingest_batch(case["texts"])
    ingest_seconds = time.perf_counter() - ingest_started

    query_started = time.perf_counter()
    query_response = await service.reason(case["question"])
    query_seconds = time.perf_counter() - query_started

    total_seconds = parse_seconds + ingest_seconds + query_seconds
    found = _proof_found(query_response.proof)
    expected = case.get("expected_proof")
    correct = None if expected is None else (found == expected)
    state_after = _knowledge_state(service)

    return {
        "case": case,
        "collection": collection,
        "atomspace_path": str(atomspace_path),
        "timing": {
            "parser_init_seconds": 0.0,
            "parse_only_seconds": round(parse_seconds, 4),
            "ingest_seconds": round(ingest_seconds, 4),
            "query_seconds": round(query_seconds, 4),
            "total_seconds": round(total_seconds, 4),
        },
        "knowledge_state_before_case": state_before,
        "knowledge_state_after_case": state_after,
        "parse_only": {
            "statements": [result.statements for result in statement_parse],
            "statement_counts": [len(result.statements) for result in statement_parse],
            "query_statements": query_parse.statements,
            "query_statements_count": len(query_parse.statements),
            "queries": query_parse.queries,
            "query_count": len(query_parse.queries),
        },
        "end_to_end": {
            "ingest": _compact_ingest_results(ingest_results),
            "query": {
                "query": query_response.query,
                "pln_query": query_response.pln_query,
                "original_query": query_response.original_query,
                "executed_query": query_response.executed_query,
                "fallback_used": query_response.fallback_used,
                "query_status": query_response.query_status,
                "proof": query_response.proof,
                "sources": query_response.sources,
                "answer": query_response.answer,
                "candidate_count": query_response.candidate_count,
                "candidate_count_tried": query_response.candidate_count_tried,
                "executed_candidate_index": query_response.executed_candidate_index,
                "retry_used": query_response.retry_used,
                "context_retrieval_seconds": query_response.context_retrieval_seconds,
                "parse_query_seconds": query_response.parse_query_seconds,
                "reasoning_seconds": query_response.reasoning_seconds,
                "source_lookup_seconds": query_response.source_lookup_seconds,
                "answer_generation_seconds": query_response.answer_generation_seconds,
            },
        },
        "proof_found": found,
        "correct": correct,
    }


async def _benchmark_parser_cumulative(
    parser_name: str,
    cases: list[dict[str, Any]],
    run_id: str,
    suite_name: str,
    *,
    progress: bool = False,
    parser_index: int = 0,
    parser_total: int = 0,
) -> list[dict[str, Any]]:
    collection, atomspace_path = _configure_suite_environment(parser_name, suite_name, run_id)
    parser_factory = _get_parser_factory(parser_name)

    init_started = time.perf_counter()
    parser = parser_factory()
    parser_init_seconds = time.perf_counter() - init_started
    service = PLNRAGService(parser)
    service.reset("all")

    results: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases):
            if progress:
                case_id = case.get("case_id") or case.get("id") or case.get("name")
                label = str(case_id or case.get("name") or "case")
                prefix = (
                    f"[{parser_index}/{parser_total}]" if parser_total else ""
                )
                print(
                    f"{prefix}[{parser_name}] START {index + 1}/{len(cases)} {label}",
                    file=sys.stderr,
                    flush=True,
                )
            result = await _benchmark_case_with_service(
                parser_name,
                case,
                run_id,
                collection,
                atomspace_path,
                parser,
                service,
            )
            if progress:
                q = (result.get("end_to_end") or {}).get("query") or {}
                total_s = (result.get("timing") or {}).get("total_seconds")
                proof = bool(result.get("proof_found"))
                status = q.get("query_status")
                fb = bool(q.get("fallback_used"))
                print(
                    f"{prefix}[{parser_name}] DONE  {index + 1}/{len(cases)} "
                    f"proof_found={proof} status={status} fallback={fb} total_s={total_s}",
                    file=sys.stderr,
                    flush=True,
                )
            if index == 0:
                result["timing"]["parser_init_seconds"] = round(parser_init_seconds, 4)
                result["timing"]["total_seconds"] = round(
                    result["timing"]["total_seconds"] + parser_init_seconds, 4
                )
            results.append(result)
    finally:
        service.reset("all")
        if atomspace_path.exists():
            atomspace_path.unlink()

    return results


def _validity_warnings(summary: dict[str, dict]) -> list[str]:
    """Flag summaries that look like infrastructure failure rather than a result.

    Parsers swallow exceptions and return an empty ParseResult, so a dead LLM
    produces a clean exit code and a report full of zeros. Without this, such a
    run is indistinguishable from a genuine finding.
    """
    warnings: list[str] = []
    for parser_name, stats in summary.items():
        cases = stats.get("cases", 0)
        if not cases:
            continue
        if stats.get("no_query", 0) == cases:
            warnings.append(
                f"{parser_name}: every case produced no_query ({cases}/{cases}). "
                "This is the signature of a failing LLM, not a parser result."
            )
        if stats.get("errors", 0) == cases:
            warnings.append(f"{parser_name}: every case raised ({cases}/{cases}).")
    return warnings


def _summarize_parser(results: list[dict]) -> dict:
    total_cases = len(results)
    correct_known = [result.get("correct") for result in results if result.get("correct") is not None]
    correct = sum(1 for value in correct_known if value)
    proof_found = sum(1 for result in results if result.get("proof_found"))
    no_query = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("query_status") == "no_query"
    )
    weakly_aligned = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("query_status") == "weakly_aligned"
    )
    fallback_used = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("fallback_used")
    )
    latencies = [result.get("timing", {}).get("total_seconds", 0.0) for result in results]
    answer_graded = [result for result in results if result.get("answer_correct") is not None]
    answer_scores = [result.get("answer_score") or 0.0 for result in answer_graded]
    return {
        "cases": total_cases,
        "correct": correct,
        "correct_known": len(correct_known),
        "proof_found": proof_found,
        # `correct` above is proof presence against expectation; these grade the conclusion.
        "answer_correct": sum(1 for result in answer_graded if result["answer_correct"]),
        "answer_graded": len(answer_graded),
        "mean_answer_score": (
            round(sum(answer_scores) / len(answer_scores), 4) if answer_scores else 0.0
        ),
        "verdict_graded": sum(1 for result in answer_graded if result.get("verdict_gradable")),
        "no_query": no_query,
        "weakly_aligned": weakly_aligned,
        "fallback_used": fallback_used,
        "errors": sum(1 for result in results if result.get("error")),
        "avg_latency_seconds": round(sum(latencies) / total_cases, 4) if total_cases else 0.0,
        "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
    }


def _markdown_summary(summary: dict[str, dict]) -> str:
    # `ProofExp` was labelled `Correct`, which read as answer correctness; it has always meant
    # "proof presence matched expectation".
    lines = [
        "| Parser | Cases | ProofExp | AnsCorrect | AnsScore | Proof Found | No Query | Weak Align | Fallback | Avg Latency | Median Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for parser_name, stats in summary.items():
        graded = stats.get("answer_graded", 0)
        answer = f"{stats.get('answer_correct', 0)}/{graded}" if graded else "-"
        lines.append(
            f"| {parser_name} | {stats['cases']} | {stats['correct']} | {answer} | "
            f"{stats.get('mean_answer_score', 0.0):.3f} | {stats['proof_found']} | "
            f"{stats['no_query']} | {stats['weakly_aligned']} | {stats['fallback_used']} | "
            f"{stats['avg_latency_seconds']:.4f}s | {stats['median_latency_seconds']:.4f}s |"
        )
    return "\n".join(lines)



async def main() -> int:
    cli = argparse.ArgumentParser(description="Benchmark all PLN-RAG parsers without ConceptNet.")
    cli.add_argument(
        "--mode",
        choices=("isolated", "cumulative"),
        default="isolated",
        help="Whether to reset per case or accumulate knowledge per parser across a suite",
    )
    cli.add_argument(
        "--suite",
        choices=("smoke", "entailmentbank", "abstracts", "combined"),
        default="combined",
        help="Benchmark suite to run",
    )
    cli.add_argument(
        "--suite-file",
        help="Optional path to a custom suite JSON file (overrides --suite)",
    )
    cli.add_argument(
        "--gold-file",
        help="Gold answers for answer grading; defaults to a sibling <suite>_gold.json",
    )
    cli.add_argument(
        "--parsers",
        nargs="+",
        choices=AVAILABLE_PARSERS,
        default=list(ACTIVE_PARSERS),
        help="Parsers to include in this benchmark run",
    )
    cli.add_argument(
        "--progress",
        action="store_true",
        help="Print per-case progress to stderr",
    )
    cli.add_argument("--quick", action="store_true", help="Run a reduced representative case set")
    cli.add_argument(
        "--output-dir",
        default="data/benchmarks",
        help="Directory where benchmark JSON reports are written",
    )
    cli.add_argument(
        "--case-ids",
        action="append",
        default=[],
        help="Repeatable case_id to include (e.g. --case-ids A01)",
    )
    cli.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the LLM reachability check (only for runs that need no LLM)",
    )
    cli.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of cases to run (after filtering)",
    )
    args = cli.parse_args()

    if not args.skip_preflight:
        ok, detail = _preflight_llm()
        if not ok:
            print(
                "Pre-flight failed: the configured LLM is not reachable.\n"
                f"  {detail}\n"
                "Parsers swallow this error and return no atoms, so the run would "
                "finish with an all-zero report and exit 0. Aborting instead.\n"
                "Use --skip-preflight to override.",
                file=sys.stderr,
            )
            return 2

        ok, detail = _preflight_reasoner()
        if not ok:
            print(
                "Pre-flight failed: backward chaining is not working.\n"
                f"  {detail}\n"
                "Reasoner.query swallows this and returns no proofs, so the run "
                "would report proof_found=0 for every case and exit 0. Aborting "
                "instead. Check the PeTTa/PeTTaChainer pins in the Dockerfile.\n"
                "Use --skip-preflight to override.",
                file=sys.stderr,
            )
            return 2

    if args.suite_file:
        suite_path = Path(args.suite_file)
        suite_metadata, cases = _select_cases_from_file(suite_path, args.quick)
    else:
        suite_metadata, cases = _select_cases(args.suite, args.quick)

    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [
            case
            for case in cases
            if case.get("case_id") in wanted
            or case.get("id") in wanted
            or case.get("name") in wanted
        ]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    run_id = uuid.uuid4().hex[:8]
    suite_label = suite_metadata.get("suite") if isinstance(suite_metadata, dict) else None
    suite_label = suite_label or (Path(args.suite_file).stem if args.suite_file else args.suite)

    gold_cases, gold_file = _resolve_gold(args.gold_file, suite_metadata or {}, suite_label)
    _apply_gold_expectations(cases, gold_cases)

    payload: dict[str, object] = {
        "run_id": run_id,
        "conceptnet_enabled": False,
        "mode": args.mode,
        "suite": suite_label,
        "suite_metadata": suite_metadata,
        "gold_file": gold_file,
        "case_count": len(cases),
        "parsers": {},
        "summary": {},
    }

    payload["active_parsers"] = list(args.parsers)

    parser_total = len(args.parsers)
    for parser_index, parser_name in enumerate(args.parsers, start=1):
        results = []
        if args.mode == "cumulative":
            try:
                results = await _benchmark_parser_cumulative(
                    parser_name,
                    cases,
                    run_id,
                    suite_label,
                    progress=args.progress,
                    parser_index=parser_index,
                    parser_total=parser_total,
                )
            except Exception as exc:
                for case in cases:
                    results.append(
                        {
                            "case": case,
                            "error": str(exc),
                            "proof_found": False,
                            "correct": False,
                            "timing": {
                                "parser_init_seconds": 0.0,
                                "parse_only_seconds": 0.0,
                                "ingest_seconds": 0.0,
                                "query_seconds": 0.0,
                                "total_seconds": 0.0,
                            },
                            "end_to_end": {
                                "query": {"query_status": "error", "fallback_used": False}
                            },
                        }
                    )
        else:
            for case in cases:
                try:
                    result = await _benchmark_case(parser_name, case, run_id)
                except Exception as exc:
                    result = {
                        "case": case,
                        "error": str(exc),
                        "proof_found": False,
                        "correct": False,
                        "timing": {
                            "parser_init_seconds": 0.0,
                            "parse_only_seconds": 0.0,
                            "ingest_seconds": 0.0,
                            "query_seconds": 0.0,
                            "total_seconds": 0.0,
                        },
                        "end_to_end": {"query": {"query_status": "error", "fallback_used": False}},
                    }
                results.append(result)
        payload["parsers"][parser_name] = results
        _apply_grading(results, gold_cases)
        payload["summary"][parser_name] = _summarize_parser(results)

    warnings = _validity_warnings(payload["summary"])
    payload["valid"] = not warnings
    payload["validity_warnings"] = warnings

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"parser_benchmark_{_slugify(suite_label)}_{_slugify(args.mode)}_{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "summary": payload["summary"]}, indent=2))
    print()
    print(_markdown_summary(payload["summary"]))

    if warnings:
        print(file=sys.stderr)
        print("Report marked invalid — do not use it as a baseline:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
