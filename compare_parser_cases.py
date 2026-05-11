import asyncio
import json
import os
import uuid
from pathlib import Path

from config import get_settings
from core.service import PLNRAGService


CASES = [
    {
        "name": "fish-smart-yesno",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Is Kebede smart?",
    },
    {
        "name": "dog-animal-yesno",
        "texts": ["Dogs are animals.", "Fido is a dog."],
        "question": "Is Fido an animal?",
    },
    {
        "name": "human-mortal-yesno",
        "texts": ["Humans are mortal.", "Socrates is a human."],
        "question": "Is Socrates mortal?",
    },
    {
        "name": "coffee-awake-yesno",
        "texts": ["People who drink coffee stay awake.", "Sara drinks coffee."],
        "question": "Is Sara awake?",
    },
    {
        "name": "parent-caring-yesno",
        "texts": ["Parents care for their children.", "Alemu is a parent."],
        "question": "Does Alemu care for children?",
    },
    {
        "name": "teacher-educates-yesno",
        "texts": ["Teachers educate students.", "Marta is a teacher."],
        "question": "Does Marta educate students?",
    },
    {
        "name": "doctor-heals-yesno",
        "texts": ["Doctors heal patients.", "Meron is a doctor."],
        "question": "Does Meron heal patients?",
    },
    {
        "name": "programmer-solves-yesno",
        "texts": ["Programmers solve problems.", "Bekele is a programmer."],
        "question": "Does Bekele solve problems?",
    },
    {
        "name": "smart-open",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Who is smart?",
    },
    {
        "name": "eat-open",
        "texts": ["Kebede eats fish."],
        "question": "What does Kebede eat?",
    },
]


def _get_parser_factory(name: str):
    if name == "nl2pln":
        from parsers.nl2pln_parser import NL2PLNParser

        return NL2PLNParser
    if name == "canonical_pln_prev":
        from parsers.canonical_pln_prev_parser import CanonicalPLNPrevParser

        return CanonicalPLNPrevParser
    if name == "canonical_pln_fallback_off":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return CanonicalPLNParser
    if name == "canonical_pln_fallback_on":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return CanonicalPLNParser
    raise ValueError(f"Unsupported parser '{name}'")


async def _run_case(parser_name: str, case: dict, run_id: str) -> dict:
    collection = f"cmp_{parser_name}_{case['name']}_{run_id}".replace("-", "_")
    atomspace_path = f"data/atomspace/{collection}.metta"

    os.environ["QDRANT_COLLECTION"] = collection
    os.environ["ATOMSPACE_PATH"] = atomspace_path
    if parser_name == "canonical_pln_fallback_off":
        os.environ["QUERY_FALLBACK_ENABLED"] = "false"
    elif parser_name == "canonical_pln_fallback_on":
        os.environ["QUERY_FALLBACK_ENABLED"] = "true"
    else:
        os.environ.pop("QUERY_FALLBACK_ENABLED", None)
    get_settings.cache_clear()

    parser = _get_parser_factory(parser_name)()
    service = PLNRAGService(parser)
    service.reset("all")

    try:
        ingest_results = await service.ingest_batch(case["texts"])
        query_response = await service.query(case["question"])

        return {
            "collection": collection,
            "atomspace_path": atomspace_path,
            "ingest": [
                {
                    "text": item.text,
                    "status": item.status,
                    "atoms": item.atoms,
                    "error": item.error,
                }
                for item in ingest_results
            ],
            "query": {
                "question": query_response.question,
                "pln_query": query_response.pln_query,
                "original_query": query_response.original_query,
                "executed_query": query_response.executed_query,
                "fallback_used": query_response.fallback_used,
                "query_status": query_response.query_status,
                "raw_proof": query_response.raw_proof,
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
            "proof_found": bool(
                query_response.raw_proof and query_response.raw_proof != "[]"
            ),
        }
    finally:
        service.reset("all")
        atomspace_file = Path(atomspace_path)
        if atomspace_file.exists():
            atomspace_file.unlink()


async def main() -> int:
    run_id = uuid.uuid4().hex[:8]
    payload = {"run_id": run_id, "parsers": {}}

    for parser_name in (
        "nl2pln",
        "canonical_pln_prev",
        "canonical_pln_fallback_off",
        "canonical_pln_fallback_on",
    ):
        parser_results = []
        for case in CASES:
            parser_results.append(
                {
                    "case": case,
                    "result": await _run_case(parser_name, case, run_id),
                }
            )
        payload["parsers"][parser_name] = parser_results

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
