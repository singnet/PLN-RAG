import argparse
import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _load_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["parsers"]["canonical_pln"][:limit]
    replay_cases = []
    for result in cases:
        atoms = [
            atom
            for ingest in result["end_to_end"]["ingest"]
            for atom in ingest.get("atoms", [])
        ]
        query_result = result["end_to_end"]["query"]
        query = query_result.get("executed_query") or query_result.get("pln_query")
        replay_cases.append(
            {
                "case_id": result["case"]["case_id"],
                "atoms": atoms,
                "query": query,
                "historical_proof_found": bool(
                    query_result.get("raw_proof")
                    and query_result.get("raw_proof") != "[]"
                ),
            }
        )
    return replay_cases


def _proof_ids(atoms: list[str]) -> set[str]:
    ids = set()
    for atom in atoms:
        match = re.match(r"\(:\s+([^\s()]+)", atom)
        if match:
            ids.add(match.group(1))
    return ids


def _serialize_result(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _new_handler():
    from pettachainer.pettachainer import PeTTaChainer

    return PeTTaChainer()


def _query(handler, query: str, timeout: int) -> dict[str, Any]:
    evaluated = None
    evaluate_error = None
    if hasattr(handler, "evaluate_query"):
        try:
            evaluated = str(handler.evaluate_query(query))
        except Exception as exc:
            evaluate_error = f"{type(exc).__name__}: {exc}"

    started = time.perf_counter()
    try:
        result = _serialize_result(handler.query(query, timeout_sec=timeout))
        error = None
    except Exception as exc:
        result = []
        error = f"{type(exc).__name__}: {exc}"
    return {
        "evaluated_query": evaluated,
        "evaluate_error": evaluate_error,
        "proofs": result,
        "proof_found": bool(result),
        "query_error": error,
        "query_seconds": round(time.perf_counter() - started, 4),
    }


def _add_atoms(handler, atoms: list[str]) -> tuple[int, list[dict[str, str]]]:
    accepted = 0
    rejected = []
    for atom in atoms:
        try:
            handler.add_atom(atom)
            accepted += 1
        except Exception as exc:
            rejected.append(
                {
                    "atom": atom,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return accepted, rejected


def _run_replay(
    cases: list[dict[str, Any]], mode: str, timeout: int
) -> list[dict[str, Any]]:
    handler = _new_handler() if mode == "cumulative" else None
    results = []
    for case in cases:
        if mode == "isolated":
            handler = _new_handler()
        accepted, rejected = _add_atoms(handler, case["atoms"])
        query_result = _query(handler, case["query"], timeout)
        current_ids = _proof_ids(case["atoms"])
        proof_text = "\n".join(query_result["proofs"])
        results.append(
            {
                "case_id": case["case_id"],
                "historical_proof_found": case["historical_proof_found"],
                "query": case["query"],
                "query_has_variables": bool(
                    re.search(r"[$?](?!prf\b|tv\b)[A-Za-z_]", case["query"])
                ),
                "atoms_total": len(case["atoms"]),
                "atoms_accepted": accepted,
                "atoms_rejected": len(rejected),
                "rejection_samples": rejected[:3],
                "proof_mentions_current_case_id": any(
                    proof_id in proof_text for proof_id in current_ids
                ),
                **query_result,
            }
        )
    return results


def _run_probes(timeout: int) -> list[dict[str, Any]]:
    probes = [
        {
            "name": "ground_fact",
            "atoms": ["(: alice_human (IsA alice human) (STV 1.0 1.0))"],
            "query": "(: $prf (IsA alice human) $tv)",
        },
        {
            "name": "variable_fact",
            "atoms": ["(: alice_human (IsA alice human) (STV 1.0 1.0))"],
            "query": "(: $prf (IsA $person human) $tv)",
        },
        {
            "name": "legacy_implication",
            "atoms": [
                "(: alice_human (IsA alice human) (STV 1.0 1.0))",
                "(: human_mortal (Implication (Premises (IsA $x human)) "
                "(Conclusions (IsA $x mortal))) (STV 1.0 1.0))",
            ],
            "query": "(: $prf (IsA alice mortal) $tv)",
        },
        {
            "name": "direct_implication",
            "atoms": [
                "(: alice_human (Member alice human) (STV 1.0 1.0))",
                "(: human_mortal (Implication (Member $x human) "
                "(Member $x mortal)) (STV 1.0 1.0))",
            ],
            "query": "(: $prf (Member alice mortal) $tv)",
        },
    ]
    results = []
    for probe in probes:
        handler = _new_handler()
        accepted, rejected = _add_atoms(handler, probe["atoms"])
        results.append(
            {
                "name": probe["name"],
                "atoms_total": len(probe["atoms"]),
                "atoms_accepted": accepted,
                "atoms_rejected": len(rejected),
                "rejection_samples": rejected[:3],
                **_query(handler, probe["query"], timeout),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay frozen benchmark atoms directly through PeTTaChainer."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeouts", type=int, nargs="+", default=[0, 180])
    args = parser.parse_args()

    cases = _load_cases(Path(args.report), args.limit)
    output = {
        "label": args.label,
        "petta_commit": os.getenv("MATRIX_PETTA_COMMIT"),
        "pettachainer_commit": os.getenv("MATRIX_PETTACHAINER_COMMIT"),
        "source_report": args.report,
        "runs": [],
    }

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace:
        with contextlib.redirect_stdout(trace), contextlib.redirect_stderr(trace):
            for timeout in args.timeouts:
                for mode in ("isolated", "cumulative"):
                    results = _run_replay(cases, mode, timeout)
                    output["runs"].append(
                        {
                            "mode": mode,
                            "timeout": timeout,
                            "proofs_found": sum(
                                result["proof_found"] for result in results
                            ),
                            "query_errors": sum(
                                bool(result["query_error"]) for result in results
                            ),
                            "atoms_rejected": sum(
                                result["atoms_rejected"] for result in results
                            ),
                            "results": results,
                        }
                    )
                probes = _run_probes(timeout)
                output["runs"].append(
                    {
                        "mode": "probes",
                        "timeout": timeout,
                        "proofs_found": sum(
                            result["proof_found"] for result in probes
                        ),
                        "query_errors": sum(
                            bool(result["query_error"]) for result in probes
                        ),
                        "atoms_rejected": sum(
                            result["atoms_rejected"] for result in probes
                        ),
                        "results": probes,
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "trace": str(trace_path),
                "summary": [
                    {
                        key: run[key]
                        for key in (
                            "mode",
                            "timeout",
                            "proofs_found",
                            "query_errors",
                            "atoms_rejected",
                        )
                    }
                    for run in output["runs"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
