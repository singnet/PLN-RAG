from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _head_from_goal(goal: str) -> str:
    # goal like: "(Head arg1 arg2)" or "(Not (Head arg))"
    goal = " ".join(goal.strip().split())
    if not goal.startswith("("):
        return "__unparsed__"

    # unwrap Not
    m_not = re.fullmatch(r"\(Not\s+(\(.+\))\)", goal)
    if m_not:
        goal = m_not.group(1)

    m = re.fullmatch(r"\(([A-Za-z][A-Za-z0-9_]*)\b.*\)", goal)
    return m.group(1) if m else "__unparsed__"


def _extract_goal_atom(query: str) -> str:
    # query like: (: $prf (GoalAtom ...) $tv)
    query = " ".join(str(query).strip().split())
    m = re.fullmatch(r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)", query)
    return m.group(1) if m else ""


def _head_from_query(query: str) -> str:
    goal = _extract_goal_atom(query)
    if not goal:
        return "__unparsed__"
    return _head_from_goal(goal)


def _safe_get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _summarize_parser(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = len(rows)
    proofs = [bool(r.get("proof_found")) for r in rows]
    proof_count = sum(proofs)
    fallback_used = [bool(_safe_get(r, ["end_to_end", "query", "fallback_used"])) for r in rows]
    no_query = [(_safe_get(r, ["end_to_end", "query", "query_status"]) == "no_query") for r in rows]

    ingest_plus_query = [
        float(_safe_get(r, ["timing", "ingest_seconds"]) or 0.0)
        + float(_safe_get(r, ["timing", "query_seconds"]) or 0.0)
        for r in rows
    ]

    # proxy rate among proved cases only
    proxy_flags: list[bool] = []
    unparsed_heads = 0
    for r in rows:
        if not r.get("proof_found"):
            continue
        q = _safe_get(r, ["end_to_end", "query"]) or {}
        original = str(q.get("original_query") or "")
        executed = str(q.get("executed_query") or "")
        h_orig = _head_from_query(original)
        h_exec = _head_from_query(executed)
        if "__unparsed__" in (h_orig, h_exec):
            unparsed_heads += 1
        proxy_flags.append(h_orig != h_exec)

    proxy_rate = (sum(proxy_flags) / len(proxy_flags)) if proxy_flags else 0.0

    # optional timing breakdown fields
    breakdown_fields = [
        "context_retrieval_seconds",
        "parse_query_seconds",
        "reasoning_seconds",
        "source_lookup_seconds",
        "answer_generation_seconds",
    ]
    breakdown: dict[str, float] = {}
    for field in breakdown_fields:
        vals: list[float] = []
        for r in rows:
            v = _safe_get(r, ["end_to_end", "query", field])
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            breakdown[field] = _median(vals)

    return {
        "cases": cases,
        "proof_rate": (proof_count / cases) if cases else 0.0,
        "proofs": proof_count,
        "fallback_usage_rate": (sum(fallback_used) / cases) if cases else 0.0,
        "no_query_rate": (sum(no_query) / cases) if cases else 0.0,
        "proxy_rate_proved_only": proxy_rate,
        "proved_cases": len(proxy_flags),
        "unparsed_head_count": unparsed_heads,
        "median_ingest_plus_query": _median(ingest_plus_query),
        "breakdown_medians": breakdown,
    }


def _print_table(summary: dict[str, dict[str, Any]], title: str) -> None:
    print(f"\n## {title}")
    print(
        "| Parser | Proofs | Proof Rate | Proxy Rate (Proved) | Fallback Use | No Query | Med Ingest+Query (s) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for parser, s in summary.items():
        proofs = f"{s['proofs']}/{s['cases']}"
        print(
            "| "
            + " | ".join(
                [
                    parser,
                    proofs,
                    f"{s['proof_rate']:.3f}",
                    f"{s['proxy_rate_proved_only']:.3f}",
                    f"{s['fallback_usage_rate']:.3f}",
                    f"{s['no_query_rate']:.3f}",
                    f"{s['median_ingest_plus_query']:.2f}",
                ]
            )
            + " |"
        )

    # Print breakdown medians if present for any parser
    any_breakdowns = any(s.get("breakdown_medians") for s in summary.values())
    if any_breakdowns:
        print("\nBreakdown medians (seconds), where available:")
        for parser, s in summary.items():
            b = s.get("breakdown_medians") or {}
            if not b:
                continue
            ordered = [
                "context_retrieval_seconds",
                "parse_query_seconds",
                "reasoning_seconds",
                "source_lookup_seconds",
                "answer_generation_seconds",
            ]
            parts = [f"{k}={b[k]:.2f}" for k in ordered if k in b]
            print(f"- {parser}: " + ", ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze PLN-RAG benchmark artifacts.")
    ap.add_argument(
        "paths",
        nargs="+",
        help="One or more benchmark JSON artifact paths",
    )
    args = ap.parse_args()

    for raw in args.paths:
        path = Path(raw)
        data = _load(path)
        suite = data.get("suite")
        mode = data.get("mode")
        run_id = data.get("run_id")
        title = f"{path.name} (suite={suite}, mode={mode}, run_id={run_id})"

        parsers = data.get("parsers") or {}
        summary: dict[str, dict[str, Any]] = {}
        for parser, rows in parsers.items():
            if isinstance(rows, list):
                summary[parser] = _summarize_parser(rows)

        _print_table(summary, title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
