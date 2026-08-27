"""Grade benchmark artifacts that are already on disk, offline.

A full run costs ~28 minutes per parser, so grading is applied to artifacts already written
rather than requiring a re-run. Gold comes from `--gold-file` or a sibling `<suite>_gold.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_grading as bg  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"{path}: expected a 'cases' object")
    return cases


def find_gold(suite: Optional[str], explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    if not suite:
        return None
    candidate = BENCHMARK_DIR / f"{suite}_gold.json"
    return candidate if candidate.exists() else None


def print_case_table(parser: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n### {parser}")
    print(f"{'case':<6} {'proof':<6} {'answer':<8} {'score':>6}  reason")
    for row in sorted(rows, key=lambda r: str(r["case_id"])):
        correct = row["answer_correct"]
        verdict = {True: "correct", False: "wrong", None: "ungraded"}[correct]
        score = row["answer_score"]
        score_text = f"{score:.2f}" if score is not None else "-"
        proof = "yes" if row["proof_found"] else "no"
        print(f"{row['case_id']:<6} {proof:<6} {verdict:<8} {score_text:>6}  {row['answer_reason']}")


def print_summary(parser: str, stats: dict[str, Any]) -> None:
    graded = stats["answer_graded"]
    rate = f"{stats['answer_correct']}/{graded}" if graded else "0/0"
    pct = f"{100 * stats['answer_correct'] / graded:.0f}%" if graded else "n/a"
    print(
        f"\n{parser}: answer_correct {rate} ({pct}), "
        f"mean_score {stats['mean_answer_score']:.3f}, "
        f"verdict_graded {stats['verdict_graded']}/{graded}, "
        f"proof_found {stats['proof_found']}/{stats['cases']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade PLN-RAG benchmark artifacts against gold.")
    ap.add_argument("paths", nargs="+", help="One or more benchmark JSON artifact paths")
    ap.add_argument("--gold-file", help="Gold file; defaults to a sibling <suite>_gold.json")
    ap.add_argument("--quiet", action="store_true", help="Summaries only, no per-case table")
    args = ap.parse_args()

    status = 0
    for raw in args.paths:
        path = Path(raw)
        data = json.loads(path.read_text(encoding="utf-8"))
        suite = data.get("suite")
        gold_path = find_gold(suite, args.gold_file)

        print(f"\n== {path.name} (suite={suite}, mode={data.get('mode')}, "
              f"run_id={data.get('run_id')})")
        if gold_path is None:
            print("no gold file found -- nothing to grade")
            status = 1
            continue
        print(f"gold: {gold_path.name}")
        if data.get("mode") == "cumulative":
            print("note: cumulative mode accumulates knowledge across cases; quote isolated runs")

        gold_cases = load_gold(gold_path)
        for parser, results in (data.get("parsers") or {}).items():
            if not isinstance(results, list):
                continue
            try:
                rows = bg.grade_results(gold_cases, results)
            except bg.ArtifactSchemaError as exc:
                print(f"\n### {parser}\nunreadable artifact: {exc}")
                status = 1
                continue
            if not args.quiet:
                print_case_table(parser, rows)
            print_summary(parser, bg.summarize(rows))

    return status


if __name__ == "__main__":
    raise SystemExit(main())
