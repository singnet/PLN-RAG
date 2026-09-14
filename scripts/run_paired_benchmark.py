from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


BACKENDS = ("none", "fcoref", "lingmess")


def _benchmark_command(args: argparse.Namespace, backend: str, pair_id: str) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "benchmark_parsers.py"),
        "--mode",
        args.mode,
        "--parsers",
        *args.parsers,
        "--coreference-backend",
        backend,
        "--pair-id",
        pair_id,
        "--output-dir",
        args.output_dir,
        "--llm-cassette-mode",
        getattr(args, "llm_cassette_mode", "off"),
    ]
    cassette_mode = getattr(args, "llm_cassette_mode", "off")
    cassette_path = getattr(args, "llm_cassette_path", None)
    if cassette_mode != "off":
        command.extend(
            [
                "--llm-cassette-path",
                cassette_path,
                "--llm-cassette-scope",
                backend,
            ]
        )
    if args.suite_file:
        command.extend(["--suite-file", args.suite_file])
    else:
        command.extend(["--suite", args.suite])
    if args.quick:
        command.append("--quick")
    if args.progress:
        command.append("--progress")
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    for case_id in args.case_ids:
        command.extend(["--case-ids", case_id])
    return command


def _output_from_stdout(stdout: str) -> str:
    decoder = json.JSONDecoder()
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        output = value.get("output") if isinstance(value, dict) else None
        if output:
            return str(output)
    raise ValueError("benchmark did not emit its output manifest as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run comparable coreference benchmarks sequentially.")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--mode", choices=("isolated", "cumulative"), default="cumulative")
    parser.add_argument("--suite", default="combined")
    parser.add_argument("--suite-file")
    parser.add_argument("--parsers", nargs="+", default=["nl2pln", "canonical_pln"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--case-ids", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="data/benchmarks")
    parser.add_argument(
        "--llm-cassette-mode",
        choices=("off", "capture", "replay"),
        default="off",
    )
    parser.add_argument("--llm-cassette-path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(set(args.backends)) != len(args.backends):
        raise SystemExit("backends must be unique")
    if len(args.backends) < 2:
        raise SystemExit("paired benchmark requires at least two backends")
    if args.llm_cassette_mode != "off" and not args.llm_cassette_path:
        raise SystemExit("--llm-cassette-path is required for capture or replay")

    pair_id = args.pair_id or uuid.uuid4().hex[:12]
    root = Path(__file__).resolve().parents[1]
    runs: list[dict[str, Any]] = []
    for backend in args.backends:
        command = _benchmark_command(args, backend, pair_id)
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        run: dict[str, Any] = {
            "backend": backend,
            "returncode": completed.returncode,
            "command": command,
            "output": None,
        }
        try:
            run["output"] = _output_from_stdout(completed.stdout)
        except ValueError as exc:
            run["error"] = str(exc)
        runs.append(run)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"paired_benchmark_{pair_id}.json"
    manifest = {
        "schema_version": 1,
        "pair_id": pair_id,
        "settings": {
            "mode": args.mode,
            "suite": args.suite,
            "suite_file": args.suite_file,
            "parsers": args.parsers,
            "quick": args.quick,
            "case_ids": args.case_ids,
            "limit": args.limit,
            "llm_cassette_mode": args.llm_cassette_mode,
            "llm_cassette_path": args.llm_cassette_path,
        },
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "pair_id": pair_id, "runs": runs}, indent=2))
    return 0 if all(run["returncode"] == 0 and run["output"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
