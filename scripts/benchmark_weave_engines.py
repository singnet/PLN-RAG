"""Deterministic, service-free microbenchmark for SENF weave engines."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from core.senf.extractor import extract_senf
from core.senf.types import SENF
from core.senf.weave import WeaveResult, build_weaves


SCHEMA = "senf-weave-engine-microbenchmark-v1"
ENGINES = ("beam", "hierarchical")
MAX_SIZE_COUNT = 8
MAX_SPARSITY_COUNT = 8
MAX_SIZE = 64
MAX_SOURCES = 8
MAX_REPEATS = 20
MAX_WARMUPS = 10
MAX_TOP_K = 8
BENCHMARK_SETTINGS = {
    "senf_weave_per_source_k": 3,
    "senf_weave_beam_width": 32,
    "senf_weave_max_frames": 64,
    "senf_weave_max_pair_candidates": 256,
    "senf_weave_max_exemplar_alternatives": 4,
    "senf_weave_max_cost": 2.0,
    "senf_query_max_priors": 16,
    "senf_query_max_source_frames": 128,
    "senf_weave_global_candidate_cap": 512,
    "senf_weave_max_cells": 65536,
    "senf_weave_max_seeds": 32,
    "senf_weave_max_iterations": 200,
    "senf_weave_sinkhorn_tolerance": 1e-7,
    "senf_weave_sinkhorn_regularization": 0.25,
    "senf_weave_forget_fine_costs": False,
    "senf_weave_coarse_identity": False,
    "senf_weave_max_conflict_cost": 1000000.0,
    "senf_weave_max_transport_cost": 1000000.0,
    "senf_weave_require_context_match": False,
}


def _bounded_ints(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or len(values) > MAX_SIZE_COUNT or len(values) != len(set(values)) or any(value <= 0 or value > MAX_SIZE for value in values):
        raise argparse.ArgumentTypeError(f"sizes must be at most {MAX_SIZE_COUNT} unique integers in [1, {MAX_SIZE}]")
    return values


def _sparsities(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values or len(values) > MAX_SPARSITY_COUNT or len(values) != len(set(values)) or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError(f"sparsities must be at most {MAX_SPARSITY_COUNT} unique finite values in [0, 1]")
    return values


def _bounded_cli_int(name: str, lower: int, upper: int):
    def parse(raw: str) -> int:
        value = int(raw)
        if not lower <= value <= upper:
            raise argparse.ArgumentTypeError(f"{name} must be in [{lower}, {upper}]")
        return value
    return parse


def _selected(frame_index: int, source_index: int, sparsity: float) -> bool:
    # A fixed integer lattice avoids PRNG/version dependence.
    rank = (frame_index * 619 + source_index * 277 + 101) % 1000
    return rank >= round(sparsity * 1000)


def synthetic_case(size: int, sparsity: float, source_count: int) -> tuple[SENF, list[SENF]]:
    """Build bounded SENFs where sparsity is the incompatible-pair fraction."""
    if type(size) is not int or type(source_count) is not int or size <= 0 or size > MAX_SIZE or source_count <= 0 or source_count > MAX_SOURCES:
        raise ValueError(f"size must be in [1, {MAX_SIZE}] and source_count in [1, {MAX_SOURCES}]")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")

    query_atoms = [f"(: q{i:03d} (Pred{i:03d} entity{i:03d}) $tv)" for i in range(size)]
    query = extract_senf("bench-query", "synthetic benchmark query", query_atoms)
    sources = []
    for source_index in range(source_count):
        atoms = []
        for frame_index in range(size):
            head = (
                f"Pred{frame_index:03d}"
                if _selected(frame_index, source_index, sparsity)
                else f"Noise{source_index:02d}_{frame_index:03d}"
            )
            atoms.append(
                f"(: s{source_index:02d}_{frame_index:03d} "
                f"({head} entity{frame_index:03d}) (STV 1 1))"
            )
        sources.append(extract_senf(
            f"bench-source-{source_index:02d}",
            f"synthetic benchmark source {source_index}",
            atoms,
        ))
    return query, sources


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def result_hash(results: Sequence[WeaveResult]) -> str:
    payload = json.dumps(_jsonable(results), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def _engine_setting(engine: str) -> Iterator[None]:
    settings = get_settings()
    previous = settings.senf_weave_engine
    settings.senf_weave_engine = engine
    try:
        yield
    finally:
        settings.senf_weave_engine = previous


@contextmanager
def _benchmark_settings() -> Iterator[None]:
    settings = get_settings()
    previous = {name: getattr(settings, name) for name in BENCHMARK_SETTINGS}
    for name, value in BENCHMARK_SETTINGS.items():
        setattr(settings, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


def _run_once(engine: str, query: SENF, sources: Sequence[SENF], k: int) -> tuple[WeaveResult, ...]:
    with _engine_setting(engine):
        return build_weaves(query, sources, k=k)


def _engine_row(
    engine: str,
    query: SENF,
    sources: Sequence[SENF],
    repeats: int,
    warmups: int,
    k: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        _run_once(engine, query, sources, k)

    elapsed = []
    hashes = []
    results: tuple[WeaveResult, ...] = ()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        results = _run_once(engine, query, sources, k)
        elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
        hashes.append(result_hash(results))
    reverse_results = _run_once(engine, query, tuple(reversed(sources)), k)
    reverse_hash = result_hash(reverse_results)
    diagnostics = [item.polish for item in results if item.polish is not None]
    residuals = [value for item in results for value in (item.residuals or ())]
    fallback_reasons = sorted({
        item.fallback_reason for item in results if item.fallback_reason
    })
    return {
        "engine": engine,
        "result_hash": hashes[0],
        "repeat_hashes_identical": len(set(hashes)) == 1,
        "source_order_hash": reverse_hash,
        "source_order_deterministic": hashes[0] == reverse_hash,
        "result_count": len(results),
        "pair_counts": [len(item.pairs) for item in results],
        "aligned_result_count": sum(item.aligned for item in results),
        "candidate_count": max((item.candidate_count for item in diagnostics), default=0),
        "residuals": [list(item.residuals) if item.residuals is not None else None for item in results],
        "max_residual": max(residuals, default=None),
        "fallback": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "wall_time_ms": {
            "samples": [round(value, 6) for value in elapsed],
            "median": round(statistics.median(elapsed), 6),
            "min": round(min(elapsed), 6),
            "max": round(max(elapsed), 6),
        },
    }


def _bounds(query: SENF, sources: Sequence[SENF], compatible_pairs: int) -> dict[str, Any]:
    settings = get_settings()
    query_frames = min(settings.senf_weave_max_frames, sum(frame.clause_role == "fact" for frame in query.frames))
    source_frames = sum(min(settings.senf_weave_max_frames, sum(frame.clause_role != "premise" for frame in source.frames)) for source in sources)
    pair_work = query_frames * source_frames
    pair_work_cap = settings.senf_weave_global_candidate_cap * settings.senf_query_max_priors
    # Every compatible source frame is unique in this synthetic fixture. This is
    # the conservative global stage shape; local/block stages cannot exceed it.
    participating_queries = min(query_frames, compatible_pairs)
    sinkhorn_cells = (participating_queries + compatible_pairs) ** 2
    assignment_cells = query_frames * (compatible_pairs + query_frames)
    largest_matrix_cells = max(sinkhorn_cells, assignment_cells)
    return {
        "query_frames": query_frames,
        "source_frames": source_frames,
        "pair_work": pair_work,
        "pair_work_cap": pair_work_cap,
        "pair_work_within_bound": pair_work <= pair_work_cap,
        "compatible_pairs": compatible_pairs,
        "candidate_cap": settings.senf_weave_global_candidate_cap,
        "candidates_within_bound": compatible_pairs <= settings.senf_weave_global_candidate_cap,
        "sinkhorn_cells_upper_bound": sinkhorn_cells,
        "assignment_cells_upper_bound": assignment_cells,
        "largest_matrix_cells_upper_bound": largest_matrix_cells,
        "matrix_cell_cap": settings.senf_weave_max_cells,
        "matrix_cells_within_bound": largest_matrix_cells <= settings.senf_weave_max_cells,
    }


def build_ablation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize beam-to-hierarchical deltas in an existing report payload."""
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise ValueError("benchmark report must contain a cases list")
    comparisons = []
    for row in rows:
        engines = {item["engine"]: item for item in row.get("engines", [])}
        if not all(engine in engines for engine in ENGINES):
            continue
        beam, hierarchy = (engines[engine] for engine in ENGINES)
        comparisons.append({
            "case_id": row["case_id"],
            "result_hash_equal": beam["result_hash"] == hierarchy["result_hash"],
            "pair_count_delta": sum(hierarchy["pair_counts"]) - sum(beam["pair_counts"]),
            "fallback_changed": beam["fallback"] != hierarchy["fallback"],
            "hierarchical_to_beam_median_wall_time_ratio": (
                round(hierarchy["wall_time_ms"]["median"] / beam["wall_time_ms"]["median"], 6)
                if beam["wall_time_ms"]["median"] else None
            ),
        })
    return {
        "baseline": "beam",
        "variant": "hierarchical",
        "case_count": len(comparisons),
        "equal_result_hash_count": sum(item["result_hash_equal"] for item in comparisons),
        "source_order_deterministic_case_count": sum(
            all(engine["source_order_deterministic"] for engine in row.get("engines", []))
            for row in rows
            if len(row.get("engines", [])) == len(ENGINES)
        ),
        "comparisons": comparisons,
    }


def run_benchmark(
    sizes: Sequence[int],
    sparsities: Sequence[float],
    source_count: int = 2,
    repeats: int = 3,
    warmups: int = 1,
    k: int = 3,
) -> dict[str, Any]:
    if not sizes or len(sizes) > MAX_SIZE_COUNT or len(set(sizes)) != len(sizes) or any(type(value) is not int or not 1 <= value <= MAX_SIZE for value in sizes):
        raise ValueError("sizes exceed benchmark bounds")
    if not sparsities or len(sparsities) > MAX_SPARSITY_COUNT or len(set(sparsities)) != len(sparsities) or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in sparsities):
        raise ValueError("sparsities exceed benchmark bounds")
    if any(type(value) is not int for value in (source_count, repeats, warmups, k)) or not 1 <= source_count <= MAX_SOURCES or not 1 <= repeats <= MAX_REPEATS or not 0 <= warmups <= MAX_WARMUPS or not 1 <= k <= MAX_TOP_K:
        raise ValueError("sources, repeats, warmups, or top-k exceed benchmark bounds")
    cases = []
    fixture_hasher = hashlib.sha256()
    with _benchmark_settings():
        for size in sizes:
            for sparsity in sparsities:
                query, sources = synthetic_case(size, sparsity, source_count)
                fixture_hasher.update(json.dumps(_jsonable((query, sources)), sort_keys=True, separators=(",", ":")).encode())
                compatible_pairs = sum(
                    _selected(frame_index, source_index, sparsity)
                    for source_index in range(source_count)
                    for frame_index in range(size)
                )
                case_id = f"n{size}-s{sparsity:.3f}-sources{source_count}"
                cases.append({
                    "case_id": case_id,
                    "size": size,
                    "sparsity": sparsity,
                    "target_compatible_pair_density": 1.0 - sparsity,
                    "source_count": source_count,
                    "bounds": _bounds(query, sources, compatible_pairs),
                    "engines": [
                        _engine_row(engine, query, sources, repeats, warmups, k)
                        for engine in ENGINES
                    ],
                })
    command = (
        "python scripts/benchmark_weave_engines.py "
        f"--sizes {','.join(map(str, sizes))} --sparsities {','.join(map(str, sparsities))} "
        f"--sources {source_count} --repeats {repeats} --warmups {warmups} --top-k {k}"
    )
    report = {
        "schema": SCHEMA,
        "generator": "scripts/benchmark_weave_engines.py",
        "parameters": {
            "sizes": list(sizes),
            "sparsities": list(sparsities),
            "source_count": source_count,
            "repeats": repeats,
            "warmups": warmups,
            "top_k": k,
        },
        "timing": {
            "classification": "descriptive-only",
            "note": "Wall time is environment-sensitive; hashes, counts, residuals, fallbacks, and bounds are deterministic checks.",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
                "cpu_count": os.cpu_count(),
            },
        },
        "reproducibility": _reproducibility(command, fixture_hasher.hexdigest()),
        "cases": cases,
    }
    report["ablation"] = build_ablation_report(report)
    return report


def _reproducibility(command: str, dataset_hash: str) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    dependencies = {}
    for name in ("numpy", "pydantic", "pydantic-settings"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {
        "command": command,
        "dataset_sha256": dataset_hash,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": revision,
        "settings": {**dict(sorted(BENCHMARK_SETTINGS.items())), "senf_weave_engine": list(ENGINES)},
        "dependencies": dependencies,
        "determinism_scope": "same-process repeats plus one fresh-process hash test in pytest",
    }


def render_markdown(report: dict[str, Any], command: str | None = None) -> str:
    command = command or report["reproducibility"]["command"]
    lines = [
        "# SENF Weave Engine Microbenchmark",
        "",
        "This is a deterministic synthetic, service-free comparison. No LLM calls are made.",
        "Wall times are descriptive for the recorded environment and are not regression thresholds.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Environment",
        "",
    ]
    environment = report["timing"]["environment"]
    lines.extend(f"- `{key}`: {value}" for key, value in environment.items())
    lines.extend([
        "",
        "## Results",
        "",
        "| Case | Engine | Hash | Order stable | Pairs | Candidates | Max residual | Fallback | Median ms | Work / cap | Matrix / cap |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    for case in report["cases"]:
        bounds = case["bounds"]
        for engine in case["engines"]:
            residual = engine["max_residual"]
            lines.append(
                f"| {case['case_id']} | {engine['engine']} | `{engine['result_hash'][:12]}` | "
                f"{str(engine['source_order_deterministic']).lower()} | {sum(engine['pair_counts'])} | "
                f"{engine['candidate_count']} | {residual if residual is not None else 'n/a'} | "
                f"{','.join(engine['fallback_reasons']) or 'none'} | {engine['wall_time_ms']['median']:.6f} | "
                f"{bounds['pair_work']} / {bounds['pair_work_cap']} | "
                f"{bounds['largest_matrix_cells_upper_bound']} / {bounds['matrix_cell_cap']} |"
            )
    lines.extend([
        "",
        "`Pairs` is the sum across returned top-k results. `Candidates` is hierarchical polish diagnostics; the beam engine does not expose an equivalent counter. Matrix values are conservative dense-cell upper bounds for the hierarchical Sinkhorn and assignment stages.",
        "",
        "## Reproducibility",
        "",
        f"- Dataset SHA-256: `{report['reproducibility']['dataset_sha256']}`",
        f"- Generator SHA-256: `{report['reproducibility']['generator_sha256']}`",
        f"- Git revision: `{report['reproducibility']['git_revision']}`",
        f"- Determinism scope: {report['reproducibility']['determinism_scope']}",
        "",
        "```json",
        json.dumps({"settings": report["reproducibility"]["settings"], "dependencies": report["reproducibility"]["dependencies"]}, indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_bounded_ints, default=[2, 4, 8])
    parser.add_argument("--sparsities", type=_sparsities, default=[0.25, 0.5, 1.0])
    parser.add_argument("--sources", type=_bounded_cli_int("sources", 1, MAX_SOURCES), default=2)
    parser.add_argument("--repeats", type=_bounded_cli_int("repeats", 1, MAX_REPEATS), default=3)
    parser.add_argument("--warmups", type=_bounded_cli_int("warmups", 0, MAX_WARMUPS), default=1)
    parser.add_argument("--top-k", type=_bounded_cli_int("top-k", 1, MAX_TOP_K), default=3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.sizes, args.sparsities, args.sources, args.repeats, args.warmups, args.top_k
    )
    command = (
        "python scripts/benchmark_weave_engines.py "
        f"--sizes {','.join(map(str, args.sizes))} "
        f"--sparsities {','.join(map(str, args.sparsities))} "
        f"--sources {args.sources} --repeats {args.repeats} --warmups {args.warmups} "
        f"--top-k {args.top_k}"
    )
    if args.json_output:
        command += f" --json-output {args.json_output}"
    if args.markdown_output:
        command += f" --markdown-output {args.markdown_output}"
    report["reproducibility"]["command"] = command
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(report, command), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
