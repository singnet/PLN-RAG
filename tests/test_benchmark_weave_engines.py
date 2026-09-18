import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_weave_engines import (
    SCHEMA,
    build_ablation_report,
    main,
    render_markdown,
    result_hash,
    run_benchmark,
    synthetic_case,
)


def test_synthetic_case_and_hash_are_deterministic():
    first_query, first_sources = synthetic_case(2, 0.5, 2)
    second_query, second_sources = synthetic_case(2, 0.5, 2)

    assert first_query == second_query
    assert first_sources == second_sources
    first_report = run_benchmark([1], [0.0], source_count=1, repeats=1, warmups=0, k=1)
    second_report = run_benchmark([1], [0.0], source_count=1, repeats=1, warmups=0, k=1)
    first_results = [row["result_hash"] for row in first_report["cases"][0]["engines"]]
    second_results = [row["result_hash"] for row in second_report["cases"][0]["engines"]]
    assert first_results == second_results
    assert result_hash(()) != result_hash(first_sources)


def test_tiny_benchmark_captures_determinism_and_bounds():
    report = run_benchmark([2], [0.5], source_count=2, repeats=2, warmups=0, k=2)

    assert report["schema"] == SCHEMA
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["bounds"]["pair_work_within_bound"]
    assert case["bounds"]["matrix_cells_within_bound"]
    assert {row["engine"] for row in case["engines"]} == {"beam", "hierarchical"}
    assert all(row["repeat_hashes_identical"] for row in case["engines"])
    assert all(row["source_order_deterministic"] for row in case["engines"])
    assert all(row["wall_time_ms"]["samples"] for row in case["engines"])
    query, sources = synthetic_case(2, 0.5, 2)
    assert case["bounds"]["query_frames"] == len(query.frames)
    assert case["bounds"]["source_frames"] == sum(len(item.frames) for item in sources)
    assert case["bounds"]["pair_work"] == len(query.frames) * sum(len(item.frames) for item in sources)


def test_ablation_report_is_reusable_for_loaded_report():
    report = run_benchmark([1], [1.0], repeats=1, warmups=0)
    loaded = json.loads(json.dumps(report))

    summary = build_ablation_report(loaded)

    assert summary["baseline"] == "beam"
    assert summary["variant"] == "hierarchical"
    assert summary["case_count"] == 1
    assert summary["comparisons"][0]["case_id"] == report["cases"][0]["case_id"]


def test_full_sparsity_records_hierarchical_fallback():
    report = run_benchmark([1], [1.0], source_count=1, repeats=1, warmups=0)
    engines = {row["engine"]: row for row in report["cases"][0]["engines"]}

    assert not engines["beam"]["fallback"]
    assert engines["hierarchical"]["fallback"]
    assert engines["hierarchical"]["fallback_reasons"] == [
        "no_hard_compatible_candidates"
    ]


def test_tracked_small_report_has_deterministic_checks():
    path = Path("docs/benchmarks/senf_weave_engine_microbenchmark.json")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["schema"] == SCHEMA
    assert report["timing"]["classification"] == "descriptive-only"
    assert all(
        engine["repeat_hashes_identical"] and engine["source_order_deterministic"]
        for case in report["cases"]
        for engine in case["engines"]
    )
    assert all(
        case["bounds"]["pair_work_within_bound"]
        and case["bounds"]["matrix_cells_within_bound"]
        for case in report["cases"]
    )


def _without_descriptive_timing(report):
    report = json.loads(json.dumps(report))
    reproducibility = report["reproducibility"]
    assert reproducibility["git_revision"]
    assert reproducibility["dependencies"]
    reproducibility["git_revision"] = "<variable>"
    reproducibility["dependencies"] = {"recorded": "<variable>"}
    report["timing"]["environment"] = {}
    for case in report["cases"]:
        for engine in case["engines"]:
            engine["wall_time_ms"] = {}
    for item in report["ablation"]["comparisons"]:
        item["hierarchical_to_beam_median_wall_time_ratio"] = None
    return report


def test_tracked_reports_match_regeneration_except_descriptive_timing():
    docs = Path("docs/benchmarks")
    tracked = json.loads((docs / "senf_weave_engine_microbenchmark.json").read_text(encoding="utf-8"))
    parameters = tracked["parameters"]
    regenerated = run_benchmark(
        parameters["sizes"], parameters["sparsities"], parameters["source_count"],
        parameters["repeats"], parameters["warmups"], parameters["top_k"],
    )
    regenerated["reproducibility"]["command"] = tracked["reproducibility"]["command"]
    assert _without_descriptive_timing(regenerated) == _without_descriptive_timing(tracked)
    assert (docs / "senf_weave_engine_microbenchmark.md").read_text(encoding="utf-8") == render_markdown(tracked)
    assert f"Dataset SHA-256: `{tracked['reproducibility']['dataset_sha256']}`" in render_markdown(tracked)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sizes", "1,2,3,4,5,6,7,8,9"],
        ["--sources", "9"],
        ["--repeats", "21"],
        ["--warmups", "11"],
        ["--top-k", "9"],
    ],
)
def test_cli_rejects_workload_caps(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


def test_result_hash_is_stable_across_fresh_processes():
    code = (
        "from scripts.benchmark_weave_engines import run_benchmark; "
        "r=run_benchmark([1],[0.0],1,1,0,1); "
        "print(','.join(x['result_hash'] for x in r['cases'][0]['engines']))"
    )
    first = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    second = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert first == second


def test_cli_writes_json_and_markdown(tmp_path):
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    assert main([
        "--sizes", "1",
        "--sparsities", "1",
        "--sources", "1",
        "--repeats", "1",
        "--warmups", "0",
        "--json-output", str(json_path),
        "--markdown-output", str(markdown_path),
    ]) == 0

    assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == SCHEMA
    assert "python scripts/benchmark_weave_engines.py" in markdown_path.read_text(encoding="utf-8")
