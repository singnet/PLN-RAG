#!/usr/bin/env python3
"""Evaluate the non-LLM coreference resolver on a synthetic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.coreference import (  # noqa: E402
    CoreferenceCluster,
    CoreferenceMention,
    CoreferencePrediction,
    CoreferenceResolver,
)


SCHEMA_NAME = "pln-rag.coreference-evaluation"
SCHEMA_VERSION = "1.0"
DEFAULT_DATASET = ROOT / "data" / "benchmarks" / "coreference_pronouns_v1.json"


class DatasetError(ValueError):
    """Raised when benchmark data cannot be evaluated safely."""


class FixtureBackend:
    """Return the declared clusters for each exact benchmark input."""

    name = "fixture"
    model_name = "gold-cluster-spans"
    loaded = True
    load_error = None

    def __init__(self, clusters_by_text: dict[str, tuple[CoreferenceCluster, ...]]) -> None:
        self._clusters_by_text = clusters_by_text
        self.calls = 0

    def predict(self, text: str) -> CoreferencePrediction:
        self.calls += 1
        if text not in self._clusters_by_text:
            raise DatasetError("Fixture received text that is not in the benchmark")
        return CoreferencePrediction(
            backend=self.name,
            model=self.model_name,
            clusters=self._clusters_by_text[text],
        )


def _require(value: Any, expected_type: type, location: str) -> Any:
    if not isinstance(value, expected_type):
        raise DatasetError(f"{location} must be {expected_type.__name__}")
    return value


def _parse_clusters(case: dict[str, Any], location: str) -> tuple[CoreferenceCluster, ...]:
    text = _require(case.get("text"), str, f"{location}.text")
    raw_clusters = _require(case.get("clusters"), list, f"{location}.clusters")
    clusters: list[CoreferenceCluster] = []
    occupied: set[tuple[int, int]] = set()

    for cluster_index, raw_cluster in enumerate(raw_clusters):
        cluster_location = f"{location}.clusters[{cluster_index}]"
        _require(raw_cluster, dict, cluster_location)
        raw_mentions = _require(raw_cluster.get("mentions"), list, f"{cluster_location}.mentions")
        if not raw_mentions:
            raise DatasetError(f"{cluster_location}.mentions must not be empty")
        mentions: list[CoreferenceMention] = []
        for mention_index, raw_mention in enumerate(raw_mentions):
            mention_location = f"{cluster_location}.mentions[{mention_index}]"
            _require(raw_mention, dict, mention_location)
            start = raw_mention.get("start")
            end = raw_mention.get("end")
            mention_text = raw_mention.get("text")
            if type(start) is not int or type(end) is not int:
                raise DatasetError(f"{mention_location} start/end must be integers")
            if not isinstance(mention_text, str):
                raise DatasetError(f"{mention_location}.text must be str")
            if start < 0 or end <= start or end > len(text):
                raise DatasetError(f"{mention_location} has invalid span [{start}, {end})")
            if text[start:end] != mention_text:
                raise DatasetError(
                    f"{mention_location} text mismatch: {text[start:end]!r} != {mention_text!r}"
                )
            span = (start, end)
            if span in occupied:
                raise DatasetError(f"{mention_location} duplicates span [{start}, {end})")
            occupied.add(span)
            mentions.append(CoreferenceMention(start, end, mention_text))
        clusters.append(CoreferenceCluster(cluster_index, tuple(mentions)))
    return tuple(clusters)


def load_dataset(path: Path) -> dict[str, Any]:
    """Load and validate benchmark structure and every declared character span."""
    try:
        raw_bytes = path.read_bytes()
        dataset = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not load {path}: {exc}") from exc
    _require(dataset, dict, "dataset")
    cases = _require(dataset.get("cases"), list, "dataset.cases")
    if not cases:
        raise DatasetError("dataset.cases must not be empty")
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for index, case in enumerate(cases):
        location = f"dataset.cases[{index}]"
        _require(case, dict, location)
        case_id = _require(case.get("id"), str, f"{location}.id")
        text = _require(case.get("text"), str, f"{location}.text")
        _require(case.get("expected_resolved"), str, f"{location}.expected_resolved")
        _require(case.get("category"), str, f"{location}.category")
        if not case_id or case_id in seen_ids:
            raise DatasetError(f"{location}.id must be non-empty and unique")
        if not text or text in seen_texts:
            raise DatasetError(f"{location}.text must be non-empty and unique")
        seen_ids.add(case_id)
        seen_texts.add(text)
        case["_clusters"] = _parse_clusters(case, location)
    dataset["_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return dataset


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _replacement_key(replacement: Any) -> tuple[int, int, str]:
    return (replacement.start, replacement.end, replacement.replacement)


def _make_resolver(
    backend_name: str,
    dataset: dict[str, Any],
    model_name: str | None,
    min_confidence: float,
) -> tuple[CoreferenceResolver, FixtureBackend | None]:
    if backend_name == "fixture":
        fixture = FixtureBackend(
            {case["text"]: case["_clusters"] for case in dataset["cases"]}
        )
        return (
            CoreferenceResolver(
                enabled=True,
                min_confidence=min_confidence,
                backend=fixture,
            ),
            fixture,
        )
    return (
        CoreferenceResolver(
            enabled=backend_name != "none",
            backend_name=backend_name,
            model_name=model_name,
            min_confidence=min_confidence,
            device="cpu",
            fail_open=True,
        ),
        None,
    )


def evaluate(
    dataset_path: Path,
    backend_name: str = "fixture",
    *,
    model_name: str | None = None,
    min_confidence: float = 0.65,
) -> dict[str, Any]:
    """Evaluate a backend and return a JSON-serializable artifact."""
    dataset = load_dataset(dataset_path)
    resolver, fixture_backend = _make_resolver(
        backend_name, dataset, model_name, min_confidence
    )
    rows: list[dict[str, Any]] = []
    true_positives = false_positives = false_negatives = 0
    exact_matches = change_exact_matches = expected_changes = 0
    change_detection_matches = 0
    latencies: list[float] = []

    for case in dataset["cases"]:
        gold_backend = FixtureBackend({case["text"]: case["_clusters"]})
        gold = CoreferenceResolver(enabled=True, backend=gold_backend).resolve(case["text"])
        if gold.resolved != case["expected_resolved"]:
            raise DatasetError(
                f"{case['id']}.expected_resolved disagrees with its declared clusters: "
                f"{gold.resolved!r}"
            )

        result = resolver.resolve(case["text"])
        predicted_keys = {_replacement_key(item) for item in result.replacements}
        gold_keys = {_replacement_key(item) for item in gold.replacements}
        true_positives += len(predicted_keys & gold_keys)
        false_positives += len(predicted_keys - gold_keys)
        false_negatives += len(gold_keys - predicted_keys)
        exact = result.resolved == gold.resolved
        expected_change = gold.resolved != gold.original
        actual_change = result.resolved != result.original
        exact_matches += int(exact)
        expected_changes += int(expected_change)
        change_exact_matches += int(expected_change and exact)
        change_detection_matches += int(expected_change == actual_change)
        latencies.append(result.duration_seconds)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "original": result.original,
                "expected_resolved": gold.resolved,
                "resolved": result.resolved,
                "replacements": [asdict(item) for item in result.replacements],
                "expected_replacements": [asdict(item) for item in gold.replacements],
                "status": result.status,
                "timing": {"seconds": round(result.duration_seconds, 6)},
                "error": result.error,
                "diagnostics": list(result.diagnostics),
                "exact": exact,
            }
        )

    total = len(rows)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 1.0
    recall = true_positives / recall_denominator if recall_denominator else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    status = resolver.status()
    invocations = fixture_backend.calls if fixture_backend is not None else status.documents_processed
    failures = sum(row["status"] == "failed_open" for row in rows)

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "run": {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": backend_name,
            "device": "cpu",
        },
        "dataset": {
            "id": dataset.get("suite_id"),
            "version": dataset.get("version"),
            "path": str(dataset_path),
            "sha256": dataset["_sha256"],
            "cases": total,
        },
        "resolver": {
            **asdict(status),
            "configured_model": model_name,
            "min_confidence": min_confidence,
            "invocations": invocations,
        },
        "cases": rows,
        "summary": {
            "cases": total,
            "exact_matches": exact_matches,
            "exact_accuracy": exact_matches / total,
            "expected_change_cases": expected_changes,
            "change_exact_matches": change_exact_matches,
            "change_accuracy": (
                change_exact_matches / expected_changes if expected_changes else 1.0
            ),
            "change_detection_accuracy": change_detection_matches / total,
            "replacement_true_positives": true_positives,
            "replacement_false_positives": false_positives,
            "replacement_false_negatives": false_negatives,
            "replacement_precision": precision,
            "replacement_recall": recall,
            "replacement_f1": f1,
            "failures": failures,
            "median_latency_seconds": round(float(statistics.median(latencies)), 6),
            "p95_latency_seconds": round(_percentile(latencies, 0.95), 6),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact path")
    parser.add_argument(
        "--backend",
        choices=("fixture", "none", "fcoref", "lingmess"),
        default="fixture",
    )
    parser.add_argument("--model", default=None, help="Optional real-backend model name")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--require-effective",
        action="store_true",
        help="Fail if resolution failed open or the backend was never invoked",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = evaluate(
            args.dataset,
            args.backend,
            model_name=args.model,
            min_confidence=args.min_confidence,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = artifact["summary"]
    print(
        f"{args.backend}: exact={summary['exact_accuracy']:.3f} "
        f"replacement_f1={summary['replacement_f1']:.3f} "
        f"failures={summary['failures']} artifact={args.output}"
    )
    ineffective = summary["failures"] > 0 or artifact["resolver"]["invocations"] == 0
    return 3 if args.require_effective and ineffective else 0


if __name__ == "__main__":
    raise SystemExit(main())
