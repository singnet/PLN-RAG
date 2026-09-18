#!/usr/bin/env python3
"""Run the deterministic SENF component gold benchmark without external services."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-senf-benchmark")

from config import get_settings  # noqa: E402
from core.senf.bridge import _arguments, executable_bridge_atoms  # noqa: E402
from core.senf.exemplars import score_exemplars  # noqa: E402
from core.senf.extractor import extract_senf  # noqa: E402
from core.senf.identity import resolve_identity  # noqa: E402
from core.senf.types import EntityRef  # noqa: E402
from core.senf.weave import build_weaves, weave  # noqa: E402

DEFAULT_GOLD = ROOT / "data" / "benchmarks" / "senf_components_v1.json"
SCHEMA_VERSION = "senf-components-gold/v1"
MAPPING_TYPES = ("frame", "entity", "role", "exemplar", "time", "location")
GUARD_RE = re.compile(r"^[A-Za-z0-9_.:-]+(?:\+[A-Za-z0-9_.:-]+)*->[A-Za-z0-9_.:-]+$")
MAPPING_RE = re.compile(r"^[^\s>|]+->[^\s>|]+$")
ADAPTER_ATOM_RE = re.compile(r"^\(: senf_adapter_\d+_[^\s]+ \(Implication .+\) \(STV ([0-9.]+) ([0-9.]+)\)\)$")
COMPONENT_SETTINGS = {
    "senf_identity_threshold": 0.75,
    "senf_exemplar_enabled": True,
    "senf_weave_top_k": 3,
    "senf_weave_per_source_k": 3,
    "senf_weave_beam_width": 32,
    "senf_weave_max_frames": 64,
    "senf_weave_max_pair_candidates": 256,
    "senf_weave_max_exemplar_alternatives": 4,
    "senf_weave_engine": "hierarchical",
    "senf_query_max_priors": 16,
    "senf_query_max_source_frames": 128,
    "senf_query_max_mentions": 256,
    "senf_query_max_candidate_work": 64,
    "senf_weave_global_candidate_cap": 512,
    "senf_weave_max_cells": 65536,
    "senf_weave_max_seeds": 32,
    "senf_weave_max_iterations": 200,
    "senf_weave_sinkhorn_tolerance": 1e-7,
    "senf_weave_sinkhorn_regularization": 0.25,
    "senf_weave_max_cost": 2.0,
    "senf_weave_max_conflict_cost": 1000000.0,
    "senf_weave_max_transport_cost": 1000000.0,
    "senf_weave_require_context_match": False,
    "senf_weave_forget_fine_costs": False,
    "senf_weave_coarse_identity": False,
}


class GoldSchemaError(ValueError):
    """The gold file is malformed or internally inconsistent."""


def _fail(path: str, message: str) -> None:
    raise GoldSchemaError(f"{path}: {message}")


def _object(value: Any, path: str, keys: set[str], required: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        _fail(path, "expected object")
    unknown = set(value) - keys
    missing = (required or keys) - set(value)
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        _fail(path, f"missing field(s): {', '.join(sorted(missing))}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected non-empty string")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail(path, "expected list of non-empty strings")
    if len(value) != len(set(value)):
        _fail(path, "duplicate values")
    return value


def _strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail(path, "expected list of non-empty strings")
    return value


def _formatted_strings(value: Any, path: str, pattern: re.Pattern[str]) -> list[str]:
    values = _string_list(value, path)
    for index, item in enumerate(values):
        if pattern.fullmatch(item) is None:
            _fail(f"{path}[{index}]", "invalid format")
    return values


def _pairs(value: Any, path: str, known_mentions: set[str]) -> list[list[str]]:
    if not isinstance(value, list):
        _fail(path, "expected list")
    normalized = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            _fail(f"{path}[{index}]", "expected a two-item mention pair")
        left, right = (_string(item, f"{path}[{index}]") for item in pair)
        if left == right or left not in known_mentions or right not in known_mentions:
            _fail(f"{path}[{index}]", "pair must contain two known, distinct mention ids")
        normalized.append(sorted((left, right)))
    if len(normalized) != len({tuple(item) for item in normalized}):
        _fail(path, "duplicate pairs")
    return normalized


def _validate_document(value: Any, path: str) -> dict:
    doc = _object(value, path, {"id", "text", "atoms"})
    _string(doc["id"], f"{path}.id")
    if not isinstance(doc["text"], str):
        _fail(f"{path}.text", "expected string")
    _string_list(doc["atoms"], f"{path}.atoms")
    return doc


def validate_gold(payload: Any) -> dict:
    root = _object(
        payload,
        "$",
        {"schema_version", "benchmark_id", "benchmark_version", "description", "top_k", "residual_tolerance", "cases"},
    )
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", f"expected {SCHEMA_VERSION!r}")
    _string(root["benchmark_id"], "$.benchmark_id")
    _string(root["description"], "$.description")
    if type(root["benchmark_version"]) is not int or root["benchmark_version"] < 1:
        _fail("$.benchmark_version", "expected positive integer")
    if type(root["top_k"]) is not int or not 1 <= root["top_k"] <= 16:
        _fail("$.top_k", "expected integer in [1, 16]")
    if not isinstance(root["residual_tolerance"], (int, float)) or not math.isfinite(root["residual_tolerance"]) or root["residual_tolerance"] <= 0:
        _fail("$.residual_tolerance", "expected finite positive number")
    if not isinstance(root["cases"], list) or not root["cases"]:
        _fail("$.cases", "expected non-empty list")

    case_ids: set[str] = set()
    for index, raw_case in enumerate(root["cases"]):
        path = f"$.cases[{index}]"
        case = _object(raw_case, path, {"id", "component", "tags", "documents", "query", "gold"}, {"id", "component", "tags", "documents", "gold"})
        case_id = _string(case["id"], f"{path}.id")
        if case_id in case_ids:
            _fail(f"{path}.id", "duplicate case id")
        case_ids.add(case_id)
        component = case["component"]
        if component not in {"spans", "identity", "exemplar", "weave"}:
            _fail(f"{path}.component", "unknown component")
        tags = _string_list(case["tags"], f"{path}.tags")
        if not isinstance(case["documents"], list) or not case["documents"]:
            _fail(f"{path}.documents", "expected non-empty list")
        documents = [_validate_document(item, f"{path}.documents[{i}]") for i, item in enumerate(case["documents"])]
        doc_ids = [item["id"] for item in documents]
        if len(doc_ids) != len(set(doc_ids)):
            _fail(f"{path}.documents", "duplicate document id")
        document_senfs = [extract_senf(doc["id"], doc["text"], doc["atoms"]) for doc in documents]
        known_mentions = {
            mention.mention_id
            for senf in document_senfs
            for mention in senf.mentions
        }
        all_pairs = {
            tuple(sorted((left, right)))
            for left in known_mentions
            for right in known_mentions
            if left < right
        }
        gold_path = f"{path}.gold"
        if component == "spans":
            gold = _object(case["gold"], gold_path, {"mentions"})
            if not isinstance(gold["mentions"], list):
                _fail(f"{gold_path}.mentions", "expected list")
            seen = set()
            by_id = {doc["id"]: doc["text"] for doc in documents}
            for mention_index, item in enumerate(gold["mentions"]):
                item_path = f"{gold_path}.mentions[{mention_index}]"
                if not isinstance(item, list) or len(item) != 5:
                    _fail(item_path, "expected [document, symbol, surface, start, end]")
                doc_id, symbol, surface, start, end = item
                if doc_id not in by_id or not isinstance(symbol, str) or not symbol or not isinstance(surface, str):
                    _fail(item_path, "invalid document, symbol, or surface")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(by_id[doc_id]):
                    _fail(item_path, "invalid half-open span")
                if by_id[doc_id][start:end] != surface:
                    _fail(item_path, "surface does not equal text at span")
                if tuple(item) in seen:
                    _fail(item_path, "duplicate mention")
                seen.add(tuple(item))
        elif component == "identity":
            gold = _object(case["gold"], gold_path, {"merge_pairs", "non_merge_pairs", "abstain_pairs", "id_plus", "id_minus"})
            merge = _pairs(gold["merge_pairs"], f"{gold_path}.merge_pairs", known_mentions)
            non_merge = _pairs(gold["non_merge_pairs"], f"{gold_path}.non_merge_pairs", known_mentions)
            abstain = _pairs(gold["abstain_pairs"], f"{gold_path}.abstain_pairs", known_mentions)
            plus = _pairs(gold["id_plus"], f"{gold_path}.id_plus", known_mentions)
            minus = _pairs(gold["id_minus"], f"{gold_path}.id_minus", known_mentions)
            if set(map(tuple, merge)) & set(map(tuple, abstain)):
                _fail(gold_path, "a pair cannot be both merge and abstain")
            merge_set, non_merge_set = set(map(tuple, merge)), set(map(tuple, non_merge))
            if merge_set & non_merge_set or merge_set | non_merge_set != all_pairs:
                _fail(gold_path, "merge_pairs and non_merge_pairs must partition every mention pair")
            if not set(map(tuple, abstain)) <= non_merge_set:
                _fail(f"{gold_path}.abstain_pairs", "abstentions must be non-merge pairs")
            if not set(map(tuple, plus)) <= all_pairs or not set(map(tuple, minus)) <= all_pairs:
                _fail(gold_path, "identity evidence must use the explicit pair universe")
        elif component == "exemplar":
            gold = _object(case["gold"], gold_path, {"assignments"})
            if not isinstance(gold["assignments"], list) or not gold["assignments"]:
                _fail(f"{gold_path}.assignments", "expected non-empty list")
            assigned_mentions = set()
            for assignment_index, item in enumerate(gold["assignments"]):
                if not isinstance(item, list) or len(item) != 2 or item[0] not in known_mentions:
                    _fail(f"{gold_path}.assignments[{assignment_index}]", "expected [known mention id, exemplar]")
                _string(item[1], f"{gold_path}.assignments[{assignment_index}][1]")
                if item[0] in assigned_mentions:
                    _fail(f"{gold_path}.assignments[{assignment_index}]", "duplicate mention assignment")
                assigned_mentions.add(item[0])
        else:
            if "query" not in case:
                _fail(path, "weave case requires query")
            query = _validate_document(case["query"], f"{path}.query")
            if query["id"] in doc_ids:
                _fail(f"{path}.query.id", "query id duplicates a document id")
            gold = _object(case["gold"], gold_path, {"top_k_guards", "mappings", "bridge", "adapters"})
            guards = _formatted_strings(gold["top_k_guards"], f"{gold_path}.top_k_guards", GUARD_RE)
            expected_guard_suffix = f"->{query['id']}"
            if any(not guard.endswith(expected_guard_suffix) for guard in guards):
                _fail(f"{gold_path}.top_k_guards", "guard target must equal the query id")
            for guard in guards:
                sources_part = guard.split("->", 1)[0].split("+")
                if len(sources_part) != len(set(sources_part)) or not set(sources_part) <= set(doc_ids):
                    _fail(f"{gold_path}.top_k_guards", "guard sources must be unique document ids")
            query_senf = extract_senf(query["id"], query["text"], query["atoms"])
            source_frames = {frame.frame_id: (frame, senf) for senf in document_senfs for frame in senf.frames}
            query_frames = {frame.frame_id: frame for frame in query_senf.frames}
            query_mentions = {mention.mention_id for mention in query_senf.mentions}
            mappings = _object(gold["mappings"], f"{gold_path}.mappings", set(MAPPING_TYPES))
            for mapping_type in MAPPING_TYPES:
                _formatted_strings(mappings[mapping_type], f"{gold_path}.mappings.{mapping_type}", MAPPING_RE)
            for mapping in mappings["frame"]:
                left, right = mapping.split("->", 1)
                if left not in source_frames or right not in query_frames:
                    _fail(f"{gold_path}.mappings.frame", "frame mapping must reference known source/query frames")
            for mapping in mappings["entity"]:
                left, right = mapping.split("->", 1)
                if left not in known_mentions or right not in query_mentions:
                    _fail(f"{gold_path}.mappings.entity", "entity mapping must reference known source/query mentions")
            if gold["bridge"] not in {"allow", "deny", "none"}:
                _fail(f"{gold_path}.bridge", "expected allow, deny, or none")
            if not isinstance(gold["adapters"], list):
                _fail(f"{gold_path}.adapters", "expected list")
            adapter_keys = set()
            for adapter_index, adapter in enumerate(gold["adapters"]):
                adapter_path = f"{gold_path}.adapters[{adapter_index}]"
                adapter = _object(adapter, adapter_path, {
                    "source_frame", "query_frame", "source_predicate", "query_predicate",
                    "source_arguments", "query_arguments", "atom",
                })
                for name in ("source_frame", "query_frame", "source_predicate", "query_predicate", "atom"):
                    _string(adapter[name], f"{adapter_path}.{name}")
                _strings(adapter["source_arguments"], f"{adapter_path}.source_arguments")
                _strings(adapter["query_arguments"], f"{adapter_path}.query_arguments")
                if adapter["source_frame"] not in source_frames:
                    _fail(f"{adapter_path}.source_frame", "unknown source frame")
                if adapter["query_frame"] not in query_frames:
                    _fail(f"{adapter_path}.query_frame", "unknown query frame")
                source_frame, source_senf = source_frames[adapter["source_frame"]]
                query_frame = query_frames[adapter["query_frame"]]
                if adapter["source_predicate"] != source_frame.predicate_head or adapter["source_arguments"] != list(_arguments(source_frame, source_senf) or ()):
                    _fail(adapter_path, "source predicate or arguments do not match the source frame")
                if adapter["query_predicate"] != query_frame.predicate_head or adapter["query_arguments"] != list(_arguments(query_frame, query_senf) or ()):
                    _fail(adapter_path, "query predicate or arguments do not match the query frame")
                atom_match = ADAPTER_ATOM_RE.fullmatch(adapter["atom"])
                if atom_match is None or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in atom_match.groups()):
                    _fail(f"{adapter_path}.atom", "invalid executable adapter atom or truth value")
                source_call = f"({adapter['source_predicate']} {' '.join(adapter['source_arguments'])})"
                query_call = f"({adapter['query_predicate']} {' '.join(adapter['query_arguments'])})"
                if f"(Premises {source_call})" not in adapter["atom"] or f"(Conclusions {query_call})" not in adapter["atom"]:
                    _fail(f"{adapter_path}.atom", "adapter atom does not match its semantic signature")
                key = json.dumps(adapter, sort_keys=True)
                if key in adapter_keys:
                    _fail(adapter_path, "duplicate adapter")
                adapter_keys.add(key)
            if gold["bridge"] == "allow" and not gold["adapters"]:
                _fail(f"{gold_path}.adapters", "allow requires at least one exact adapter")
            if gold["bridge"] != "allow" and gold["adapters"]:
                _fail(f"{gold_path}.adapters", "only allow cases may contain adapters")

    components = {case["component"] for case in root["cases"]}
    if components != {"spans", "identity", "exemplar", "weave"}:
        _fail("$.cases", "must contain spans, identity, exemplar, and weave records")
    if not any(case["component"] == "identity" and case["gold"]["abstain_pairs"] for case in root["cases"]):
        _fail("$.cases", "identity coverage requires an actual abstention")
    if not any(case["component"] == "identity" and case["gold"]["id_minus"] for case in root["cases"]):
        _fail("$.cases", "identity coverage requires actual negative evidence")
    weave_cases = [case for case in root["cases"] if case["component"] == "weave"]
    if not any(case["gold"]["bridge"] == "allow" and case["gold"]["adapters"] for case in weave_cases):
        _fail("$.cases", "weave coverage requires an exact allowed adapter")
    if not any(case["gold"]["bridge"] == "deny" for case in weave_cases):
        _fail("$.cases", "weave coverage requires a denied bridge")
    if not any(len(case["documents"]) > 1 and case["gold"]["top_k_guards"] for case in weave_cases):
        _fail("$.cases", "weave coverage requires a successful multi-source case")
    if any(not any(case["gold"]["mappings"][name] for case in weave_cases) for name in MAPPING_TYPES):
        _fail("$.cases", "weave coverage requires every typed mapping category")
    return root


def load_gold(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldSchemaError(f"{path}: {exc}") from exc
    return validate_gold(payload)


@contextmanager
def _benchmark_settings() -> Iterator[None]:
    settings = get_settings()
    pinned = COMPONENT_SETTINGS
    previous = {name: getattr(settings, name) for name in pinned}
    for name, value in pinned.items():
        setattr(settings, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


def _senfs(case: dict) -> list:
    return [score_exemplars(extract_senf(doc["id"], doc["text"], doc["atoms"])) for doc in case["documents"]]


def _pair(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def _context_value(frame, name: str) -> str | None:
    return getattr(frame, name, None) or (getattr(frame.context, name, None) if frame.context else None)


def _mapping_sets(result, sources: list, query) -> dict[str, list[str]]:
    source_by_senf = {item.senf_id: item for item in sources}
    source_by_frame = {
        frame.frame_id: senf
        for senf in sources
        for frame in senf.frames
    }
    query_frames = {frame.frame_id: frame for frame in query.frames}
    mappings: dict[str, set[str]] = {name: set() for name in MAPPING_TYPES}
    for pair in result.pairs:
        source = source_by_senf.get(pair.source_id) or source_by_frame.get(pair.source_frame_id)
        source_frame = next((frame for frame in source.frames if frame.frame_id == pair.source_frame_id), None) if source else None
        query_frame = query_frames.get(pair.query_frame_id)
        if source_frame is None or query_frame is None:
            continue
        mappings["frame"].add(f"{pair.source_frame_id}->{pair.query_frame_id}")
        for name in ("time_ref", "location_ref"):
            left, right = _context_value(source_frame, name), _context_value(query_frame, name)
            if left is not None and right is not None:
                mappings["time" if name == "time_ref" else "location"].add(f"{left}->{right}")
    for item in result.entity_maps:
        mappings["entity"].add(f"{item.source_mention_id}->{item.target_mention_id}")
    mappings["role"].update(f"{left}->{right}" for left, right in result.role_maps)
    mappings["exemplar"].update(f"{left}->{right}" for left, right in result.exemplar_maps)
    return {name: sorted(values) for name, values in mappings.items()}


def _run_case(case: dict, top_k: int) -> dict:
    senfs = _senfs(case)
    component = case["component"]
    if component == "spans":
        mentions = sorted(
            [senf.sentence_id, mention.canonical_symbol, mention.surface, mention.char_span[0], mention.char_span[1]]
            for senf in senfs
            for mention in senf.mentions
            if mention.char_span is not None
        )
        return {"mentions": mentions}
    if component == "identity":
        graph = resolve_identity(senfs)
        return {
            "merge_pairs": sorted(_pair(*edge.mention_ids) for edge in graph.merged),
            "id_plus": sorted(_pair(*edge.mention_ids) for edge in graph.edges if edge.strength > 0),
            "id_minus": sorted(_pair(*edge.mention_ids) for edge in graph.edges if edge.negative_strength > 0),
        }
    if component == "exemplar":
        return {
            "assignments": sorted(
                [mention.mention_id, exemplar]
                for senf in senfs
                for mention in senf.mentions
                if (exemplar := senf.nearest_exemplar_for(mention)) is not None
            )
        }

    query_doc = case["query"]
    query = score_exemplars(extract_senf(query_doc["id"], query_doc["text"], query_doc["atoms"]))
    graph = resolve_identity([*senfs, query])
    best = weave(query, senfs, identity_graph=graph)
    ranked = build_weaves(query, senfs, k=top_k, identity_graph=graph)
    bridges = executable_bridge_atoms(best, senfs, query, graph)
    source_by_senf = {senf.senf_id: senf for senf in senfs}
    source_frames = {frame.frame_id: frame for senf in senfs for frame in senf.frames}
    query_frames = {frame.frame_id: frame for frame in query.frames}
    adapters = []
    for atom in bridges:
        match = re.search(r"\(: senf_adapter_(\d+)_", atom)
        if match is None or int(match.group(1)) >= len(best.pairs):
            continue
        pair = best.pairs[int(match.group(1))]
        source_frame, query_frame = source_frames[pair.source_frame_id], query_frames[pair.query_frame_id]
        source_args = list(_arguments(source_frame, source_by_senf.get(pair.source_id) or next(item for item in senfs if source_frame in item.frames)) or ())
        query_args = list(_arguments(query_frame, query) or ())
        adapters.append({
            "source_frame": pair.source_frame_id,
            "query_frame": pair.query_frame_id,
            "source_predicate": source_frame.predicate_head,
            "query_predicate": query_frame.predicate_head,
            "source_arguments": source_args,
            "query_arguments": query_args,
            "atom": atom,
        })
    return {
        "top_k_guards": [item.guard for item in ranked[:top_k] if item.aligned],
        "mappings": _mapping_sets(best, senfs, query),
        "adapters": adapters,
        "residuals": list(best.residuals) if best.residuals is not None else [],
        "polish_converged": best.polish_converged,
        "guard": best.guard,
    }


def _prf(expected: set[str], predicted: set[str]) -> dict[str, Any]:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _add(bucket: dict[str, set[str]], name: str, side: str, case_id: str, values) -> None:
    bucket.setdefault(f"{name}:{side}", set()).update(f"{case_id}:{value}" for value in values)


def evaluate(payload: dict, dataset_sha256: str | None = None) -> dict:
    validate_gold(payload)
    top_k = payload["top_k"]
    buckets: dict[str, set[str]] = {}
    rows = []
    false_merges = forbidden_merges = false_bridge_cases = denied_bridges = 0
    top_k_hits = wrong_guard_count = extra_guard_count = 0
    residuals: list[float] = []
    converged = residual_runs = 0
    deterministic = True

    with _benchmark_settings():
        for case in payload["cases"]:
            first = _run_case(case, top_k)
            second = _run_case(case, top_k)
            stable = first == second
            deterministic &= stable
            component, gold, case_id = case["component"], case["gold"], case["id"]
            row: dict[str, Any] = {"id": case_id, "component": component, "deterministic": stable}
            if component == "spans":
                expected = {json.dumps(item, separators=(",", ":")) for item in gold["mentions"]}
                predicted = {json.dumps(item, separators=(",", ":")) for item in first["mentions"]}
                _add(buckets, "spans", "expected", case_id, expected)
                _add(buckets, "spans", "predicted", case_id, predicted)
                row["exact"] = expected == predicted
            elif component == "identity":
                expected_merge = {_pair(*item) for item in gold["merge_pairs"]}
                expected_non_merge = {_pair(*item) for item in gold["non_merge_pairs"]}
                expected_plus = {_pair(*item) for item in gold["id_plus"]}
                expected_minus = {_pair(*item) for item in gold["id_minus"]}
                predicted_merge = set(first["merge_pairs"])
                predicted_plus = set(first["id_plus"])
                predicted_minus = set(first["id_minus"])
                for name, expected, predicted in (
                    ("identity_merge", expected_merge, predicted_merge),
                    ("id_plus", expected_plus, predicted_plus),
                    ("id_minus", expected_minus, predicted_minus),
                ):
                    _add(buckets, name, "expected", case_id, expected)
                    _add(buckets, name, "predicted", case_id, predicted)
                violations = predicted_merge & expected_non_merge
                false_merges += len(violations)
                forbidden_merges += len(expected_non_merge)
                row.update({"false_merges": len(violations), "merge_pairs": first["merge_pairs"]})
            elif component == "exemplar":
                expected = {"|".join(item) for item in gold["assignments"]}
                predicted = {"|".join(item) for item in first["assignments"]}
                _add(buckets, "exemplar", "expected", case_id, expected)
                _add(buckets, "exemplar", "predicted", case_id, predicted)
                row["exact"] = expected == predicted
            else:
                expected_guards = set(gold["top_k_guards"])
                predicted_guards = set(first["top_k_guards"])
                hit = bool(expected_guards & predicted_guards) if expected_guards else not predicted_guards
                wrong_guards = predicted_guards - expected_guards
                extra_guards = wrong_guards if hit else set()
                top_k_hits += int(hit)
                wrong_guard_count += len(wrong_guards)
                extra_guard_count += len(extra_guards)
                row.update({"top_k_hit": hit, "wrong_guards": sorted(wrong_guards), "extra_guards": sorted(extra_guards)})
                for mapping_type in MAPPING_TYPES:
                    expected = set(gold["mappings"][mapping_type])
                    predicted = set(first["mappings"][mapping_type])
                    _add(buckets, f"mapping_{mapping_type}", "expected", case_id, expected)
                    _add(buckets, f"mapping_{mapping_type}", "predicted", case_id, predicted)
                bridge = gold["bridge"]
                expected_adapters = {json.dumps(item, sort_keys=True) for item in gold["adapters"]}
                predicted_adapters = {json.dumps(item, sort_keys=True) for item in first["adapters"]}
                _add(buckets, "bridge_adapter", "expected", case_id, expected_adapters)
                _add(buckets, "bridge_adapter", "predicted", case_id, predicted_adapters)
                if bridge == "deny":
                    denied_bridges += 1
                    false_bridge_cases += int(bool(predicted_adapters))
                if first["residuals"]:
                    residual_runs += 1
                    residuals.extend(first["residuals"])
                    converged += int(first["polish_converged"] is True and max(first["residuals"]) <= payload["residual_tolerance"])
                row.update({"guard": first["guard"], "adapter_count": len(first["adapters"]), "max_residual": max(first["residuals"], default=None)})
            rows.append(row)

    metric_names = sorted({key.rsplit(":", 1)[0] for key in buckets})
    metrics = {
        name: _prf(buckets.get(f"{name}:expected", set()), buckets.get(f"{name}:predicted", set()))
        for name in metric_names
    }
    mapping_expected = set().union(*(buckets.get(f"mapping_{name}:expected", set()) for name in MAPPING_TYPES))
    mapping_predicted = set().union(*(buckets.get(f"mapping_{name}:predicted", set()) for name in MAPPING_TYPES))
    metrics["typed_mappings_micro"] = _prf(mapping_expected, mapping_predicted)
    metrics["top_k"] = {
        "k": top_k, "cases": len([row for row in rows if row["component"] == "weave"]),
        "hits": top_k_hits, "hit_rate": round(top_k_hits / len([row for row in rows if row["component"] == "weave"]), 6),
        "wrong_guard_count": wrong_guard_count, "extra_prediction_count": extra_guard_count,
    }
    metrics["false_merge_rate"] = round(false_merges / forbidden_merges, 6) if forbidden_merges else 0.0
    metrics["false_bridge_rate"] = round(false_bridge_cases / denied_bridges, 6) if denied_bridges else 0.0
    metrics["residual"] = {
        "runs": residual_runs,
        "converged": converged,
        "convergence_rate": round(converged / residual_runs, 6) if residual_runs else 1.0,
        "max": max(residuals, default=0.0),
        "mean": round(sum(residuals) / len(residuals), 12) if residuals else 0.0,
        "tolerance": payload["residual_tolerance"],
    }
    metrics["determinism"] = {"cases": len(rows), "stable": sum(row["deterministic"] for row in rows), "rate": round(sum(row["deterministic"] for row in rows) / len(rows), 6)}
    return {
        "schema_version": "senf-components-report/v2",
        "benchmark_id": payload["benchmark_id"],
        "benchmark_version": payload["benchmark_version"],
        "case_count": len(rows),
        "metrics": metrics,
        "cases": rows,
        "reproducibility": _reproducibility(payload, dataset_sha256),
    }


def _reproducibility(payload: dict, dataset_sha256: str | None = None) -> dict[str, Any]:
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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "command": "python scripts/evaluate_senf_components.py --gold data/benchmarks/senf_components_v1.json",
        "dataset_sha256": dataset_sha256 or hashlib.sha256(canonical).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": revision,
        "settings": dict(sorted(COMPONENT_SETTINGS.items())),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
            "dependencies": dependencies,
        },
        "determinism_scope": "two sequential runs in the same process",
    }


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        f"# SENF Component Benchmark v{report['benchmark_version']}",
        "",
        f"Cases: **{report['case_count']}**  ",
        f"Schema: `{report['schema_version']}`",
        "",
        "## Classification And Mapping Metrics",
        "",
        "| Metric | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        if isinstance(value, dict) and "precision" in value:
            lines.append(f"| {name} | {value['precision']:.6f} | {value['recall']:.6f} | {value['f1']:.6f} | {value['tp']} | {value['fp']} | {value['fn']} |")
    lines.extend([
        "",
        "## Safety And Numerical Metrics",
        "",
        f"- Hit@{metrics['top_k']['k']}: **{metrics['top_k']['hits']}/{metrics['top_k']['cases']}** ({metrics['top_k']['hit_rate']:.6f})",
        f"- Wrong guards: **{metrics['top_k']['wrong_guard_count']}**; extras alongside a hit: **{metrics['top_k']['extra_prediction_count']}**",
        f"- False-merge rate: **{metrics['false_merge_rate']:.6f}**",
        f"- False-bridge rate: **{metrics['false_bridge_rate']:.6f}**",
        f"- Exact adapter F1: **{metrics['bridge_adapter']['f1']:.6f}**",
        f"- Residual convergence: **{metrics['residual']['converged']}/{metrics['residual']['runs']}** (max `{metrics['residual']['max']:.3g}`, tolerance `{metrics['residual']['tolerance']:.3g}`)",
        f"- Determinism: **{metrics['determinism']['stable']}/{metrics['determinism']['cases']}**",
        "",
        "## Cases",
        "",
        "| Case | Component | Deterministic | Result |",
        "|---|---|---:|---|",
    ])
    for row in report["cases"]:
        result = "top-k hit" if row.get("top_k_hit") else "exact" if row.get("exact") else row.get("guard") or ("no false merge" if not row.get("false_merges") else "false merge")
        lines.append(f"| {row['id']} | {row['component']} | {'yes' if row['deterministic'] else 'no'} | {result} |")
    repro = report["reproducibility"]
    lines.extend(["", "## Reproducibility", "", f"- Command: `{repro['command']}`", f"- Dataset SHA-256: `{repro['dataset_sha256']}`", f"- Generator SHA-256: `{repro['generator_sha256']}`", f"- Git revision: `{repro['git_revision']}`", f"- Determinism scope: {repro['determinism_scope']}", "", "```json", json.dumps({"settings": repro["settings"], "environment": repro["environment"]}, indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Versioned gold JSON")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Write output instead of stdout")
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_gold(args.gold), hashlib.sha256(args.gold.read_bytes()).hexdigest())
    except GoldSchemaError as exc:
        parser.error(str(exc))
    command = ["python", "scripts/evaluate_senf_components.py", "--gold", str(args.gold), "--format", args.format]
    if args.output:
        command.extend(("--output", str(args.output)))
    report["reproducibility"]["command"] = shlex.join(command)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else render_markdown(report)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
