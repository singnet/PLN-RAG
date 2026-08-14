import math
import re
from dataclasses import dataclass
from typing import Iterable

from core.senf.identity import IdentityEdge
from core.senf.weave import WeaveResult


@dataclass(frozen=True)
class TransportedTruth:
    strength: float
    weight: float


def transport_truth(
    strength: float,
    weight: float,
    cost: float,
    lambda_strength: float = 1.0,
    lambda_weight: float = 1.0,
) -> TransportedTruth:
    return TransportedTruth(
        round(max(0.0, min(1.0, strength * math.exp(-lambda_strength * cost))), 6),
        round(max(0.0, min(1.0, weight / (1.0 + lambda_weight * cost))), 6),
    )


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "mention"


def identity_bridge_atoms(
    edges: Iterable[IdentityEdge], threshold: float = 0.75
) -> list[str]:
    atoms: list[str] = []
    for index, edge in enumerate(edges):
        if edge.strength < threshold or edge.negative_strength >= 0.5:
            continue
        tv = transport_truth(edge.strength, edge.confidence, edge.positive_cost)
        left, right = edge.symbols
        atoms.append(
            f"(: senf_identity_{index}_{_safe(left)}_{_safe(right)} "
            f"(SimilarityLink {left} {right}) (STV {tv.strength} {tv.weight}))"
        )
    return atoms


def weave_bridge_atoms(weave: WeaveResult) -> list[str]:
    atoms: list[str] = []
    for index, mapping in enumerate(weave.entity_maps):
        tv = transport_truth(1.0, 1.0, weave.total_cost + mapping.cost)
        atoms.append(
            f"(: senf_weave_{index}_{_safe(mapping.source_symbol)}_{_safe(mapping.target_symbol)} "
            f"(SimilarityLink {mapping.source_symbol} {mapping.target_symbol}) "
            f"(STV {tv.strength} {tv.weight}))"
        )
    return atoms


def predicate_bridge_atoms(weave: WeaveResult) -> list[str]:
    atoms: list[str] = []
    for index, mapping in enumerate(weave.predicate_maps):
        if mapping.source_head == mapping.query_head:
            continue
        tv = transport_truth(1.0, 1.0, mapping.cost)
        atoms.append(
            f"(: senf_predicate_{index}_{_safe(mapping.source_head)}_{_safe(mapping.query_head)} "
            f"(PredicateBridge {mapping.source_head} {mapping.query_head}) "
            f"(STV {tv.strength} {tv.weight}))"
        )
    return atoms
