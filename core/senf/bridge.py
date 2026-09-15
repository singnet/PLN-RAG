import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from core.senf.identity import IdentityEdge
from core.senf.types import (
    ACTUAL_BRANCH_ID,
    EntityRef,
    FrameRef,
    KindRef,
    SENF,
    SENFFrame,
    ValueRef,
)
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
        left_id, right_id = edge.mention_ids
        atoms.append(
            f"(: senf_identity_{index}_{_safe(left_id)}_{_safe(right_id)} "
            f"(MentionIdentity {_safe(left_id)} {_safe(right_id)}) "
            f"(STV {tv.strength} {tv.weight}))"
        )
    return atoms


def weave_bridge_atoms(weave: WeaveResult) -> list[str]:
    atoms: list[str] = []
    for index, mapping in enumerate(weave.entity_maps):
        tv = transport_truth(1.0, 1.0, weave.total_cost + mapping.cost)
        atoms.append(
            f"(: senf_weave_{index}_{_safe(mapping.source_entity_id)}_{_safe(mapping.target_entity_id)} "
            f"(EntityAlignment {_safe(mapping.source_entity_id)} {_safe(mapping.target_entity_id)}) "
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


def _frame(senf: SENF, frame_id: str) -> Optional[SENFFrame]:
    return next((frame for frame in senf.frames if frame.frame_id == frame_id), None)


def _arguments(frame: SENFFrame, senf: SENF) -> Optional[list[str]]:
    arguments: list[str] = []
    for role in sorted(frame.roles, key=lambda item: item.position):
        filler = role.filler
        if isinstance(filler, EntityRef):
            entity = senf.entity(filler.entity_id)
            arguments.append(entity.canonical_symbol if entity else filler.entity_id)
        elif isinstance(filler, KindRef):
            arguments.append(filler.canonical_symbol)
        elif isinstance(filler, ValueRef):
            arguments.append(filler.value)
        elif isinstance(filler, FrameRef):
            return None
    return arguments


def _root_frame_ids(senf: SENF) -> set[str]:
    nested = {
        role.filler.frame_id
        for frame in senf.frames
        for role in frame.roles
        if isinstance(role.filler, FrameRef)
    }
    return {frame.frame_id for frame in senf.frames if frame.frame_id not in nested}


def _unguarded_context(frame: SENFFrame) -> tuple[Optional[str], ...]:
    context = frame.context
    speaker = context.speaker if context else None
    modality = (context.modality if context else None) or frame.modality
    time_ref = (context.time_ref if context else None) or frame.time_ref
    location_ref = (context.location_ref if context else None) or frame.location_ref
    if frame.predicate_head == "AtTime":
        time_ref = None
    if frame.predicate_head in {"AtLocation", "LocatedAt", "LocatedIn"}:
        location_ref = None
    return speaker, modality, time_ref, location_ref


def _contexts_allow_executable_bridge(
    source_frame: SENFFrame, query_frame: SENFFrame
) -> bool:
    source_context = _unguarded_context(source_frame)
    query_context = _unguarded_context(query_frame)
    return (
        source_context[:2] == (None, None)
        and query_context[:2] == (None, None)
        and source_context == query_context
    )


def executable_bridge_atoms(
    weave: WeaveResult,
    source: SENF,
    query: SENF,
    identity_graph=None,
) -> list[str]:
    """Build query-local source-to-query rules for selected frame mappings.

    The rules are concrete by design. This keeps adapters for multiple candidates
    isolated when the parser returns one shared transient query context.
    """
    atoms: list[str] = []
    source_roots = _root_frame_ids(source)
    query_roots = _root_frame_ids(query)
    for index, pair in enumerate(weave.pairs):
        decision = (
            weave.transport_decisions[index]
            if index < len(weave.transport_decisions)
            else None
        )
        if decision is not None and not decision.allowed:
            continue
        source_frame = _frame(source, pair.source_frame_id)
        query_frame = _frame(query, pair.query_frame_id)
        if source_frame is None or query_frame is None:
            continue
        source_branch = (
            source_frame.context.branch_id
            if source_frame.context else ACTUAL_BRANCH_ID
        )
        query_branch = (
            query_frame.context.branch_id
            if query_frame.context else ACTUAL_BRANCH_ID
        )
        if (
            decision is None
            and (source_branch != ACTUAL_BRANCH_ID or query_branch != ACTUAL_BRANCH_ID)
        ):
            continue
        if (
            source_frame.frame_id not in source_roots
            or query_frame.frame_id not in query_roots
            or source_frame.clause_role != "fact"
            or query_frame.clause_role != "fact"
            or not source_frame.source_atom_id
            or source_frame.polarity != query_frame.polarity
            or not _contexts_allow_executable_bridge(source_frame, query_frame)
        ):
            continue
        source_args = _arguments(source_frame, source)
        query_args = _arguments(query_frame, query)
        if source_args is None or query_args is None or len(source_args) != len(query_args):
            continue

        justified = True
        identity_cost = 0.0
        source_entity_ids = {
            role.filler.entity_id
            for role in source_frame.roles
            if isinstance(role.filler, EntityRef)
        }
        query_entity_ids = {
            role.filler.entity_id
            for role in query_frame.roles
            if isinstance(role.filler, EntityRef)
        }
        for mapping in weave.entity_maps:
            if (
                mapping.source_entity_id not in source_entity_ids
                or mapping.target_entity_id not in query_entity_ids
            ):
                continue
            edge = next(
                (
                    item
                    for item in identity_graph.edges
                    if {mapping.source_mention_id, mapping.target_mention_id}
                    == set(item.mention_ids)
                ),
                None,
            ) if identity_graph is not None else None
            if identity_graph is None or not identity_graph.same_entity(
                mapping.source_entity_id, mapping.target_entity_id
            ):
                justified = False
                break
            identity_cost += mapping.cost
        predicate_changed = source_frame.predicate_head != query_frame.predicate_head
        if not justified or (not predicate_changed and source_args == query_args):
            continue

        probability = weave.branch_probability if decision is not None else 1.0
        tv = transport_truth(
            probability, probability, weave.total_cost + identity_cost
        )
        source_body = f"({source_frame.predicate_head} {' '.join(source_args)})"
        query_body = f"({query_frame.predicate_head} {' '.join(query_args)})"
        atoms.append(
            f"(: senf_adapter_{index}_{_safe(pair.source_frame_id)}_{_safe(pair.query_frame_id)} "
            f"(Implication (Premises {source_body}) (Conclusions {query_body})) "
            f"(STV {tv.strength} {tv.weight}))"
        )
    return atoms
