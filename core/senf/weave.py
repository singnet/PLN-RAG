from dataclasses import dataclass, replace
import math
from itertools import product
from typing import Optional, Sequence

from config import get_settings
from core.senf.exemplars import exemplar_distance
from core.senf.identity import IdentityEdge, IdentityGraph
from core.senf.temporal import BranchingContextTree, TransportDecision, assess_transport
from core.senf.types import Context, EntityPersistence, EntityRef, FrameRef, KindRef, Mention, Role, SENF, SENFFrame, ValidityInterval, ValueRef
from core.senf.weave_model import (
    FrameAlignment,
    PairComponentCosts,
    PolishDiagnostics,
    SourceFrameKey,
)


MIN_PAIR_SCORE = 0.3

_SAFE_PREDICATE_ALIASES = frozenset({
    frozenset(("AtLocation", "LocatedAt")),
    frozenset(("AtLocation", "LocatedIn")),
    frozenset(("LocatedAt", "LocatedIn")),
    frozenset(("IsA", "InstanceOf")),
})

_INVERSE_PREDICATE_PAIRS = frozenset({
    frozenset(("PartOf", "HasPart")),
    frozenset(("HasProperty", "PropertyOf")),
    frozenset(("UsedFor", "PurposeOf")),
})


@dataclass(frozen=True)
class EntityMap:
    source_entity_id: str
    target_entity_id: str
    source_mention_id: str
    target_mention_id: str
    source_symbol: str
    target_symbol: str
    cost: float
    identity_supported: bool = False
    source_id: str = ""
    source_frame_id: str = ""
    query_frame_id: str = ""


@dataclass(frozen=True)
class PredicateMap:
    source_head: str
    query_head: str
    cost: float
    source_id: str = ""
    source_frame_id: str = ""
    query_frame_id: str = ""


@dataclass(frozen=True)
class FramePair:
    query_frame_id: str
    source_frame_id: str
    score: float
    evidence: tuple[str, ...] = ()
    cost: float = 1.0
    source_id: str = ""
    transport_cost: float = 0.0
    branch_probability: float = 1.0


@dataclass(frozen=True)
class WeaveResult:
    pairs: tuple[FramePair, ...] = ()
    distortion: float = 0.0
    grounded_symbols: frozenset[str] = frozenset()
    grounded_entity_ids: frozenset[str] = frozenset()
    role_signatures: frozenset[tuple[str, str]] = frozenset()
    entity_maps: tuple[EntityMap, ...] = ()
    predicate_maps: tuple[PredicateMap, ...] = ()
    kind_maps: tuple[tuple[str, str], ...] = ()
    exemplar_maps: tuple[tuple[str, str], ...] = ()
    role_maps: tuple[tuple[str, str], ...] = ()
    structural_cost: float = 0.0
    exemplar_cost: float = 0.0
    identity_cost: float = 0.0
    conflict_cost: float = 0.0
    time_cost: float = 0.0
    location_cost: float = 0.0
    modality_cost: float = 0.0
    unmatched_cost: float = 0.0
    distortion_cost: float = 0.0
    branch_cost: float = 0.0
    temporal_decay_cost: float = 0.0
    persistence_cost: float = 0.0
    branch_probability: float = 1.0
    branch_lca: str = "actual_root"
    transport_decisions: tuple[TransportDecision, ...] = ()
    total_cost: float = 0.0
    guard: str = ""
    residuals: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rejected_pairs: tuple[FramePair, ...] = ()
    alignments: tuple[FrameAlignment, ...] = ()
    polish: Optional[PolishDiagnostics] = None

    @property
    def aligned(self) -> bool:
        return bool(self.pairs)


@dataclass(frozen=True)
class _PairHypothesis:
    pair: FramePair
    entity_maps: tuple[EntityMap, ...]
    predicate_maps: tuple[PredicateMap, ...]
    exemplar_maps: tuple[tuple[str, str], ...]
    role_maps: tuple[tuple[str, str], ...]
    structural: float
    exemplar: float
    identity: float
    conflict: float
    time: float
    location: float
    modality: float
    branch: float
    temporal_decay: float
    persistence: float
    transport_decision: TransportDecision

    @property
    def local_cost(self) -> float:
        return (
            self.structural
            + self.exemplar
            + self.identity
            + self.conflict
            + self.time
            + self.location
            + self.modality
            + self.branch
            + self.temporal_decay
            + self.persistence
        )


@dataclass(frozen=True)
class _SearchState:
    selected: tuple[_PairHypothesis, ...] = ()
    used_sources: frozenset[str] = frozenset()
    skipped: int = 0


@dataclass(frozen=True)
class _Limits:
    beam_width: int
    per_source_k: int
    max_frames: int
    max_pair_candidates: int
    max_exemplar_alternatives: int
    max_cost: float


def _limits() -> _Limits:
    settings = get_settings()
    return _Limits(
        max(1, settings.senf_weave_beam_width),
        max(1, settings.senf_weave_per_source_k),
        max(1, settings.senf_weave_max_frames),
        max(1, settings.senf_weave_max_pair_candidates),
        max(1, settings.senf_weave_max_exemplar_alternatives),
        max(0.0, settings.senf_weave_max_cost),
    )


def _head_score(left: str, right: str) -> tuple[float, Optional[str]]:
    if left == right:
        return 0.5, "head_exact"
    return 0.0, None


def _predicate_cost(query_head: str, source_head: str) -> Optional[float]:
    if query_head == source_head:
        return 0.0
    pair = frozenset((query_head, source_head))
    if pair in _INVERSE_PREDICATE_PAIRS:
        return None
    if pair in _SAFE_PREDICATE_ALIASES:
        return 0.25
    return None


def _mention_for(senf: SENF, ref: EntityRef) -> Optional[Mention]:
    return next(
        (
            mention
            for mention in senf.mentions
            if mention.mention_id == ref.mention_id and mention.entity_id == ref.entity_id
        ),
        None,
    )


def _entity_symbol(senf: SENF, entity_id: str) -> str:
    entity = senf.entity(entity_id)
    return entity.canonical_symbol if entity else entity_id


def _entity_kinds(senf: SENF, entity_id: str) -> frozenset[str]:
    return frozenset(
        assertion.kind.canonical_symbol
        for assertion in senf.kind_assertions
        if assertion.entity_id == entity_id and assertion.polarity
    )


def _identity_edge(
    graph: Optional[IdentityGraph], left: Optional[Mention], right: Optional[Mention]
) -> Optional[IdentityEdge]:
    if graph is None or left is None or right is None:
        return None
    keys = {left.mention_id, right.mention_id}
    return next((edge for edge in graph.edges if keys == set(edge.mention_ids)), None)


def _identity_cost(
    graph: Optional[IdentityGraph],
    left: Optional[Mention],
    right: Optional[Mention],
    left_id: str,
    right_id: str,
) -> tuple[float, float, bool]:
    if left_id == right_id:
        return 0.0, 0.0, True
    edge = _identity_edge(graph, left, right)
    conflict = edge.negative_strength if edge else 0.0
    if graph is None:
        return 0.0, conflict, False
    if graph.same_entity(left_id, right_id):
        # A direct accepted edge exposes positive uncertainty separately from its
        # negative evidence. Multi-hop transport uses the graph's bounded path cost.
        cost = edge.positive_cost if edge is not None else graph.transport_cost(left_id, right_id)
        return cost, conflict, True
    if edge is not None and edge.strength > 0.0:
        return edge.positive_cost, conflict, False
    return 1.0, conflict, False


def _frame(senf: SENF, frame_id: str) -> Optional[SENFFrame]:
    return next((frame for frame in senf.frames if frame.frame_id == frame_id), None)


def _frame_ref_cost(
    query_senf: SENF,
    source_senf: SENF,
    query_ref: FrameRef,
    source_ref: FrameRef,
    graph: Optional[IdentityGraph],
    seen: set[tuple[str, str]],
) -> Optional[float]:
    key = (query_ref.frame_id, source_ref.frame_id)
    if key in seen:
        return 0.0
    query_frame = _frame(query_senf, query_ref.frame_id)
    source_frame = _frame(source_senf, source_ref.frame_id)
    if query_frame is None or source_frame is None:
        return None
    predicate_cost = _predicate_cost(query_frame.predicate_head, source_frame.predicate_head)
    if predicate_cost is None or len(query_frame.roles) != len(source_frame.roles):
        return None
    cost = predicate_cost
    next_seen = seen | {key}
    for query_role, source_role in zip(
        sorted(query_frame.roles, key=lambda role: role.position),
        sorted(source_frame.roles, key=lambda role: role.position),
    ):
        filler_cost = _filler_cost(
            query_senf, source_senf, query_role, source_role, graph, next_seen
        )
        if filler_cost is None:
            return None
        cost += filler_cost + (0.0 if query_role.name == source_role.name else 0.25)
    return min(1.0, cost / max(1, len(query_frame.roles)))


def _filler_cost(
    query_senf: SENF,
    source_senf: SENF,
    query_role: Role,
    source_role: Role,
    graph: Optional[IdentityGraph],
    seen: Optional[set[tuple[str, str]]] = None,
) -> Optional[float]:
    query_filler, source_filler = query_role.filler, source_role.filler
    if type(query_filler) is not type(source_filler):
        return None
    if isinstance(query_filler, EntityRef) and isinstance(source_filler, EntityRef):
        query_kinds = _entity_kinds(query_senf, query_filler.entity_id)
        source_kinds = _entity_kinds(source_senf, source_filler.entity_id)
        if query_kinds and source_kinds and query_kinds.isdisjoint(source_kinds):
            return None
        if query_filler.entity_id == source_filler.entity_id:
            return 0.0
        query_mention = _mention_for(query_senf, query_filler)
        source_mention = _mention_for(source_senf, source_filler)
        _, _, supported = _identity_cost(
            graph, query_mention, source_mention, query_filler.entity_id, source_filler.entity_id
        )
        if supported:
            return 0.0
        return 0.1 if _entity_symbol(
            query_senf, query_filler.entity_id
        ) == _entity_symbol(source_senf, source_filler.entity_id) else 0.35
    if isinstance(query_filler, KindRef) and isinstance(source_filler, KindRef):
        return 0.0 if query_filler.canonical_symbol == source_filler.canonical_symbol else None
    if isinstance(query_filler, ValueRef) and isinstance(source_filler, ValueRef):
        if query_filler == source_filler:
            return 0.0
        return 0.35 if query_filler.value_type == source_filler.value_type else None
    if isinstance(query_filler, FrameRef) and isinstance(source_filler, FrameRef):
        return _frame_ref_cost(
            query_senf, source_senf, query_filler, source_filler, graph, seen or set()
        )
    return None


def _context_value(frame: SENFFrame, name: str) -> Optional[str]:
    value = getattr(frame, name, None)
    if value is None and frame.context is not None:
        value = getattr(frame.context, name, None)
    return value


def _mismatch(query_frame: SENFFrame, source_frame: SENFFrame, name: str) -> float:
    query_value = _context_value(query_frame, name)
    source_value = _context_value(source_frame, name)
    return 1.0 if query_value is not None and source_value is not None and query_value != source_value else 0.0


def _exemplar_options(
    query_senf: SENF,
    query_mention: Optional[Mention],
    query_frame: SENFFrame,
    source_senf: SENF,
    source_mention: Optional[Mention],
    source_frame: SENFFrame,
    cap: int,
) -> tuple[tuple[Optional[str], Optional[str], float], ...]:
    if query_mention is None or source_mention is None:
        return ((None, None, 0.0),)
    def active_for(senf: SENF, mention: Mention, frame: SENFFrame) -> list[str]:
        context = frame.context
        alternatives = senf.exemplar_scores.get(mention.mention_id, ())
        if not alternatives:
            return sorted(senf.active_exemplars_for(mention))[:cap]
        selected = []
        for alternative in alternatives:
            if alternative.exemplar not in senf.active_exemplars_for(mention):
                continue
            guard = alternative.guard
            checks = (
                ("source_unit_ids", context.source_unit_id if context else mention.source_unit_id),
                ("sentence_ids", mention.sentence_id),
                ("speakers", context.speaker if context else None),
                ("modalities", context.modality if context else frame.modality),
                ("time_refs", context.time_ref if context else frame.time_ref),
                ("location_refs", context.location_ref if context else frame.location_ref),
                ("branch_ids", context.branch_id if context else None),
                ("validity_interval_ids", context.validity_interval_id if context else None),
            )
            if guard is None or all(
                not getattr(guard, name) or value in getattr(guard, name)
                for name, value in checks
            ):
                selected.append(alternative.exemplar)
        return sorted(set(selected))[:cap]

    query_active = active_for(query_senf, query_mention, query_frame)
    source_active = active_for(source_senf, source_mention, source_frame)
    if not query_active and not source_active:
        return ((None, None, 0.0),)
    if not query_active or not source_active:
        return ((None, None, 0.5 * exemplar_distance(
            query_senf, query_mention, source_senf, source_mention
        )),)
    options = []
    for source_exemplar, query_exemplar in product(source_active, query_active):
        if source_exemplar == query_exemplar:
            distance = 0.0
        elif source_exemplar.startswith("generic_") or query_exemplar.startswith("generic_"):
            distance = 0.35
        else:
            distance = 0.8
        options.append((source_exemplar, query_exemplar, 0.5 * distance))
    return tuple(sorted(options, key=lambda item: (item[2], item[0] or "", item[1] or "")))


def _pair_hypotheses(
    query_senf: SENF,
    source_senf: SENF,
    query_frame: SENFFrame,
    source_frame: SENFFrame,
    graph: Optional[IdentityGraph],
    limits: _Limits,
    tree: BranchingContextTree,
    intervals: dict[str, ValidityInterval],
) -> tuple[_PairHypothesis, ...]:
    source_context = source_frame.context or Context(source_senf.source_units[0].source_unit_id)
    query_context = query_frame.context or Context(query_senf.source_units[0].source_unit_id)
    persistence_policies = _frame_persistence(source_frame, source_senf)
    if any(
        item.validity_interval_id is not None
        and item.validity_interval_id != source_context.validity_interval_id
        for item in persistence_policies
    ):
        return ()
    transport_decisions = [
        assess_transport(
            tree, source_context, query_context, intervals, persistence,
            temporal_decay_rate=max(0.0, get_settings().senf_temporal_decay_rate),
            purpose="fact",
        )
        for persistence in (persistence_policies or (None,))
    ]
    denied = next(
        (decision for decision in transport_decisions if not decision.allowed), None
    )
    if denied is not None:
        return ()
    transport = TransportDecision(
        True,
        "allowed",
        source_context.branch_id,
        query_context.branch_id,
        transport_decisions[0].branch_lca,
        min(item.branch_probability for item in transport_decisions),
        transport_decisions[0].interval_relation,
        min(item.temporal_decay for item in transport_decisions),
        max(item.branch_cost for item in transport_decisions),
        max(item.temporal_cost for item in transport_decisions),
        max(item.persistence_cost for item in transport_decisions),
    )
    predicate_cost = _predicate_cost(query_frame.predicate_head, source_frame.predicate_head)
    if predicate_cost is None or len(query_frame.roles) != len(source_frame.roles):
        return ()
    score, head_reason = _head_score(query_frame.predicate_head, source_frame.predicate_head)
    evidence = [head_reason] if head_reason else []
    if score == 0.0:
        score = 0.3
        evidence.append("predicate_bridge")

    structural = predicate_cost
    identity = conflict = 0.0
    base_entities: list[EntityMap] = []
    role_maps: list[tuple[str, str]] = []
    exemplar_choices: list[tuple[tuple[Optional[str], Optional[str], float], ...]] = []
    query_roles = sorted(query_frame.roles, key=lambda role: role.position)
    source_roles = sorted(source_frame.roles, key=lambda role: role.position)
    entity_role_count = sum(isinstance(role.filler, EntityRef) for role in query_roles)
    query_symbols = {
        _entity_symbol(query_senf, role.filler.entity_id)
        for role in query_roles if isinstance(role.filler, EntityRef)
    }
    source_symbols = {
        _entity_symbol(source_senf, role.filler.entity_id)
        for role in source_roles if isinstance(role.filler, EntityRef)
    }

    for query_role, source_role in zip(query_roles, source_roles):
        filler_cost = _filler_cost(query_senf, source_senf, query_role, source_role, graph)
        if filler_cost is None:
            return ()
        structural += filler_cost + (0.0 if query_role.name == source_role.name else 0.25)
        role_maps.append((source_role.name, query_role.name))
        if isinstance(query_role.filler, EntityRef) and isinstance(source_role.filler, EntityRef):
            query_symbol = _entity_symbol(query_senf, query_role.filler.entity_id)
            source_symbol = _entity_symbol(source_senf, source_role.filler.entity_id)
            if query_symbol != source_symbol and (
                query_symbol in source_symbols or source_symbol in query_symbols
            ):
                return ()
            query_mention = _mention_for(query_senf, query_role.filler)
            source_mention = _mention_for(source_senf, source_role.filler)
            identity_part, conflict_part, supported = _identity_cost(
                graph,
                query_mention,
                source_mention,
                query_role.filler.entity_id,
                source_role.filler.entity_id,
            )
            identity += identity_part
            conflict += conflict_part
            base_entities.append(EntityMap(
                source_role.filler.entity_id,
                query_role.filler.entity_id,
                source_mention.mention_id if source_mention else "",
                query_mention.mention_id if query_mention else "",
                source_symbol,
                query_symbol,
                0.0,
                supported,
                source_senf.senf_id,
                source_frame.frame_id,
                query_frame.frame_id,
            ))
            exemplar_choices.append(_exemplar_options(
                query_senf, query_mention, query_frame,
                source_senf, source_mention, source_frame,
                limits.max_exemplar_alternatives,
            ))
            if supported:
                score += 0.4 / max(1, entity_role_count)
                evidence.append("role_filler")
        else:
            score += 0.15 / max(1, len(query_roles))
            evidence.append("typed_filler")

    if query_frame.polarity == source_frame.polarity:
        score += 0.1
        evidence.append("polarity_agree")
    else:
        score -= 0.3
        conflict += 0.4
        evidence.append("polarity_conflict")
    if score < MIN_PAIR_SCORE:
        return ()

    time = _mismatch(query_frame, source_frame, "time_ref")
    location = _mismatch(query_frame, source_frame, "location_ref")
    modality = _mismatch(query_frame, source_frame, "modality")
    if _mismatch(query_frame, source_frame, "speaker"):
        conflict += 1.0
    pair = FramePair(
        query_frame.frame_id,
        source_frame.frame_id,
        round(score, 4),
        tuple(dict.fromkeys(evidence)),
        round(max(0.0, min(1.0, 1.0 - score)), 4),
        source_senf.senf_id,
        0.0,
        transport.branch_probability,
    )

    combinations = product(*exemplar_choices) if exemplar_choices else [()]
    hypotheses: list[_PairHypothesis] = []
    for combination in combinations:
        exemplar_cost = sum(choice[2] for choice in combination)
        exemplar_maps = tuple(
            (source_exemplar, query_exemplar)
            for source_exemplar, query_exemplar, _ in combination
            if source_exemplar is not None and query_exemplar is not None
        )
        entities = tuple(
            EntityMap(
                mapping.source_entity_id,
                mapping.target_entity_id,
                mapping.source_mention_id,
                mapping.target_mention_id,
                mapping.source_symbol,
                mapping.target_symbol,
                round(min(1.0, identity_part + conflict_part + exemplar_part[2]), 4),
                mapping.identity_supported,
                mapping.source_id,
                mapping.source_frame_id,
                mapping.query_frame_id,
            )
            for mapping, exemplar_part, identity_part, conflict_part in zip(
                base_entities,
                combination,
                _entity_identity_parts(base_entities, graph),
                _entity_conflict_parts(base_entities, graph),
            )
        )
        local_cost = (
            structural + exemplar_cost + identity + conflict + time + location
            + modality + transport.branch_cost + transport.temporal_cost
            + transport.persistence_cost
        )
        hypotheses.append(_PairHypothesis(
            replace(
                pair,
                transport_cost=round(local_cost, 4),
                branch_probability=round(transport.branch_probability, 8),
            ),
            entities,
            (PredicateMap(
                source_frame.predicate_head,
                query_frame.predicate_head,
                predicate_cost,
                source_senf.senf_id,
                source_frame.frame_id,
                query_frame.frame_id,
            ),),
            exemplar_maps,
            tuple(role_maps),
            structural,
            exemplar_cost,
            identity,
            conflict,
            time,
            location,
            modality,
            transport.branch_cost,
            transport.temporal_cost,
            transport.persistence_cost,
            transport,
        ))
        if len(hypotheses) >= limits.max_exemplar_alternatives:
            break
    return tuple(sorted(hypotheses, key=_pair_key))


def _entity_identity_parts(
    mappings: Sequence[EntityMap], graph: Optional[IdentityGraph]
) -> tuple[float, ...]:
    if graph is None:
        return tuple(0.0 for _ in mappings)
    parts = []
    for mapping in mappings:
        if mapping.source_entity_id == mapping.target_entity_id:
            parts.append(0.0)
            continue
        edge = next((
            edge for edge in graph.edges
            if {mapping.source_mention_id, mapping.target_mention_id} == set(edge.mention_ids)
        ), None)
        if graph.same_entity(mapping.source_entity_id, mapping.target_entity_id):
            parts.append(edge.positive_cost if edge else graph.transport_cost(
                mapping.source_entity_id, mapping.target_entity_id
            ))
        else:
            parts.append(edge.positive_cost if edge and edge.strength > 0 else 1.0)
    return tuple(parts)


def _entity_conflict_parts(
    mappings: Sequence[EntityMap], graph: Optional[IdentityGraph]
) -> tuple[float, ...]:
    if graph is None:
        return tuple(0.0 for _ in mappings)
    return tuple(
        next((
            edge.negative_strength for edge in graph.edges
            if {mapping.source_mention_id, mapping.target_mention_id} == set(edge.mention_ids)
        ), 0.0)
        for mapping in mappings
    )


def _pair_key(item: _PairHypothesis) -> tuple:
    return (
        round(item.local_cost, 8),
        -item.pair.score,
        item.pair.query_frame_id,
        item.pair.source_frame_id,
        item.exemplar_maps,
        tuple((mapping.source_entity_id, mapping.target_entity_id) for mapping in item.entity_maps),
    )


def _state_key(state: _SearchState, remaining: int, query_count: int) -> tuple:
    # Future pair costs are non-negative. Dividing the cost already incurred by
    # the largest possible final selection gives a lower bound on final average
    # pair cost, in the same units as WeaveResult.total_cost.
    max_selected = len(state.selected) + remaining
    local_lower_bound = sum(item.local_cost for item in state.selected) / max(1, max_selected)
    unmatched_lower_bound = state.skipped / max(1, query_count)
    return (
        round(local_lower_bound + unmatched_lower_bound, 8),
        state.skipped,
        tuple(_pair_key(item) for item in state.selected),
    )


def _cross_frame_distortion(
    query: SENF, source: SENF, selected: Sequence[_PairHypothesis]
) -> float:
    if len(selected) < 2:
        return 0.0
    violations = checks = 0

    source_targets: dict[str, set[str]] = {}
    target_sources: dict[str, set[str]] = {}
    for item in selected:
        for mapping in item.entity_maps:
            source_targets.setdefault(mapping.source_entity_id, set()).add(mapping.target_entity_id)
            target_sources.setdefault(mapping.target_entity_id, set()).add(mapping.source_entity_id)
    for values in tuple(source_targets.values()) + tuple(target_sources.values()):
        checks += 1
        violations += max(0, len(values) - 1)

    role_targets: dict[tuple[str, str], set[str]] = {}
    role_sources: dict[tuple[str, str], set[str]] = {}
    for item in selected:
        source_head = item.predicate_maps[0].source_head
        query_head = item.predicate_maps[0].query_head
        for source_role, query_role in item.role_maps:
            role_targets.setdefault((source_head, source_role), set()).add(query_role)
            role_sources.setdefault((query_head, query_role), set()).add(source_role)
    for values in tuple(role_targets.values()) + tuple(role_sources.values()):
        checks += 1
        violations += max(0, len(values) - 1)

    source_to_query = {item.pair.source_frame_id: item.pair.query_frame_id for item in selected}
    query_to_source = {value: key for key, value in source_to_query.items()}

    def relationship_violations(
        left: SENF,
        right: SENF,
        mapping: dict[str, str],
    ) -> tuple[int, int]:
        bad = total = 0
        for parent_id, mapped_parent_id in mapping.items():
            parent = _frame(left, parent_id)
            mapped_parent = _frame(right, mapped_parent_id)
            if parent is None or mapped_parent is None:
                continue
            mapped_children = {
                role.position: role.filler.frame_id
                for role in mapped_parent.roles if isinstance(role.filler, FrameRef)
            }
            for role in parent.roles:
                if not isinstance(role.filler, FrameRef) or role.filler.frame_id not in mapping:
                    continue
                total += 1
                if mapped_children.get(role.position) != mapping[role.filler.frame_id]:
                    bad += 1
        return bad, total

    for left, right, mapping in (
        (source, query, source_to_query),
        (query, source, query_to_source),
    ):
        bad, total = relationship_violations(left, right, mapping)
        violations += bad
        checks += total
    return min(1.0, violations / max(1, checks))


def _result(
    query: SENF,
    source: SENF,
    state: _SearchState,
    query_count: int,
    graph: Optional[IdentityGraph],
) -> WeaveResult:
    selected = state.selected
    pairs = tuple(item.pair for item in selected)
    entity_maps = tuple(mapping for item in selected for mapping in item.entity_maps)
    predicate_maps = tuple(mapping for item in selected for mapping in item.predicate_maps)
    exemplar_maps = tuple(mapping for item in selected for mapping in item.exemplar_maps)
    role_maps = tuple(mapping for item in selected for mapping in item.role_maps)
    divisor = max(1, len(selected))

    def average(name: str) -> float:
        return sum(getattr(item, name) for item in selected) / divisor

    structural = average("structural")
    exemplar = average("exemplar")
    identity = average("identity")
    conflict = average("conflict")
    time = average("time")
    location = average("location")
    modality = average("modality")
    branch = average("branch")
    temporal_decay_cost = average("temporal_decay")
    persistence_cost = average("persistence")
    unmatched = state.skipped / max(1, query_count)
    cross_distortion = _cross_frame_distortion(query, source, selected)
    distortion = min(1.0, unmatched + cross_distortion)
    total = (
        structural + exemplar + identity + conflict + time + location
        + modality + branch + temporal_decay_cost + persistence_cost
        + unmatched + cross_distortion
    )

    used_source = {pair.source_frame_id for pair in pairs}
    grounded: set[str] = set()
    grounded_entity_ids: set[str] = set()
    signatures: set[tuple[str, str]] = set()
    for frame in source.frames:
        if frame.frame_id not in used_source:
            continue
        for role in frame.roles:
            if not isinstance(role.filler, EntityRef):
                continue
            entity_id = role.filler.entity_id
            resolved = graph.resolve_entity(entity_id) if graph else entity_id
            grounded_entity_ids.add(resolved)
            symbol = next(
                (
                    mapping.target_symbol
                    for mapping in entity_maps
                    if mapping.source_entity_id == entity_id
                    and mapping.identity_supported
                ),
                graph.entity_symbols.get(resolved, _entity_symbol(source, entity_id))
                if graph else _entity_symbol(source, entity_id),
            )
            grounded.add(symbol)
            signatures.add((frame.predicate_head, symbol))

    kind_maps = tuple(sorted({
        (source_kind, query_kind)
        for mapping in entity_maps
        for source_kind in _entity_kinds(source, mapping.source_entity_id)
        for query_kind in _entity_kinds(query, mapping.target_entity_id)
    }))
    rounded_distortion = round(distortion, 4)
    return WeaveResult(
        pairs=pairs,
        distortion=rounded_distortion,
        grounded_symbols=frozenset(grounded),
        grounded_entity_ids=frozenset(grounded_entity_ids),
        role_signatures=frozenset(signatures),
        entity_maps=entity_maps,
        predicate_maps=predicate_maps,
        kind_maps=kind_maps,
        exemplar_maps=exemplar_maps,
        role_maps=role_maps,
        structural_cost=round(structural, 4),
        exemplar_cost=round(exemplar, 4),
        identity_cost=round(identity, 4),
        conflict_cost=round(conflict, 4),
        time_cost=round(time, 4),
        location_cost=round(location, 4),
        modality_cost=round(modality, 4),
        unmatched_cost=round(unmatched, 4),
        distortion_cost=round(cross_distortion, 4),
        branch_cost=round(branch, 4),
        temporal_decay_cost=round(temporal_decay_cost, 4),
        persistence_cost=round(persistence_cost, 4),
        branch_probability=round(min(
            (item.transport_decision.branch_probability for item in selected),
            default=1.0,
        ), 8),
        branch_lca=next((
            item.transport_decision.branch_lca
            for item in selected if item.transport_decision.branch_lca
        ), "actual_root"),
        transport_decisions=tuple(item.transport_decision for item in selected),
        total_cost=round(total, 4),
        guard=f"{source.sentence_id}->{query.sentence_id}",
        residuals=(rounded_distortion, rounded_distortion, rounded_distortion),
    )


def _weave_source(
    query: SENF,
    source: SENF,
    graph: Optional[IdentityGraph],
    limits: _Limits,
) -> tuple[WeaveResult, ...]:
    tree = BranchingContextTree.from_senfs((source, query))
    intervals: dict[str, ValidityInterval] = {}
    for senf in (source, query):
        for interval in senf.validity_intervals:
            existing = intervals.get(interval.interval_id)
            if existing is not None and existing != interval:
                return (WeaveResult(guard=f"{source.sentence_id}->{query.sentence_id}"),)
            intervals[interval.interval_id] = interval
    all_query_frames = sorted(
        (frame for frame in query.frames if frame.clause_role == "fact"),
        key=lambda frame: frame.frame_id,
    )
    query_frames = all_query_frames[:limits.max_frames]
    source_frames = sorted(
        (frame for frame in source.frames if frame.clause_role != "premise"),
        key=lambda frame: frame.frame_id,
    )[:limits.max_frames]
    by_query: dict[str, list[_PairHypothesis]] = {frame.frame_id: [] for frame in query_frames}
    for query_frame in query_frames:
        for source_frame in source_frames:
            by_query[query_frame.frame_id].extend(_pair_hypotheses(
                query, source, query_frame, source_frame, graph, limits, tree, intervals
            ))
        by_query[query_frame.frame_id].sort(key=_pair_key)

    # Allocate in rounds so no frame receives a second candidate while another
    # frame with candidates has received none, including when the cap is small.
    retained: dict[str, list[_PairHypothesis]] = {
        frame.frame_id: [] for frame in query_frames
    }
    allocated = 0
    round_index = 0
    while allocated < limits.max_pair_candidates:
        available = [
            (by_query[frame.frame_id][round_index], frame)
            for frame in query_frames
            if round_index < len(by_query[frame.frame_id])
        ]
        if not available:
            break
        available.sort(key=lambda item: (_pair_key(item[0]), item[1].frame_id))
        for candidate, frame in available:
            retained[frame.frame_id].append(candidate)
            allocated += 1
            if allocated >= limits.max_pair_candidates:
                break
        round_index += 1
    by_query = retained

    beam = [_SearchState(skipped=max(0, len(all_query_frames) - len(query_frames)))]
    for frame_index, query_frame in enumerate(query_frames):
        expanded: list[_SearchState] = []
        for state in beam:
            expanded.append(_SearchState(state.selected, state.used_sources, state.skipped + 1))
            for candidate in by_query[query_frame.frame_id]:
                if candidate.pair.source_frame_id in state.used_sources:
                    continue
                expanded.append(_SearchState(
                    state.selected + (candidate,),
                    state.used_sources | {candidate.pair.source_frame_id},
                    state.skipped,
                ))
        remaining = len(query_frames) - frame_index - 1
        expanded.sort(key=lambda state: _state_key(state, remaining, len(all_query_frames)))
        beam = expanded[:limits.beam_width]

    results = [
        _result(query, source, state, len(all_query_frames), graph) for state in beam
    ]
    results.append(_result(
        query,
        source,
        _SearchState(skipped=len(all_query_frames)),
        len(all_query_frames),
        graph,
    ))
    results = [
        _reject_over_cost(result) if result.aligned and result.total_cost > limits.max_cost
        else result
        for result in results
    ]
    unique: dict[tuple, WeaveResult] = {}
    for result in results:
        key = (
            tuple((pair.query_frame_id, pair.source_frame_id) for pair in result.pairs),
            tuple((pair.query_frame_id, pair.source_frame_id) for pair in result.rejected_pairs),
            result.exemplar_maps,
            tuple((item.source_entity_id, item.target_entity_id) for item in result.entity_maps),
        )
        unique.setdefault(key, result)
    ranked = sorted(unique.values(), key=_result_key)
    return tuple(ranked[:limits.per_source_k])


def _result_key(result: WeaveResult) -> tuple:
    return (
        1 if result.rejected_pairs else 0,
        result.total_cost,
        result.distortion,
        result.guard,
        tuple((pair.query_frame_id, pair.source_frame_id) for pair in result.pairs),
        tuple((pair.query_frame_id, pair.source_frame_id) for pair in result.rejected_pairs),
        result.exemplar_maps,
        tuple((item.source_entity_id, item.target_entity_id) for item in result.entity_maps),
    )


def _reject_over_cost(result: WeaveResult) -> WeaveResult:
    return WeaveResult(
        rejected_pairs=result.pairs,
        distortion=result.distortion,
        structural_cost=result.structural_cost,
        exemplar_cost=result.exemplar_cost,
        identity_cost=result.identity_cost,
        conflict_cost=result.conflict_cost,
        time_cost=result.time_cost,
        location_cost=result.location_cost,
        modality_cost=result.modality_cost,
        unmatched_cost=result.unmatched_cost,
        distortion_cost=result.distortion_cost,
        branch_cost=result.branch_cost,
        temporal_decay_cost=result.temporal_decay_cost,
        persistence_cost=result.persistence_cost,
        branch_probability=result.branch_probability,
        branch_lca=result.branch_lca,
        transport_decisions=result.transport_decisions,
        total_cost=result.total_cost,
        guard=result.guard,
        residuals=result.residuals,
        alignments=result.alignments,
        polish=result.polish,
    )


def _frame_persistence(
    frame: SENFFrame, senf: SENF
) -> tuple[EntityPersistence, ...]:
    entity_ids = {
        role.filler.entity_id
        for role in frame.roles
        if isinstance(role.filler, EntityRef)
    }
    candidates = [
        item
        for item in senf.entity_persistence
        if item.entity_id in entity_ids
        and item.branch_id == (frame.context.branch_id if frame.context else "actual_root")
    ]
    return tuple(sorted(candidates, key=lambda item: (item.entity_id, item.persistence_type)))


def _hierarchy_candidates(
    query: SENF,
    sources: Sequence[SENF],
    graph: Optional[IdentityGraph],
    limits: _Limits,
    cap: int,
) -> tuple[
    tuple[FrameAlignment, ...],
    dict[FrameAlignment, tuple[SENF, _PairHypothesis]],
]:
    cap = max(0, cap)
    buckets: dict[
        str, dict[str, list[tuple[FrameAlignment, SENF, _PairHypothesis]]]
    ] = {}

    def candidate_rank(
        item: tuple[FrameAlignment, SENF, _PairHypothesis],
    ) -> tuple:
        return (
            _pair_key(item[2]),
            item[0].source_frame.source_id,
            item[0].source_frame.frame_id,
        )
    query_frames = {
        frame.frame_id: frame
        for frame in sorted(
            (frame for frame in query.frames if frame.clause_role == "fact"),
            key=lambda frame: frame.frame_id,
        )[:limits.max_frames]
    }
    for source in sorted(sources, key=lambda item: (item.sentence_id, item.senf_id)):
        source_id = source.senf_id
        tree = BranchingContextTree.from_senfs((source, query))
        intervals: dict[str, ValidityInterval] = {}
        invalid_intervals = False
        for senf in (source, query):
            for interval in senf.validity_intervals:
                existing = intervals.get(interval.interval_id)
                if existing is not None and existing != interval:
                    invalid_intervals = True
                    break
                intervals[interval.interval_id] = interval
            if invalid_intervals:
                break
        if invalid_intervals:
            continue
        source_frames = {
            frame.frame_id: frame
            for frame in sorted(
                (frame for frame in source.frames if frame.clause_role != "premise"),
                key=lambda frame: frame.frame_id,
            )[:limits.max_frames]
        }
        for query_id, query_frame in sorted(query_frames.items()):
            bucket = buckets.setdefault(query_id, {}).setdefault(source_id, [])
            for source_frame_id, source_frame in sorted(source_frames.items()):
                hypotheses = _pair_hypotheses(
                    query, source, query_frame, source_frame, graph, limits, tree,
                    intervals,
                )
                if not hypotheses:
                    continue
                hypothesis = hypotheses[0]
                costs = PairComponentCosts(
                    hypothesis.structural,
                    hypothesis.exemplar,
                    hypothesis.identity,
                    hypothesis.conflict,
                    hypothesis.time,
                    hypothesis.location,
                    hypothesis.modality,
                    hypothesis.branch,
                    hypothesis.temporal_decay,
                    hypothesis.persistence,
                )
                alignment = FrameAlignment(
                    query_id,
                    SourceFrameKey(source_id, source_frame_id),
                    hypothesis.pair.score,
                    costs,
                    hypothesis.pair.evidence,
                )
                bucket.append((alignment, source, hypothesis))
                bucket.sort(key=candidate_rank)
                if len(bucket) > cap:
                    bucket.pop()

    # First interleave sources for each query, then interleave queries globally.
    # No bucket receives a second slot while another eligible bucket at the same
    # level has received none.
    fair_by_query: dict[
        str, list[tuple[FrameAlignment, SENF, _PairHypothesis]]
    ] = {}
    for query_id, by_source in sorted(buckets.items()):
        fair = fair_by_query.setdefault(query_id, [])
        depth = 0
        while len(fair) < cap:
            available = [
                values[depth]
                for _, values in sorted(by_source.items())
                if depth < len(values)
            ]
            if not available:
                break
            fair.extend(sorted(available, key=candidate_rank)[:cap - len(fair)])
            depth += 1

    ordered: list[tuple[FrameAlignment, SENF, _PairHypothesis]] = []
    depth = 0
    while len(ordered) < cap:
        available = [
            values[depth]
            for _, values in sorted(fair_by_query.items())
            if depth < len(values)
        ]
        if not available:
            break
        ordered.extend(sorted(available, key=candidate_rank)[:cap - len(ordered)])
        depth += 1
    ordered.sort(key=candidate_rank)
    alignments = tuple(item[0] for item in ordered)
    return alignments, {
        alignment: (source, hypothesis)
        for alignment, source, hypothesis in ordered
    }


def _hierarchy_result(
    query: SENF,
    sources: Sequence[SENF],
    selected: Sequence[FrameAlignment],
    lookup: dict[FrameAlignment, tuple[SENF, _PairHypothesis]],
    graph: Optional[IdentityGraph],
    diagnostics: PolishDiagnostics,
) -> WeaveResult:
    chosen = [(alignment, *lookup[alignment]) for alignment in selected]
    hypotheses = [item[2] for item in chosen]
    query_count = sum(frame.clause_role == "fact" for frame in query.frames)
    divisor = max(1, len(hypotheses))

    def average(name: str) -> float:
        return sum(getattr(item, name) for item in hypotheses) / divisor

    source_ids = sorted({
        source.sentence_id for _, source, _ in chosen
    } or {
        source.sentence_id for source in sources
    })
    guard = f"{'+'.join(source_ids)}->{query.sentence_id}"
    grouped: dict[str, tuple[SENF, list[_PairHypothesis]]] = {}
    for _, source, hypothesis in chosen:
        grouped.setdefault(source.senf_id, (source, []))[1].append(hypothesis)
    partials = [
        _result(
            query,
            source,
            _SearchState(tuple(items), frozenset(
                item.pair.source_frame_id for item in items
            )),
            query_count,
            graph,
        )
        for source, items in (grouped[key] for key in sorted(grouped))
    ]

    entity_maps = tuple(
        mapping for item in hypotheses for mapping in item.entity_maps
    )
    source_targets: dict[tuple[str, str], set[str]] = {}
    target_sources: dict[str, list[EntityMap]] = {}
    for alignment, _, hypothesis in chosen:
        for mapping in hypothesis.entity_maps:
            source_key = (alignment.source_frame.source_id, mapping.source_entity_id)
            source_targets.setdefault(source_key, set()).add(mapping.target_entity_id)
            target_sources.setdefault(mapping.target_entity_id, []).append(mapping)
    consistency_violations = sum(
        max(0, len(values) - 1) for values in source_targets.values()
    )
    consistency_checks = len(source_targets)
    for mappings in target_sources.values():
        groups: list[list[EntityMap]] = []
        for mapping in mappings:
            matching = [
                group for group in groups
                if any(
                    item.source_symbol == mapping.source_symbol
                    or graph is not None and graph.same_entity(
                        item.source_entity_id, mapping.source_entity_id
                    )
                    for item in group
                )
            ]
            if not matching:
                groups.append([mapping])
            else:
                merged = [mapping]
                for group in matching:
                    merged.extend(group)
                    groups.remove(group)
                groups.append(merged)
        consistency_violations += max(0, len(groups) - 1)
        consistency_checks += 1
    cross_distortion = max(
        [item.distortion_cost for item in partials]
        + [consistency_violations / max(1, consistency_checks)]
    )
    matched_queries = {item.pair.query_frame_id for item in hypotheses}
    unmatched = max(0, query_count - len(matched_queries)) / max(1, query_count)
    distortion = min(1.0, unmatched + cross_distortion)
    component_names = (
        "structural", "exemplar", "identity", "conflict", "time", "location",
        "modality", "branch", "temporal_decay", "persistence",
    )
    component_values = {name: average(name) for name in component_names}
    total = sum(component_values.values()) + unmatched + cross_distortion
    result = WeaveResult(
        pairs=tuple(item.pair for item in hypotheses),
        distortion=round(distortion, 4),
        grounded_symbols=frozenset().union(*(
            item.grounded_symbols for item in partials
        )),
        grounded_entity_ids=frozenset().union(*(
            item.grounded_entity_ids for item in partials
        )),
        role_signatures=frozenset().union(*(
            item.role_signatures for item in partials
        )),
        entity_maps=entity_maps,
        predicate_maps=tuple(
            mapping for item in hypotheses for mapping in item.predicate_maps
        ),
        kind_maps=tuple(sorted({mapping for item in partials for mapping in item.kind_maps})),
        exemplar_maps=tuple(
            mapping for item in hypotheses for mapping in item.exemplar_maps
        ),
        role_maps=tuple(mapping for item in hypotheses for mapping in item.role_maps),
        structural_cost=round(component_values["structural"], 4),
        exemplar_cost=round(component_values["exemplar"], 4),
        identity_cost=round(component_values["identity"], 4),
        conflict_cost=round(component_values["conflict"], 4),
        time_cost=round(component_values["time"], 4),
        location_cost=round(component_values["location"], 4),
        modality_cost=round(component_values["modality"], 4),
        unmatched_cost=round(unmatched, 4),
        distortion_cost=round(cross_distortion, 4),
        branch_cost=round(component_values["branch"], 4),
        temporal_decay_cost=round(component_values["temporal_decay"], 4),
        persistence_cost=round(component_values["persistence"], 4),
        branch_probability=round(min(
            (item.transport_decision.branch_probability for item in hypotheses),
            default=1.0,
        ), 8),
        branch_lca=next((
            item.transport_decision.branch_lca
            for item in hypotheses if item.transport_decision.branch_lca
        ), "actual_root"),
        transport_decisions=tuple(item.transport_decision for item in hypotheses),
        total_cost=round(total, 4),
        guard=guard,
        residuals=diagnostics.residuals,
        alignments=tuple(selected),
        polish=diagnostics,
    )
    settings = get_settings()
    if result.aligned and result.total_cost > max(0.0, settings.senf_weave_max_cost):
        return _reject_over_cost(result)
    return result


def build_weaves(
    query: SENF,
    sources: Sequence[SENF],
    k: int = 3,
    identity_graph: Optional[IdentityGraph] = None,
) -> tuple[WeaveResult, ...]:
    if not query.frames:
        return (WeaveResult(),)
    limits = _limits()
    active_sources = [source for source in sources if source.frames]
    if not active_sources:
        return ()
    settings = get_settings()
    if settings.senf_weave_engine == "beam":
        results = [
            result
            for source in active_sources
            for result in _weave_source(query, source, identity_graph, limits)
        ]
        results.sort(key=_result_key)
        return tuple(results[:max(1, k)])

    from core.senf.weave_hierarchy import (
        HierarchyFallback,
        HierarchyLimits,
        polish_hierarchy,
    )

    def invalid_integer(value: object) -> bool:
        return type(value) is not int or value <= 0

    def invalid_float(value: object) -> bool:
        return (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        )

    def fallback(reason: str) -> tuple[WeaveResult, ...]:
        diagnostics = PolishDiagnostics(
            engine="hierarchical",
            fallback_reason=reason,
            seed_count=1,
        )
        return (WeaveResult(
            distortion=1.0,
            unmatched_cost=1.0,
            total_cost=1.0,
            guard=f"unaligned->{query.sentence_id}",
            polish=diagnostics,
        ),)

    hierarchy_values = (
        settings.senf_weave_global_candidate_cap,
        settings.senf_weave_max_cells,
        settings.senf_weave_max_seeds,
        settings.senf_weave_max_iterations,
    )
    if any(invalid_integer(value) for value in hierarchy_values):
        return fallback("invalid_hierarchy_bound")
    if any(invalid_float(value) for value in (
        settings.senf_weave_sinkhorn_tolerance,
        settings.senf_weave_sinkhorn_regularization,
    )):
        return fallback("invalid_hierarchy_numeric_setting")

    max_sources = max(1, int(settings.senf_query_max_priors))
    max_source_frames = max(1, int(settings.senf_query_max_source_frames))
    if len(active_sources) > max_sources:
        return fallback("hierarchy_source_cap")
    bounded_source_frames = sum(
        min(
            limits.max_frames,
            sum(frame.clause_role != "premise" for frame in source.frames),
        )
        for source in active_sources
    )
    if bounded_source_frames > max_source_frames:
        return fallback("hierarchy_source_frame_cap")
    query_frame_count = min(
        limits.max_frames,
        sum(frame.clause_role == "fact" for frame in query.frames),
    )
    pair_work = query_frame_count * bounded_source_frames
    pair_work_cap = settings.senf_weave_global_candidate_cap * max_sources
    if pair_work > pair_work_cap:
        return fallback("hierarchy_pair_work_cap")

    source_ids = [source.senf_id for source in active_sources]
    if len(source_ids) != len(set(source_ids)):
        return fallback("duplicate_source_id")
    hierarchy_limits = HierarchyLimits(
        settings.senf_weave_global_candidate_cap,
        settings.senf_weave_max_cells,
        settings.senf_weave_max_seeds,
        settings.senf_weave_max_iterations,
        settings.senf_weave_sinkhorn_tolerance,
        settings.senf_weave_sinkhorn_regularization,
    )
    try:
        candidates, lookup = _hierarchy_candidates(
            query,
            active_sources,
            identity_graph,
            limits,
            hierarchy_limits.global_candidate_cap,
        )
        polished = polish_hierarchy(
            candidates,
            tuple(
                frame.frame_id
                for frame in sorted(
                    (item for item in query.frames if item.clause_role == "fact"),
                    key=lambda item: item.frame_id,
                )[:limits.max_frames]
            ),
            hierarchy_limits,
        )
        results = [
            _hierarchy_result(
                query,
                active_sources,
                selected,
                lookup,
                identity_graph,
                polished.diagnostics,
            )
            for selected in polished.selections
        ]
    except HierarchyFallback as exc:
        return fallback(str(exc))
    except Exception as exc:
        return fallback(f"hierarchy_error:{type(exc).__name__}")
    results.sort(key=_result_key)
    return tuple(results[:max(1, k)])


def weave(
    query: SENF,
    sources: Sequence[SENF],
    identity_graph: Optional[IdentityGraph] = None,
) -> WeaveResult:
    results = build_weaves(query, sources, 1, identity_graph=identity_graph)
    if results:
        return results[0]
    return WeaveResult(
        distortion=1.0 if query.frames else 0.0,
        unmatched_cost=1.0 if query.frames else 0.0,
        total_cost=1.0 if query.frames else 0.0,
    )
