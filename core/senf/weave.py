from dataclasses import dataclass
from typing import Optional, Sequence

from core.senf.exemplars import exemplar_distance
from core.senf.identity import IdentityGraph
from core.senf.types import EntityRef, FrameRef, KindRef, Mention, Role, SENF, SENFFrame, ValueRef


MIN_PAIR_SCORE = 0.3

_PREDICATE_FAMILIES = {
    "HasProperty": "property",
    "PropertyOf": "property",
    "AtLocation": "location",
    "LocatedAt": "location",
    "LocatedIn": "location",
    "IsA": "type",
    "InstanceOf": "type",
    "PartOf": "part",
    "HasPart": "part",
    "UsedFor": "purpose",
    "PurposeOf": "purpose",
    "CapableOf": "capability",
}


@dataclass(frozen=True)
class EntityMap:
    source_entity_id: str
    target_entity_id: str
    source_mention_id: str
    target_mention_id: str
    source_symbol: str
    target_symbol: str
    cost: float


@dataclass(frozen=True)
class PredicateMap:
    source_head: str
    query_head: str
    cost: float


@dataclass(frozen=True)
class FramePair:
    query_frame_id: str
    source_frame_id: str
    score: float
    evidence: tuple[str, ...] = ()
    cost: float = 1.0


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
    conflict_cost: float = 0.0
    unmatched_cost: float = 0.0
    total_cost: float = 0.0
    guard: str = ""
    residuals: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def aligned(self) -> bool:
        return bool(self.pairs)


def _split_head(head: str) -> list[str]:
    out: list[str] = []
    current = ""
    for char in head.replace("_", " "):
        if char == " ":
            if current:
                out.append(current.lower())
            current = ""
        elif char.isupper() and current:
            out.append(current.lower())
            current = char
        else:
            current += char
    if current:
        out.append(current.lower())
    return out


def _head_score(left: str, right: str) -> tuple[float, Optional[str]]:
    if left == right:
        return 0.5, "head_exact"
    if set(_split_head(left)) & set(_split_head(right)):
        return 0.2, "head_overlap"
    return 0.0, None


def _predicate_cost(query_head: str, source_head: str) -> Optional[float]:
    if query_head == source_head:
        return 0.0
    query_family = _PREDICATE_FAMILIES.get(query_head)
    if query_family and query_family == _PREDICATE_FAMILIES.get(source_head):
        return 0.25
    if set(_split_head(query_head)) & set(_split_head(source_head)):
        return 0.4
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


def _identity_conflict(
    graph: Optional[IdentityGraph], left: Optional[Mention], right: Optional[Mention]
) -> float:
    if graph is None or left is None or right is None:
        return 0.0
    keys = {left.mention_id, right.mention_id}
    for edge in graph.edges:
        if keys == set(edge.mention_ids):
            return edge.negative_strength
    return 0.0


def _same_entity(graph: Optional[IdentityGraph], left_id: str, right_id: str) -> bool:
    return left_id == right_id or (
        graph is not None and graph.same_entity(left_id, right_id)
    )


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
    seen.add(key)
    query_frame = next((f for f in query_senf.frames if f.frame_id == query_ref.frame_id), None)
    source_frame = next((f for f in source_senf.frames if f.frame_id == source_ref.frame_id), None)
    if query_frame is None or source_frame is None:
        return None
    predicate_cost = _predicate_cost(query_frame.predicate_head, source_frame.predicate_head)
    if predicate_cost is None or len(query_frame.roles) != len(source_frame.roles):
        return None
    cost = predicate_cost
    for query_role, source_role in zip(query_frame.roles, source_frame.roles):
        filler_cost = _filler_cost(
            query_senf, source_senf, query_role, source_role, graph, seen
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
        same = _same_entity(graph, query_filler.entity_id, source_filler.entity_id)
        if same:
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


def _pair(
    query_senf: SENF,
    source_senf: SENF,
    query_frame: SENFFrame,
    source_frame: SENFFrame,
    graph: Optional[IdentityGraph],
) -> Optional[tuple[FramePair, list[EntityMap], list[PredicateMap], list[tuple[str, str]], list[tuple[str, str]], float, float, float]]:
    predicate_cost = _predicate_cost(query_frame.predicate_head, source_frame.predicate_head)
    if predicate_cost is None or len(query_frame.roles) != len(source_frame.roles):
        return None
    score, head_reason = _head_score(query_frame.predicate_head, source_frame.predicate_head)
    evidence = [head_reason] if head_reason else []
    if score == 0.0 and predicate_cost < 1.0:
        score = 0.3
        evidence.append("predicate_bridge")
    structural_cost = predicate_cost
    exemplar_cost = 0.0
    conflict_cost = 0.0
    entity_maps: list[EntityMap] = []
    predicate_maps = [PredicateMap(source_frame.predicate_head, query_frame.predicate_head, predicate_cost)]
    exemplar_maps: list[tuple[str, str]] = []
    role_maps: list[tuple[str, str]] = []

    query_roles = sorted(query_frame.roles, key=lambda role: role.position)
    source_roles = sorted(source_frame.roles, key=lambda role: role.position)
    entity_role_count = sum(isinstance(role.filler, EntityRef) for role in query_frame.roles)
    query_symbols = {
        _entity_symbol(query_senf, role.filler.entity_id)
        for role in query_roles
        if isinstance(role.filler, EntityRef)
    }
    source_symbols = {
        _entity_symbol(source_senf, role.filler.entity_id)
        for role in source_roles
        if isinstance(role.filler, EntityRef)
    }
    for query_role, source_role in zip(query_roles, source_roles):
        filler_cost = _filler_cost(
            query_senf, source_senf, query_role, source_role, graph
        )
        if filler_cost is None:
            return None
        role_cost = 0.0 if query_role.name == source_role.name else 0.25
        structural_cost += role_cost + filler_cost
        if isinstance(query_role.filler, EntityRef) and isinstance(source_role.filler, EntityRef):
            query_symbol = _entity_symbol(query_senf, query_role.filler.entity_id)
            source_symbol = _entity_symbol(source_senf, source_role.filler.entity_id)
            if query_symbol != source_symbol and (
                query_symbol in source_symbols or source_symbol in query_symbols
            ):
                return None
            query_mention = _mention_for(query_senf, query_role.filler)
            source_mention = _mention_for(source_senf, source_role.filler)
            selected_exemplar_cost = (
                0.5 * exemplar_distance(query_senf, query_mention, source_senf, source_mention)
                if query_mention is not None and source_mention is not None
                else 0.0
            )
            selected_conflict = _identity_conflict(graph, query_mention, source_mention)
            exemplar_cost += selected_exemplar_cost
            conflict_cost += selected_conflict
            entity_maps.append(
                EntityMap(
                    source_role.filler.entity_id,
                    query_role.filler.entity_id,
                    source_mention.mention_id if source_mention else "",
                    query_mention.mention_id if query_mention else "",
                    _entity_symbol(source_senf, source_role.filler.entity_id),
                    _entity_symbol(query_senf, query_role.filler.entity_id),
                    round(min(1.0, filler_cost + selected_exemplar_cost + selected_conflict), 4),
                )
            )
            source_ex = source_senf.nearest_exemplar_for(source_mention) if source_mention else None
            query_ex = query_senf.nearest_exemplar_for(query_mention) if query_mention else None
            if source_ex and query_ex:
                exemplar_maps.append((source_ex, query_ex))
            if _same_entity(
                graph,
                source_role.filler.entity_id, query_role.filler.entity_id
            ):
                score += 0.4 / max(1, entity_role_count)
                evidence.append("role_filler")
        else:
            score += 0.15 / max(1, len(query_frame.roles))
            evidence.append("typed_filler")
        role_maps.append((source_role.name, query_role.name))

    if score > 0:
        if query_frame.polarity == source_frame.polarity:
            score += 0.1
            evidence.append("polarity_agree")
        else:
            score -= 0.3
            conflict_cost += 0.4
            evidence.append("polarity_conflict")
    return (
        FramePair(
            query_frame.frame_id,
            source_frame.frame_id,
            round(score, 4),
            tuple(dict.fromkeys(evidence)),
            round(max(0.0, min(1.0, 1.0 - score)), 4),
        ),
        entity_maps,
        predicate_maps,
        exemplar_maps,
        role_maps,
        structural_cost,
        exemplar_cost,
        conflict_cost,
    )


def _weave_one(
    query: SENF,
    source: SENF,
    graph: Optional[IdentityGraph],
) -> WeaveResult:
    candidates = []
    for query_frame in query.frames:
        if query_frame.clause_role != "fact":
            continue
        for source_frame in source.frames:
            if source_frame.clause_role == "premise":
                continue
            result = _pair(query, source, query_frame, source_frame, graph)
            if result is not None and result[0].score >= MIN_PAIR_SCORE:
                candidates.append(result)
    candidates.sort(key=lambda item: (-item[0].score, item[0].query_frame_id, item[0].source_frame_id))

    used_query: set[str] = set()
    used_source: set[str] = set()
    pairs: list[FramePair] = []
    entity_maps: list[EntityMap] = []
    predicate_maps: list[PredicateMap] = []
    exemplar_maps: list[tuple[str, str]] = []
    role_maps: list[tuple[str, str]] = []
    grounded: set[str] = set()
    grounded_entity_ids: set[str] = set()
    signatures: set[tuple[str, str]] = set()
    structural = exemplar = conflict = 0.0
    for pair, entities, predicates, exemplars, roles, pair_structural, pair_exemplar, pair_conflict in candidates:
        if pair.query_frame_id in used_query or pair.source_frame_id in used_source:
            continue
        used_query.add(pair.query_frame_id)
        used_source.add(pair.source_frame_id)
        pairs.append(pair)
        entity_maps.extend(entities)
        predicate_maps.extend(predicates)
        exemplar_maps.extend(exemplars)
        role_maps.extend(roles)
        structural += pair_structural
        exemplar += pair_exemplar
        conflict += pair_conflict

    for frame in source.frames:
        if frame.frame_id not in used_source:
            continue
        for role in frame.roles:
            if not isinstance(role.filler, EntityRef):
                continue
            entity_id = graph.resolve_entity(role.filler.entity_id) if graph else role.filler.entity_id
            symbol = (
                graph.entity_symbols.get(entity_id, _entity_symbol(source, role.filler.entity_id))
                if graph
                else _entity_symbol(source, role.filler.entity_id)
            )
            grounded_entity_ids.add(entity_id)
            grounded.add(symbol)
            signatures.add((frame.predicate_head, symbol))

    unmatched = max(0, len(query.frames) - len(pairs)) / max(1, len(query.frames))
    distortion = max(0.0, min(1.0, unmatched + sum(pair.cost for pair in pairs) / max(1, len(query.frames))))
    structural_cost = structural / max(1, len(pairs))
    exemplar_cost = exemplar / max(1, len(pairs))
    conflict_cost = conflict / max(1, len(pairs))
    total = structural_cost + exemplar_cost + conflict_cost + unmatched
    kind_maps = tuple(
        sorted(
            {
                (source_kind, query_kind)
                for mapping in entity_maps
                for source_kind in _entity_kinds(source, mapping.source_entity_id)
                for query_kind in _entity_kinds(query, mapping.target_entity_id)
            }
        )
    )
    residual = round(distortion, 4)
    return WeaveResult(
        pairs=tuple(pairs),
        distortion=round(distortion, 4),
        grounded_symbols=frozenset(grounded),
        grounded_entity_ids=frozenset(grounded_entity_ids),
        role_signatures=frozenset(signatures),
        entity_maps=tuple(entity_maps),
        predicate_maps=tuple(predicate_maps),
        kind_maps=kind_maps,
        exemplar_maps=tuple(exemplar_maps),
        role_maps=tuple(role_maps),
        structural_cost=round(structural_cost, 4),
        exemplar_cost=round(exemplar_cost, 4),
        conflict_cost=round(conflict_cost, 4),
        unmatched_cost=round(unmatched, 4),
        total_cost=round(total, 4),
        guard=f"{source.sentence_id}->{query.sentence_id}",
        residuals=(residual, residual, residual),
    )


def build_weaves(
    query: SENF,
    sources: Sequence[SENF],
    k: int = 3,
    identity_graph: Optional[IdentityGraph] = None,
) -> tuple[WeaveResult, ...]:
    if not query.frames:
        return (WeaveResult(),)
    results = [
        _weave_one(query, source, identity_graph)
        for source in sources
        if source.frames
    ]
    results.sort(key=lambda result: (result.total_cost, result.distortion, result.guard))
    return tuple(results[: max(1, k)])


def weave(
    query: SENF,
    sources: Sequence[SENF],
    identity_graph: Optional[IdentityGraph] = None,
) -> WeaveResult:
    results = build_weaves(query, sources, 1, identity_graph=identity_graph)
    if results:
        return results[0]
    return WeaveResult(distortion=1.0 if query.frames else 0.0, total_cost=1.0 if query.frames else 0.0)
