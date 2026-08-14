from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from core.senf.exemplars import exemplar_distance
from core.senf.identity import IdentityGraph
from core.senf.types import Mention, SENF, SENFFrame


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


def _mentions(frame: SENFFrame) -> list[tuple[str, Mention]]:
    return [
        (role.name, role.filler)
        for role in frame.roles
        if isinstance(role.filler, Mention)
    ]


def _identity_conflict(
    graph: Optional[IdentityGraph], left: Mention, right: Mention
) -> float:
    if graph is None:
        return 0.0
    keys = {left.mention_id, right.mention_id}
    for edge in graph.edges:
        if keys == set(edge.mention_ids):
            return edge.negative_strength
    return 0.0


def _pair(
    query_senf: SENF,
    source_senf: SENF,
    query_frame: SENFFrame,
    source_frame: SENFFrame,
    resolve: Callable[[str], str],
    graph: Optional[IdentityGraph],
) -> Optional[tuple[FramePair, list[EntityMap], list[PredicateMap], list[tuple[str, str]], list[tuple[str, str]], float, float]]:
    predicate_cost = _predicate_cost(query_frame.predicate_head, source_frame.predicate_head)
    if predicate_cost is None:
        return None
    score, head_reason = _head_score(query_frame.predicate_head, source_frame.predicate_head)
    evidence = [head_reason] if head_reason else []
    structural_cost = 1.0 - score
    conflict_cost = 0.0
    entity_maps: list[EntityMap] = []
    predicate_maps = [PredicateMap(source_frame.predicate_head, query_frame.predicate_head, predicate_cost)]
    exemplar_maps: list[tuple[str, str]] = []
    role_maps: list[tuple[str, str]] = []

    source_roles = _mentions(source_frame)
    for query_role, query_mention in _mentions(query_frame):
        candidates: list[tuple[float, str, Mention]] = []
        for source_role, source_mention in source_roles:
            query_kind = query_senf.kind_for(query_mention)
            source_kind = source_senf.kind_for(source_mention)
            if query_kind and source_kind and query_kind != source_kind:
                continue
            role_cost = 0.0 if query_role == source_role else 0.25
            symbol_cost = 0.0 if resolve(query_mention.canonical_symbol) == resolve(
                source_mention.canonical_symbol
            ) else 0.35
            ex_cost = exemplar_distance(
                query_senf, query_mention, source_senf, source_mention
            )
            conflict = _identity_conflict(graph, query_mention, source_mention)
            candidates.append(
                (role_cost + symbol_cost + 0.5 * ex_cost + conflict, source_role, source_mention)
            )
        if not candidates:
            return None
        cost, source_role, source_mention = min(
            candidates, key=lambda item: (item[0], item[2].mention_id)
        )
        entity_maps.append(
            EntityMap(
                source_mention.mention_id,
                query_mention.mention_id,
                source_mention.canonical_symbol,
                query_mention.canonical_symbol,
                round(min(1.0, cost), 4),
            )
        )
        role_maps.append((source_role, query_role))
        source_ex = source_senf.nearest_exemplar_for(source_mention)
        query_ex = query_senf.nearest_exemplar_for(query_mention)
        if source_ex and query_ex:
            exemplar_maps.append((source_ex, query_ex))
        if resolve(source_mention.canonical_symbol) == resolve(query_mention.canonical_symbol):
            score += 0.4 / max(1, len(_mentions(query_frame)))
            evidence.append("role_filler")
        structural_cost += 0.25 if source_role != query_role else 0.0
        conflict_cost += _identity_conflict(graph, query_mention, source_mention)

    if score > 0:
        if query_frame.polarity == source_frame.polarity:
            score += 0.1
            evidence.append("polarity_agree")
        else:
            score -= 0.3
            structural_cost += 0.3
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
        conflict_cost,
    )


def _weave_one(
    query: SENF,
    source: SENF,
    resolve: Callable[[str], str],
    graph: Optional[IdentityGraph],
) -> WeaveResult:
    candidates = []
    for query_frame in query.frames:
        for source_frame in source.frames:
            result = _pair(query, source, query_frame, source_frame, resolve, graph)
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
    signatures: set[tuple[str, str]] = set()
    structural = conflict = 0.0
    for pair, entities, predicates, exemplars, roles, pair_structural, pair_conflict in candidates:
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
        conflict += pair_conflict

    for frame in source.frames:
        if frame.frame_id not in used_source:
            continue
        for symbol in frame.filler_symbols():
            grounded.add(resolve(symbol))
            signatures.add((frame.predicate_head, resolve(symbol)))

    unmatched = max(0, len(query.frames) - len(pairs)) / max(1, len(query.frames))
    distortion = max(0.0, min(1.0, unmatched + sum(pair.cost for pair in pairs) / max(1, len(query.frames))))
    exemplar_cost = sum(mapping.cost for mapping in entity_maps) / max(1, len(entity_maps))
    total = structural / max(1, len(pairs)) + exemplar_cost + conflict + unmatched
    kind_maps = tuple(
        sorted(
            {
                (source.kinds[source_symbol], query.kinds[target_symbol])
                for mapping in entity_maps
                for source_symbol, target_symbol in [(mapping.source_symbol, mapping.target_symbol)]
                if source_symbol in source.kinds and target_symbol in query.kinds
            }
        )
    )
    residual = round(distortion, 4)
    return WeaveResult(
        pairs=tuple(pairs),
        distortion=round(distortion, 4),
        grounded_symbols=frozenset(grounded),
        role_signatures=frozenset(signatures),
        entity_maps=tuple(entity_maps),
        predicate_maps=tuple(predicate_maps),
        kind_maps=kind_maps,
        exemplar_maps=tuple(exemplar_maps),
        role_maps=tuple(role_maps),
        structural_cost=round(structural / max(1, len(pairs)), 4),
        exemplar_cost=round(exemplar_cost, 4),
        conflict_cost=round(conflict, 4),
        unmatched_cost=round(unmatched, 4),
        total_cost=round(total, 4),
        guard=f"{source.sentence_id}->{query.sentence_id}",
        residuals=(residual, residual, residual),
    )


def build_weaves(
    query: SENF,
    sources: Sequence[SENF],
    k: int = 3,
    resolve: Optional[Callable[[str], str]] = None,
    identity_graph: Optional[IdentityGraph] = None,
) -> tuple[WeaveResult, ...]:
    resolve = resolve or (lambda symbol: symbol)
    if not query.frames:
        return (WeaveResult(),)
    results = [
        _weave_one(query, source, resolve, identity_graph)
        for source in sources
        if source.frames
    ]
    results.sort(key=lambda result: (result.total_cost, result.distortion, result.guard))
    return tuple(results[: max(1, k)])


def weave(
    query: SENF,
    sources: Sequence[SENF],
    resolve: Optional[Callable[[str], str]] = None,
    identity_graph: Optional[IdentityGraph] = None,
) -> WeaveResult:
    results = build_weaves(query, sources, 1, resolve, identity_graph)
    if results:
        return results[0]
    return WeaveResult(distortion=1.0 if query.frames else 0.0, total_cost=1.0 if query.frames else 0.0)
