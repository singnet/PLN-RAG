import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from core.senf.types import SENF, Mention

logger = logging.getLogger(__name__)

DEFAULT_IDENTITY_THRESHOLD = 0.75
DEFAULT_MAX_MENTIONS_PER_SENTENCE = 32

# Election order for a cluster representative, applied only after specificity.
# A pronoun is never informative enough to win, whatever its shape.
_TYPE_PRIORITY = {"proper": 0, "nominal": 1, "common": 2, "pronoun": 3}

# How many tokens a compound may add over a bare symbol and still read as the same
# entity. One covers the real pattern (kebede / kebede_alemu, hsc / mouse_hsc);
# beyond that a shared leading or trailing word is a topic, not a coreference.
_MAX_NAME_EXTENSION_TOKENS = 1

# Subject-ish role slots. A pronoun's antecedent is far more often the prominent
# argument of a nearby frame than a peripheral one.
_PROMINENT_ROLES = frozenset({"Agent", "Theme", "Instance", "Member"})

# Third person only. First and second person are deictic — they point outside the
# text, so there is no antecedent to bind and an allowlist is safer than a denylist.
_ANAPHORIC_PRONOUNS = frozenset(
    {"he", "she", "it", "him", "her", "them", "they", "his", "its", "their",
     "this", "that", "these", "those"}
)


@dataclass(frozen=True)
class IdentityWeights:
    """Per-evidence contributions to an edge's strength.

    Tuned so that no single evidence type can reach the default threshold on its
    own: every merge needs corroboration. `surface_match` and `proper_compat` sit
    just below it precisely so one more signal is required.
    """

    exact_symbol: float = 0.55
    surface_match: float = 0.55
    proper_compat: float = 0.55
    head_lemma: float = 0.3
    name_extension: float = 0.55
    kind_match: float = 0.25
    role_compat: float = 0.2
    pronoun_antecedent: float = 0.4
    role_prominence: float = 0.2
    unambiguous: float = 0.35
    embed_sim: float = 0.3
    kind_conflict: float = 0.9
    proper_conflict: float = 0.9
    modifier_conflict: float = 0.8
    both_pronouns: float = 0.9
    same_frame_distinct_roles: float = 0.7
    contrastive_language: float = 0.9


@dataclass(frozen=True)
class IdentityEdge:
    """Independent positive and negative evidence about two mention occurrences."""

    left: Mention
    right: Mention
    strength: float
    confidence: float
    evidence: tuple[str, ...] = ()
    negative_strength: float = 0.0
    negative_evidence: tuple[str, ...] = ()
    positive_cost: float = 1.0
    negative_cost: float = 1.0
    guard: str = ""

    @property
    def symbols(self) -> tuple[str, str]:
        return (self.left.canonical_symbol, self.right.canonical_symbol)

    @property
    def mention_ids(self) -> tuple[str, str]:
        return (_mention_key(self.left), _mention_key(self.right))

    @property
    def crosses_sentences(self) -> bool:
        return self.left.sentence_id != self.right.sentence_id


@dataclass
class IdentityGraph:
    """Mentions, the scored edges between them, and the resulting clusters."""

    nodes: tuple[Mention, ...] = ()
    edges: tuple[IdentityEdge, ...] = ()
    representatives: dict[str, str] = field(default_factory=dict)
    merged: tuple[IdentityEdge, ...] = ()

    def resolve(self, symbol: str) -> str:
        """Map a symbol to its cluster representative."""
        return self.representatives.get(symbol, symbol)

    def clusters(self) -> list[set[str]]:
        grouped: dict[str, set[str]] = {}
        for symbol, rep in self.representatives.items():
            grouped.setdefault(rep, set()).add(symbol)
        return [members for _, members in sorted(grouped.items())]

    def transport_cost(self, source: str, target: str) -> float:
        """Lowest accumulated cost across accepted symbol-identity edges."""
        if source == target:
            return 0.0
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for edge in self.merged:
            left, right = edge.symbols
            cost = min(2.0, edge.positive_cost + edge.negative_strength)
            adjacency.setdefault(left, []).append((right, cost))
            adjacency.setdefault(right, []).append((left, cost))
        frontier: list[tuple[float, str]] = [(0.0, source)]
        best = {source: 0.0}
        while frontier:
            frontier.sort(reverse=True)
            cost, symbol = frontier.pop()
            if symbol == target:
                return round(cost, 4)
            if cost != best.get(symbol):
                continue
            for neighbour, edge_cost in adjacency.get(symbol, []):
                candidate = cost + edge_cost
                if candidate < best.get(neighbour, float("inf")):
                    best[neighbour] = candidate
                    frontier.append((candidate, neighbour))
        return 1.0

    @property
    def merge_count(self) -> int:
        return len(self.merged)


def _compound_parts(symbol: str) -> Optional[tuple[str, str]]:
    """Split a snake_case compound into (modifier, head), or None if it is bare."""
    if "_" not in symbol:
        return None
    modifier, head = symbol.rsplit("_", 1)
    return modifier, head


def _normalized_surface(mention: Mention) -> str:
    return " ".join(mention.surface.strip().lower().split())


def _bare_and_compound(left: str, right: str) -> tuple[Optional[str], Optional[str]]:
    """Order a pair as (bare symbol, compound symbol), or (None, None) if neither."""
    left_is_compound, right_is_compound = "_" in left, "_" in right
    if right_is_compound and not left_is_compound:
        return left, right
    if left_is_compound and not right_is_compound:
        return right, left
    return None, None


def _mention_key(mention: Mention) -> str:
    if mention.mention_id:
        return mention.mention_id
    return f"{mention.sentence_id}:{mention.canonical_symbol}:{mention.char_span}"


def _kind_for(mention: Mention, kinds: dict[str, str]) -> Optional[str]:
    return kinds.get(_mention_key(mention)) or kinds.get(mention.canonical_symbol)


def _negative_evidence(
    left: Mention,
    right: Mention,
    kinds: dict[str, str],
    distinct_role_pairs: set[frozenset[str]],
    source_texts: dict[str, str],
) -> list[str]:
    left_symbol, right_symbol = left.canonical_symbol, right.canonical_symbol
    found: list[str] = []

    left_kind, right_kind = _kind_for(left, kinds), _kind_for(right, kinds)
    if left_kind and right_kind and left_kind != right_kind:
        found.append("kind_conflict")

    if left.mention_type == "pronoun" and right.mention_type == "pronoun":
        found.append("both_pronouns")

    if left.mention_type == "proper" and right.mention_type == "proper":
        left_surface, right_surface = _normalized_surface(left), _normalized_surface(right)
        if left_surface and right_surface:
            if left_surface not in right_surface and right_surface not in left_surface:
                found.append("proper_conflict")

    left_parts, right_parts = _compound_parts(left_symbol), _compound_parts(right_symbol)
    if left_parts and right_parts and left_parts[1] == right_parts[1]:
        if left_parts[0] != right_parts[0]:
            # The modifier is what distinguishes compounds sharing a head:
            # fish_eater and meat_eater are contrasted, not coreferent.
            found.append("modifier_conflict")

    if frozenset({_mention_key(left), _mention_key(right)}) in distinct_role_pairs:
        found.append("same_frame_distinct_roles")

    for mention in (left, right):
        text = source_texts.get(_mention_key(mention), "")
        if mention.char_span and text:
            start = max(0, mention.char_span[0] - 16)
            prefix = text[start : mention.char_span[0]].lower()
            if any(cue in prefix.split()[-3:] for cue in ("another", "different", "other")):
                found.append("contrastive_language")
                break
    return found


def _entity_evidence(
    left: Mention,
    right: Mention,
    kinds: dict[str, str],
    roles: dict[str, set[tuple[str, str]]],
) -> list[str]:
    """Evidence for two non-pronoun mentions denoting the same entity."""
    found: list[str] = []
    left_symbol, right_symbol = left.canonical_symbol, right.canonical_symbol

    left_surface, right_surface = _normalized_surface(left), _normalized_surface(right)
    if left_symbol == right_symbol:
        found.append("exact_symbol")
    if left_surface and left_surface == right_surface:
        # Equal surfaces with unequal symbols means canonicalization diverged
        # (a protected proper name against a lemmatized one, typically).
        found.append("surface_match")

    if left.mention_type == "proper" and right.mention_type == "proper":
        # The veto already rejected non-overlapping proper names, so reaching here
        # means one name contains the other ("Kebede" within "Kebede Alemu").
        found.append("proper_compat")

    bare, compound = _bare_and_compound(left_symbol, right_symbol)
    if bare is not None and compound is not None:
        if (
            compound.rsplit("_", 1)[-1] == bare
            and compound.count("_") - bare.count("_") <= _MAX_NAME_EXTENSION_TOKENS
        ):
            # A bare head against a compound built on it: eater / fish_eater.
            found.append("head_lemma")
        elif (
            compound.split("_", 1)[0] == bare
            and compound.count("_") - bare.count("_") <= _MAX_NAME_EXTENSION_TOKENS
            and "proper" in (
                left.mention_type,
                right.mention_type,
            )
        ):
            # A name gaining or shedding tokens: kebede / kebede_alemu. Gated on
            # properness because fish / fish_eater has the identical shape and is a
            # contrast rather than a coreference. The gate is one-sided on purpose:
            # a sentence-initial name reads as `common` to the extractor, which is
            # exactly the case this needs to catch.
            #
            # Bounded by token distance because properness does not discriminate an
            # organisation named after a qualifier: `human` and
            # `human_microbiome_action_consortium` are both proper and share a
            # leading token, but denote different entities.
            found.append("name_extension")

    left_kind, right_kind = _kind_for(left, kinds), _kind_for(right, kinds)
    if left_kind and left_kind == right_kind:
        found.append("kind_match")

    if roles.get(_mention_key(left), set()) & roles.get(_mention_key(right), set()):
        found.append("role_compat")

    return found


def _pronoun_evidence(
    pronoun: Mention,
    candidate: Mention,
    prominent: set[str],
) -> list[str]:
    """Evidence that `candidate` is the antecedent of `pronoun`.

    Deliberately thin: adjacency and syntactic prominence are the only signals the
    mention model actually supports. The `unambiguous` bonus is added later by the
    resolver, once every candidate for this pronoun has been scored.
    """
    if pronoun.canonical_symbol not in _ANAPHORIC_PRONOUNS:
        return []
    if candidate.mention_type == "pronoun":
        return []
    if not _adjacent(candidate.sentence_id, pronoun.sentence_id):
        return []

    found = ["pronoun_antecedent"]
    if _mention_key(candidate) in prominent:
        found.append("role_prominence")
    return found


def _adjacent(antecedent_sentence: str, pronoun_sentence: str) -> bool:
    """True when the antecedent is in the pronoun's sentence or the one before it.

    Sentence ids are opaque strings, so a trailing integer is used when both carry
    one and exact equality is the only test otherwise. Cataphora (a forward
    reference) is not resolved.
    """
    if antecedent_sentence == pronoun_sentence:
        return True
    left_index = _sentence_index(antecedent_sentence)
    right_index = _sentence_index(pronoun_sentence)
    if left_index is None or right_index is None:
        return False
    return 0 <= right_index - left_index <= 1


def _sentence_index(sentence_id: str) -> Optional[int]:
    digits = ""
    for char in reversed(sentence_id):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else None


def _confidence(evidence: Sequence[str]) -> float:
    """Each independent evidence type halves the remaining doubt."""
    return 1.0 - 0.5 ** len(evidence) if evidence else 0.0


class IdentityResolver:
    """Scores mention pairs and clusters the ones that clear the bar.

    `embedder` is optional and consulted last, only for pairs that deterministic
    evidence left just short of the threshold, because each call is an Ollama round
    trip on the ingest path.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_IDENTITY_THRESHOLD,
        weights: Optional[IdentityWeights] = None,
        max_mentions_per_sentence: int = DEFAULT_MAX_MENTIONS_PER_SENTENCE,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        embed_band: float = 0.3,
        ambiguity_margin: float = 0.15,
    ):
        self.threshold = threshold
        self.weights = weights or IdentityWeights()
        self.max_mentions_per_sentence = max_mentions_per_sentence
        self.embedder = embedder
        self.embed_band = embed_band
        self.ambiguity_margin = ambiguity_margin

    def resolve(self, senfs: Sequence[SENF]) -> IdentityGraph:
        mentions = self._collect_mentions(senfs)
        if len(mentions) < 2:
            return IdentityGraph(nodes=tuple(mentions))

        kinds = self._merged_kinds(senfs)
        roles = self._role_index(senfs)
        prominent = self._prominent_symbols(senfs)
        distinct_role_pairs = self._distinct_role_pairs(senfs)
        source_texts = self._source_texts(senfs)

        edges = self._score_pairs(
            mentions,
            kinds,
            roles,
            prominent,
            distinct_role_pairs,
            source_texts,
        )
        edges = self._apply_ambiguity(edges)
        edges = self._collapse_to_mention_pairs(edges)
        merged = tuple(edge for edge in edges if self._should_merge(edge))
        representatives = self._elect(mentions, merged)

        for edge in merged:
            logger.info(
                "SENF identity merge %s ~ %s strength=%.2f confidence=%.2f evidence=%s",
                edge.left.canonical_symbol,
                edge.right.canonical_symbol,
                edge.strength,
                edge.confidence,
                list(edge.evidence),
            )

        return IdentityGraph(
            nodes=tuple(mentions),
            edges=edges,
            representatives=representatives,
            merged=merged,
        )

    def _collect_mentions(self, senfs: Sequence[SENF]) -> list[Mention]:
        """Flatten mentions, capping per sentence since pair scoring is O(n²)."""
        collected: list[Mention] = []
        for senf in senfs:
            collected.extend(senf.mentions[: self.max_mentions_per_sentence])
        return collected

    @staticmethod
    def _merged_kinds(senfs: Sequence[SENF]) -> dict[str, str]:
        merged: dict[str, str] = {}
        for senf in senfs:
            for symbol, kind in senf.kinds.items():
                merged.setdefault(symbol, kind)
            for mention_id, kind in senf.mention_kinds.items():
                merged.setdefault(mention_id, kind)
        return merged

    @staticmethod
    def _role_index(senfs: Sequence[SENF]) -> dict[str, set[tuple[str, str]]]:
        index: dict[str, set[tuple[str, str]]] = {}
        for senf in senfs:
            for frame in senf.frames:
                for role in frame.roles:
                    if isinstance(role.filler, Mention):
                        key = _mention_key(role.filler)
                        index.setdefault(key, set()).add((frame.predicate_head, role.name))
        return index

    @staticmethod
    def _prominent_symbols(senfs: Sequence[SENF]) -> set[str]:
        prominent: set[str] = set()
        for senf in senfs:
            for frame in senf.frames:
                for role in frame.roles:
                    if role.name in _PROMINENT_ROLES and isinstance(role.filler, Mention):
                        prominent.add(_mention_key(role.filler))
        return prominent

    @staticmethod
    def _distinct_role_pairs(senfs: Sequence[SENF]) -> set[frozenset[str]]:
        pairs: set[frozenset[str]] = set()
        for senf in senfs:
            for frame in senf.frames:
                fillers = [
                    role.filler
                    for role in frame.roles
                    if isinstance(role.filler, Mention)
                ]
                for index, left in enumerate(fillers):
                    for right in fillers[index + 1 :]:
                        if _mention_key(left) != _mention_key(right):
                            pairs.add(frozenset({_mention_key(left), _mention_key(right)}))
        return pairs

    @staticmethod
    def _source_texts(senfs: Sequence[SENF]) -> dict[str, str]:
        out: dict[str, str] = {}
        for senf in senfs:
            text = next((frame.source_text for frame in senf.frames if frame.source_text), "")
            for mention in senf.mentions:
                out[_mention_key(mention)] = text
        return out

    def _score_pairs(
        self,
        mentions: Sequence[Mention],
        kinds: dict[str, str],
        roles: dict[str, set[tuple[str, str]]],
        prominent: set[str],
        distinct_role_pairs: set[frozenset[str]],
        source_texts: dict[str, str],
    ) -> tuple[IdentityEdge, ...]:
        vectors: dict[str, Sequence[float]] = {}
        scored: list[IdentityEdge] = []

        for index, left in enumerate(mentions):
            for right in mentions[index + 1 :]:
                evidence = self._evidence_for(left, right, kinds, roles, prominent)
                negative = _negative_evidence(
                    left,
                    right,
                    kinds,
                    distinct_role_pairs,
                    source_texts,
                )
                if not evidence and not negative:
                    continue
                strength = self._strength(evidence)
                extra = self._embedding_evidence(left, right, strength, vectors)
                if extra:
                    evidence = evidence + extra
                    strength = self._strength(evidence)

                negative_strength = self._negative_strength(negative)
                scored.append(
                    self._edge(
                        left,
                        right,
                        strength,
                        evidence,
                        negative_strength,
                        negative,
                    )
                )

        return tuple(scored)

    def _evidence_for(
        self,
        left: Mention,
        right: Mention,
        kinds: dict[str, str],
        roles: dict[str, set[tuple[str, str]]],
        prominent: set[str],
    ) -> list[str]:
        left_is_pronoun = left.mention_type == "pronoun"
        right_is_pronoun = right.mention_type == "pronoun"
        if left_is_pronoun:
            return _pronoun_evidence(left, right, prominent)
        if right_is_pronoun:
            return _pronoun_evidence(right, left, prominent)
        return _entity_evidence(left, right, kinds, roles)

    def _strength(self, evidence: Sequence[str]) -> float:
        return min(1.0, sum(getattr(self.weights, name, 0.0) for name in evidence))

    def _negative_strength(self, evidence: Sequence[str]) -> float:
        return min(1.0, sum(getattr(self.weights, name, 0.0) for name in evidence))

    def _embedding_evidence(
        self,
        left: Mention,
        right: Mention,
        strength: float,
        vectors: dict[str, Sequence[float]],
    ) -> list[str]:
        """Consulted only for near misses, to bound Ollama round trips.

        Pronoun pairs are excluded outright. The embedding of "It" carries no signal
        about its antecedent, and pronoun scoring is finished later anyway — the
        `unambiguous` bonus lands after this runs, so a pair judged near-miss here
        may already be well over the line by the time it is gated.
        """
        if self.embedder is None:
            return []
        if "pronoun" in (left.mention_type, right.mention_type):
            return []
        if strength >= self.threshold or self.threshold - strength > self.embed_band:
            return []
        try:
            left_vector = self._vector(left, vectors)
            right_vector = self._vector(right, vectors)
        except Exception as exc:
            logger.debug("SENF identity embedding unavailable: %s", exc)
            return []
        return ["embed_sim"] if _cosine(left_vector, right_vector) >= 0.8 else []

    def _vector(self, mention: Mention, vectors: dict[str, Sequence[float]]) -> Sequence[float]:
        symbol = mention.canonical_symbol
        if symbol not in vectors:
            vectors[symbol] = self.embedder(mention.surface or symbol)
        return vectors[symbol]

    @staticmethod
    def _edge(
        left: Mention,
        right: Mention,
        strength: float,
        evidence: Sequence[str],
        negative_strength: float = 0.0,
        negative_evidence: Sequence[str] = (),
    ) -> IdentityEdge:
        """Orient by symbol so an edge is stable regardless of mention order."""
        if right.canonical_symbol < left.canonical_symbol:
            left, right = right, left
        return IdentityEdge(
            left=left,
            right=right,
            strength=strength,
            confidence=_confidence(tuple(evidence) + tuple(negative_evidence)),
            evidence=tuple(evidence),
            negative_strength=negative_strength,
            negative_evidence=tuple(negative_evidence),
            positive_cost=round(1.0 - strength, 4),
            negative_cost=round(1.0 - negative_strength, 4),
            guard=f"{left.sentence_id}|{right.sentence_id}",
        )

    def _apply_ambiguity(self, edges: tuple[IdentityEdge, ...]) -> tuple[IdentityEdge, ...]:
        """Reward a pronoun's antecedent only when it clearly beats the runner-up.

        This is the safety valve for anaphora. With two plausible antecedents,
        neither is credited, both stay below the threshold, and the pronoun simply
        does not resolve — two weak edges rather than one confident wrong merge.
        """
        groups: dict[tuple[str, str], list[IdentityEdge]] = {}
        for edge in edges:
            pronoun = _pronoun_endpoint(edge)
            if pronoun is not None:
                key = (pronoun.canonical_symbol, pronoun.sentence_id)
                groups.setdefault(key, []).append(edge)

        promoted: dict[int, IdentityEdge] = {}
        for candidates in groups.values():
            # Rank distinct antecedent *symbols*, not mention pairs. The same symbol
            # can be mentioned in several retrieved SENFs, and counting those as
            # rival antecedents would fabricate a tie and suppress every resolution.
            best_per_symbol: dict[str, IdentityEdge] = {}
            for edge in candidates:
                symbol = _antecedent_symbol(edge)
                current = best_per_symbol.get(symbol)
                if current is None or edge.strength > current.strength:
                    best_per_symbol[symbol] = edge

            ranked = sorted(best_per_symbol.values(), key=lambda e: (-e.strength, e.symbols))
            best = ranked[0]
            runner_up = ranked[1].strength if len(ranked) > 1 else 0.0
            if best.strength - runner_up <= self.ambiguity_margin:
                continue
            evidence = best.evidence + ("unambiguous",)
            promoted[id(best)] = IdentityEdge(
                left=best.left,
                right=best.right,
                strength=self._strength(evidence),
                confidence=_confidence(evidence),
                evidence=evidence,
                negative_strength=best.negative_strength,
                negative_evidence=best.negative_evidence,
                positive_cost=round(1.0 - self._strength(evidence), 4),
                negative_cost=best.negative_cost,
                guard=best.guard,
            )

        return tuple(promoted.get(id(edge), edge) for edge in edges)

    @staticmethod
    def _collapse_to_mention_pairs(edges: tuple[IdentityEdge, ...]) -> tuple[IdentityEdge, ...]:
        """Keep one edge per mention pair while retaining occurrence evidence."""
        best: dict[tuple[str, str], IdentityEdge] = {}
        for edge in edges:
            key = tuple(sorted(edge.mention_ids))
            current = best.get(key)
            if current is None or (edge.strength, len(edge.evidence)) > (
                current.strength,
                len(current.evidence),
            ):
                best[key] = edge
        return tuple(
            sorted(best.values(), key=lambda e: (-e.strength, e.symbols, e.mention_ids))
        )

    def _should_merge(self, edge: IdentityEdge) -> bool:
        if edge.left.canonical_symbol == edge.right.canonical_symbol:
            return False
        if edge.strength < self.threshold:
            return False
        if edge.negative_strength >= 0.5:
            return False
        if edge.crosses_sentences and len(edge.evidence) < 2:
            # One signal is never enough to bind entities the text introduced
            # separately; this is the cheapest guard against a runaway merge.
            return False
        return True

    @staticmethod
    def _elect(
        mentions: Sequence[Mention], merged: Sequence[IdentityEdge]
    ) -> dict[str, str]:
        parent: dict[str, str] = {}

        def find(symbol: str) -> str:
            parent.setdefault(symbol, symbol)
            while parent[symbol] != symbol:
                parent[symbol] = parent[parent[symbol]]
                symbol = parent[symbol]
            return symbol

        for edge in merged:
            left_root, right_root = find(edge.symbols[0]), find(edge.symbols[1])
            if left_root != right_root:
                parent[right_root] = left_root

        ranks: dict[str, tuple[int, int, int, int, str]] = {}
        for position, mention in enumerate(mentions):
            symbol = mention.canonical_symbol
            type_rank = _TYPE_PRIORITY.get(mention.mention_type, 9)
            candidate = (
                1 if mention.mention_type == "pronoun" else 0,
                -symbol.count("_"),
                type_rank,
                position,
                symbol,
            )
            if symbol not in ranks or candidate < ranks[symbol]:
                ranks[symbol] = candidate

        clusters: dict[str, list[str]] = {}
        for symbol in parent:
            clusters.setdefault(find(symbol), []).append(symbol)

        representatives: dict[str, str] = {}
        for members in clusters.values():
            if len(members) < 2:
                continue
            winner = min(members, key=lambda s: ranks.get(s, (0, 0, 9, 0, s)))
            for member in members:
                representatives[member] = winner
        return representatives


def _pronoun_endpoint(edge: IdentityEdge) -> Optional[Mention]:
    if edge.left.mention_type == "pronoun":
        return edge.left
    if edge.right.mention_type == "pronoun":
        return edge.right
    return None


def _antecedent_symbol(edge: IdentityEdge) -> str:
    if edge.left.mention_type == "pronoun":
        return edge.right.canonical_symbol
    return edge.left.canonical_symbol


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def resolve_identity(
    senfs: Sequence[SENF],
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> IdentityGraph:
    """Convenience wrapper for callers that do not need to hold a resolver."""
    return IdentityResolver(threshold=threshold).resolve(senfs)
