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


@dataclass(frozen=True)
class IdentityEdge:
    """Graded evidence that two symbols denote the same entity.

    `evidence` holds only the signals that fired, so it doubles as the audit trail
    for a merge and as the reason a near-miss did not merge.
    """

    left: Mention
    right: Mention
    strength: float
    confidence: float
    evidence: tuple[str, ...] = ()

    @property
    def symbols(self) -> tuple[str, str]:
        return (self.left.canonical_symbol, self.right.canonical_symbol)

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
        """Map a symbol to its cluster representative.

        Total by design: an unknown symbol returns itself, so a caller rewriting a
        statement can call this on every token without first checking membership.
        """
        return self.representatives.get(symbol, symbol)

    def clusters(self) -> list[set[str]]:
        grouped: dict[str, set[str]] = {}
        for symbol, rep in self.representatives.items():
            grouped.setdefault(rep, set()).add(symbol)
        return [members for _, members in sorted(grouped.items())]

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


def _vetoed(left: Mention, right: Mention, kinds: dict[str, str]) -> Optional[str]:
    """Return the name of the disqualifying conflict, or None if the pair is open.

    A veto beats any amount of positive evidence. These are the cases where the
    text is actively telling us the two entities are distinct, and merging them
    would be the confidently-wrong-proof failure this module exists to avoid.
    """
    left_symbol, right_symbol = left.canonical_symbol, right.canonical_symbol

    left_kind, right_kind = kinds.get(left_symbol), kinds.get(right_symbol)
    if left_kind and right_kind and left_kind != right_kind:
        return "kind_conflict"

    if left.mention_type == "pronoun" and right.mention_type == "pronoun":
        return "both_pronouns"

    if left.mention_type == "proper" and right.mention_type == "proper":
        left_surface, right_surface = _normalized_surface(left), _normalized_surface(right)
        if left_surface and right_surface:
            if left_surface not in right_surface and right_surface not in left_surface:
                return "proper_conflict"

    left_parts, right_parts = _compound_parts(left_symbol), _compound_parts(right_symbol)
    if left_parts and right_parts and left_parts[1] == right_parts[1]:
        if left_parts[0] != right_parts[0]:
            # The modifier is what distinguishes compounds sharing a head:
            # fish_eater and meat_eater are contrasted, not coreferent.
            return "modifier_conflict"

    return None


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

    left_kind, right_kind = kinds.get(left_symbol), kinds.get(right_symbol)
    if left_kind and left_kind == right_kind:
        found.append("kind_match")

    if roles.get(left_symbol, set()) & roles.get(right_symbol, set()):
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
    if candidate.canonical_symbol in prominent:
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

        edges = self._score_pairs(mentions, kinds, roles, prominent)
        edges = self._apply_ambiguity(edges)
        edges = self._collapse_to_symbol_pairs(edges)
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
        return merged

    @staticmethod
    def _role_index(senfs: Sequence[SENF]) -> dict[str, set[tuple[str, str]]]:
        index: dict[str, set[tuple[str, str]]] = {}
        for senf in senfs:
            for frame in senf.frames:
                for role in frame.roles:
                    if isinstance(role.filler, Mention):
                        key = role.filler.canonical_symbol
                        index.setdefault(key, set()).add((frame.predicate_head, role.name))
        return index

    @staticmethod
    def _prominent_symbols(senfs: Sequence[SENF]) -> set[str]:
        prominent: set[str] = set()
        for senf in senfs:
            for frame in senf.frames:
                for role in frame.roles:
                    if role.name in _PROMINENT_ROLES and isinstance(role.filler, Mention):
                        prominent.add(role.filler.canonical_symbol)
        return prominent

    def _score_pairs(
        self,
        mentions: Sequence[Mention],
        kinds: dict[str, str],
        roles: dict[str, set[tuple[str, str]]],
        prominent: set[str],
    ) -> tuple[IdentityEdge, ...]:
        vectors: dict[str, Sequence[float]] = {}
        scored: list[IdentityEdge] = []

        for index, left in enumerate(mentions):
            for right in mentions[index + 1 :]:
                if left.canonical_symbol == right.canonical_symbol:
                    continue
                if _vetoed(left, right, kinds):
                    continue

                evidence = self._evidence_for(left, right, kinds, roles, prominent)
                if not evidence:
                    continue
                strength = self._strength(evidence)
                extra = self._embedding_evidence(left, right, strength, vectors)
                if extra:
                    evidence = evidence + extra
                    strength = self._strength(evidence)

                scored.append(self._edge(left, right, strength, evidence))

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
        left: Mention, right: Mention, strength: float, evidence: Sequence[str]
    ) -> IdentityEdge:
        """Orient by symbol so an edge is stable regardless of mention order."""
        if right.canonical_symbol < left.canonical_symbol:
            left, right = right, left
        return IdentityEdge(
            left=left,
            right=right,
            strength=strength,
            confidence=_confidence(evidence),
            evidence=tuple(evidence),
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
            )

        return tuple(promoted.get(id(edge), edge) for edge in edges)

    @staticmethod
    def _collapse_to_symbol_pairs(edges: tuple[IdentityEdge, ...]) -> tuple[IdentityEdge, ...]:
        """One edge per symbol pair — the graph resolves symbols, not occurrences."""
        best: dict[tuple[str, str], IdentityEdge] = {}
        for edge in edges:
            current = best.get(edge.symbols)
            if current is None or (edge.strength, len(edge.evidence)) > (
                current.strength,
                len(current.evidence),
            ):
                best[edge.symbols] = edge
        return tuple(sorted(best.values(), key=lambda e: (-e.strength, e.symbols)))

    def _should_merge(self, edge: IdentityEdge) -> bool:
        if edge.strength < self.threshold:
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
