from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from core.senf.types import SENF, SENFFrame


_HEAD_EXACT = 0.5
_HEAD_PARTIAL = 0.2
_ROLE_MATCH = 0.4
_POLARITY_AGREE = 0.1
_POLARITY_CONFLICT = -0.3

# Below this a pair is noise: a shared token in the predicate head and nothing else.
MIN_PAIR_SCORE = 0.3


@dataclass(frozen=True)
class FramePair:
    query_frame_id: str
    source_frame_id: str
    score: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeaveResult:
    """Alignment of a question against the corpus.

    `distortion` is 0.0 when every query frame found a strong partner and 1.0 when
    none did, so it reads as "how much of this question is unsupported".
    """

    pairs: tuple[FramePair, ...] = ()
    distortion: float = 0.0
    grounded_symbols: frozenset[str] = frozenset()
    role_signatures: frozenset[tuple[str, str]] = frozenset()

    @property
    def aligned(self) -> bool:
        return bool(self.pairs)


def _head_score(left: str, right: str) -> tuple[float, Optional[str]]:
    if left == right:
        return _HEAD_EXACT, "head_exact"
    left_tokens = set(_split_head(left))
    right_tokens = set(_split_head(right))
    if left_tokens & right_tokens:
        return _HEAD_PARTIAL, "head_overlap"
    return 0.0, None


def _split_head(head: str) -> list[str]:
    """Split PascalCase or snake_case predicate heads into comparable tokens."""
    out: list[str] = []
    current = ""
    for char in head.replace("_", " "):
        if char == " ":
            if current:
                out.append(current.lower())
            current = ""
            continue
        if char.isupper() and current:
            out.append(current.lower())
            current = char
        else:
            current += char
    if current:
        out.append(current.lower())
    return out


def _pair_score(
    query_frame: SENFFrame,
    source_frame: SENFFrame,
    resolve: Callable[[str], str],
) -> tuple[float, tuple[str, ...]]:
    score, head_evidence = _head_score(
        query_frame.predicate_head, source_frame.predicate_head
    )
    evidence: list[str] = [head_evidence] if head_evidence else []

    query_symbols = {resolve(s) for s in query_frame.filler_symbols()}
    source_symbols = {resolve(s) for s in source_frame.filler_symbols()}
    shared = query_symbols & source_symbols
    if shared and query_symbols:
        score += _ROLE_MATCH * (len(shared) / len(query_symbols))
        evidence.append("role_filler")

    if score > 0.0:
        if query_frame.polarity == source_frame.polarity:
            score += _POLARITY_AGREE
            evidence.append("polarity_agree")
        else:
            score += _POLARITY_CONFLICT
            evidence.append("polarity_conflict")

    return score, tuple(evidence)


def weave(
    query: SENF,
    sources: Sequence[SENF],
    resolve: Optional[Callable[[str], str]] = None,
) -> WeaveResult:
    """Greedy best-first pairing of query frames against source frames.

    Greedy rather than optimal: frame counts per sentence are small, and a
    best-first pass is stable and O(n log n) where an assignment solver would add
    a dependency for a difference no downstream signal can observe.
    """
    resolve = resolve or (lambda symbol: symbol)
    source_frames = [frame for senf in sources for frame in senf.frames]
    if not query.frames or not source_frames:
        return WeaveResult(distortion=1.0 if query.frames else 0.0)

    scored: list[tuple[float, tuple[str, ...], SENFFrame, SENFFrame]] = []
    for query_frame in query.frames:
        for source_frame in source_frames:
            score, evidence = _pair_score(query_frame, source_frame, resolve)
            if score >= MIN_PAIR_SCORE:
                scored.append((score, evidence, query_frame, source_frame))

    # Ties broken on frame ids so the result does not depend on dict ordering.
    scored.sort(key=lambda item: (-item[0], item[2].frame_id, item[3].frame_id))

    used_query: set[str] = set()
    used_source: set[str] = set()
    pairs: list[FramePair] = []
    grounded: set[str] = set()
    signatures: set[tuple[str, str]] = set()
    total = 0.0

    for score, evidence, query_frame, source_frame in scored:
        if query_frame.frame_id in used_query or source_frame.frame_id in used_source:
            continue
        used_query.add(query_frame.frame_id)
        used_source.add(source_frame.frame_id)
        pairs.append(
            FramePair(query_frame.frame_id, source_frame.frame_id, score, evidence)
        )
        total += min(score, 1.0)
        for symbol in source_frame.filler_symbols():
            grounded.add(resolve(symbol))
            signatures.add((source_frame.predicate_head, resolve(symbol)))

    distortion = max(0.0, min(1.0, 1.0 - total / len(query.frames)))
    return WeaveResult(
        pairs=tuple(pairs),
        distortion=distortion,
        grounded_symbols=frozenset(grounded),
        role_signatures=frozenset(signatures),
    )
