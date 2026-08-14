"""SENF (Semantic Entity-Network Frame) data structures.

Plain dataclasses, no external dependencies and no imports from the rest of the
system, so this module stays cheap to import and safe to unit test.
"""

from dataclasses import dataclass, field
from typing import Literal as TypingLiteral, Optional, Union

MentionType = TypingLiteral["proper", "common", "pronoun", "nominal"]


@dataclass(frozen=True)
class Mention:
    """One reference to an entity, as it appeared at a specific place in the text."""

    surface: str
    canonical_symbol: str
    sentence_id: str
    mention_id: str = ""
    char_span: Optional[tuple[int, int]] = None
    mention_type: MentionType = "common"
    head_lemma: str = ""


@dataclass(frozen=True)
class Literal:
    """A role filler that is a value rather than an entity (numbers, dates)."""

    value: str
    literal_type: str = "unknown"


Filler = Union[Mention, Literal]


@dataclass(frozen=True)
class Role:
    name: str
    filler: Filler


@dataclass(frozen=True)
class ExemplarScore:
    kind: str
    exemplar: str
    distance: float
    reasons: tuple[str, ...] = ()


@dataclass
class SENFFrame:
    """One predication: a predicate head plus its filled roles."""

    frame_id: str
    predicate_head: str
    roles: list[Role] = field(default_factory=list)
    polarity: bool = True
    modality: Optional[str] = None
    time_ref: Optional[str] = None
    location_ref: Optional[str] = None
    source_sentence_id: str = ""
    source_text: str = ""

    def role(self, name: str) -> Optional[Role]:
        for role in self.roles:
            if role.name == name:
                return role
        return None

    def filler_symbols(self) -> list[str]:
        return [
            role.filler.canonical_symbol
            for role in self.roles
            if isinstance(role.filler, Mention)
        ]


@dataclass
class SENF:
    """All frames and mentions derived from one sentence."""

    senf_id: str
    sentence_id: str
    frames: list[SENFFrame] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)
    mention_kinds: dict[str, str] = field(default_factory=dict)
    exemplar_scores: dict[str, list[ExemplarScore]] = field(default_factory=dict)
    nearest_exemplars: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.frames and not self.mentions

    def symbols(self) -> set[str]:
        return {mention.canonical_symbol for mention in self.mentions}

    def kind_for(self, mention: Mention) -> Optional[str]:
        return self.mention_kinds.get(mention.mention_id) or self.kinds.get(
            mention.canonical_symbol
        )

    def nearest_exemplar_for(self, mention: Mention) -> Optional[str]:
        return self.nearest_exemplars.get(mention.mention_id)


# Bumped when the payload shape changes incompatibly. Readers must tolerate both
# an absent version (points written before SENF existed) and a future one, since
# a Qdrant collection outlives any single deploy.
SENF_PAYLOAD_VERSION = 2

SENF_PAYLOAD_KEY = "senf"


def _mention_to_payload(mention: Mention) -> dict:
    return {
        "surface": mention.surface,
        "symbol": mention.canonical_symbol,
        "sentence_id": mention.sentence_id,
        "mention_id": mention.mention_id,
        "char_span": list(mention.char_span) if mention.char_span else None,
        "mention_type": mention.mention_type,
        "head_lemma": mention.head_lemma,
    }


def _mention_from_payload(blob: dict) -> Mention:
    span = blob.get("char_span")
    return Mention(
        surface=blob.get("surface", ""),
        canonical_symbol=blob.get("symbol", ""),
        sentence_id=blob.get("sentence_id", ""),
        mention_id=blob.get("mention_id", ""),
        # JSON has no tuples, so a span round-trips as a list.
        char_span=(int(span[0]), int(span[1])) if span and len(span) == 2 else None,
        mention_type=blob.get("mention_type", "common"),
        head_lemma=blob.get("head_lemma", ""),
    )


def _filler_to_payload(filler: Filler) -> dict:
    if isinstance(filler, Mention):
        return {
            "kind": "mention",
            "mention_id": filler.mention_id,
            "symbol": filler.canonical_symbol,
        }
    return {"kind": "literal", "value": filler.value, "literal_type": filler.literal_type}


def _filler_from_payload(
    blob: dict,
    by_id: dict[str, Mention],
    by_symbol: dict[str, list[Mention]],
) -> Filler:
    if blob.get("kind") == "mention":
        mention_id = blob.get("mention_id", "")
        symbol = blob.get("symbol", "")
        candidates = by_symbol.get(symbol, [])
        return by_id.get(mention_id) or (candidates[0] if candidates else None) or Mention(
            surface=symbol,
            canonical_symbol=symbol,
            sentence_id="",
            mention_id=mention_id,
        )
    return Literal(blob.get("value", ""), blob.get("literal_type", "unknown"))


def _frame_to_payload(frame: SENFFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "predicate_head": frame.predicate_head,
        "roles": [
            {"name": role.name, "filler": _filler_to_payload(role.filler)}
            for role in frame.roles
        ],
        "polarity": frame.polarity,
        "modality": frame.modality,
        "time_ref": frame.time_ref,
        "location_ref": frame.location_ref,
        "source_sentence_id": frame.source_sentence_id,
        "source_text": frame.source_text,
    }


def _frame_from_payload(
    blob: dict,
    by_id: dict[str, Mention],
    by_symbol: dict[str, list[Mention]],
) -> SENFFrame:
    return SENFFrame(
        frame_id=blob.get("frame_id", ""),
        predicate_head=blob.get("predicate_head", ""),
        roles=[
            Role(
                role.get("name", ""),
                _filler_from_payload(role.get("filler", {}), by_id, by_symbol),
            )
            for role in blob.get("roles", [])
        ],
        polarity=bool(blob.get("polarity", True)),
        modality=blob.get("modality"),
        time_ref=blob.get("time_ref"),
        location_ref=blob.get("location_ref"),
        source_sentence_id=blob.get("source_sentence_id", ""),
        source_text=blob.get("source_text", ""),
    )


def senf_to_payload(senf: SENF) -> dict:
    """Serialize a SENF into a JSON-safe blob for the Qdrant point payload."""
    return {
        "senf_version": SENF_PAYLOAD_VERSION,
        "senf_id": senf.senf_id,
        "sentence_id": senf.sentence_id,
        "mentions": [_mention_to_payload(mention) for mention in senf.mentions],
        "frames": [_frame_to_payload(frame) for frame in senf.frames],
        "kinds": dict(senf.kinds),
        "mention_kinds": dict(senf.mention_kinds),
        "exemplar_scores": {
            mention_id: [
                {
                    "kind": score.kind,
                    "exemplar": score.exemplar,
                    "distance": score.distance,
                    "reasons": list(score.reasons),
                }
                for score in scores
            ]
            for mention_id, scores in senf.exemplar_scores.items()
        },
        "nearest_exemplars": dict(senf.nearest_exemplars),
        "constraints": list(senf.constraints),
    }


def senf_from_payload(blob: object) -> Optional[SENF]:
    """Rebuild a SENF from a stored blob, or return None if there isn't one."""
    if not isinstance(blob, dict):
        return None
    if not blob:
        return None
    version = blob.get("senf_version")
    if version is not None and not isinstance(version, int):
        return None
    if version is not None and version > SENF_PAYLOAD_VERSION:
        # Written by a newer deploy against the same collection.
        return None
    raw_mentions = blob.get("mentions", [])
    raw_frames = blob.get("frames", [])
    if not isinstance(raw_mentions, list) or not isinstance(raw_frames, list):
        return None
    # Fail closed on a structurally bad entry rather than skipping it. Dropping one
    # silently would change a frame's arity or a mention set with no signal, and
    # returning None just restores the pre-SENF path, which is always safe.
    if any(not isinstance(m, dict) for m in raw_mentions):
        return None
    if any(not isinstance(f, dict) for f in raw_frames):
        return None
    try:
        mentions = [_mention_from_payload(m) for m in raw_mentions]
        # Version-1 payloads have no mention ids. Give each occurrence a stable
        # local id so role references and identity scoring remain occurrence-based.
        mentions = [
            mention
            if mention.mention_id
            else Mention(
                surface=mention.surface,
                canonical_symbol=mention.canonical_symbol,
                sentence_id=mention.sentence_id,
                mention_id=f"{blob.get('sentence_id', '')}:m{index}",
                char_span=mention.char_span,
                mention_type=mention.mention_type,
                head_lemma=mention.head_lemma,
            )
            for index, mention in enumerate(mentions)
        ]
        by_id = {mention.mention_id: mention for mention in mentions if mention.mention_id}
        by_symbol: dict[str, list[Mention]] = {}
        for mention in mentions:
            by_symbol.setdefault(mention.canonical_symbol, []).append(mention)
        frames = [_frame_from_payload(f, by_id, by_symbol) for f in raw_frames]
        kinds = blob.get("kinds") or {}
        if not isinstance(kinds, dict):
            return None
        mention_kinds = blob.get("mention_kinds") or {}
        if not isinstance(mention_kinds, dict):
            return None
        raw_scores = blob.get("exemplar_scores") or {}
        if not isinstance(raw_scores, dict):
            return None
        exemplar_scores: dict[str, list[ExemplarScore]] = {}
        for mention_id, score_blobs in raw_scores.items():
            if not isinstance(score_blobs, list):
                return None
            exemplar_scores[str(mention_id)] = [
                ExemplarScore(
                    kind=str(score.get("kind", "")),
                    exemplar=str(score.get("exemplar", "")),
                    distance=float(score.get("distance", 1.0)),
                    reasons=tuple(str(reason) for reason in score.get("reasons", [])),
                )
                for score in score_blobs
                if isinstance(score, dict)
            ]
        nearest_exemplars = blob.get("nearest_exemplars") or {}
        constraints = blob.get("constraints") or []
        if not isinstance(nearest_exemplars, dict) or not isinstance(constraints, list):
            return None
        return SENF(
            senf_id=blob.get("senf_id", ""),
            sentence_id=blob.get("sentence_id", ""),
            frames=frames,
            mentions=mentions,
            kinds={str(k): str(v) for k, v in kinds.items()},
            mention_kinds={str(k): str(v) for k, v in mention_kinds.items()},
            exemplar_scores=exemplar_scores,
            nearest_exemplars={str(k): str(v) for k, v in nearest_exemplars.items()},
            constraints=[str(value) for value in constraints],
        )
    except Exception:
        return None
