"""SENF (Semantic Entity-Network Frame) data structures.

Plain dataclasses, no external dependencies and no imports from the rest of the
system, so this module stays cheap to import and safe to unit test.

SENF is a *sidecar* representation: it describes the entities and frames behind a
set of already-canonicalized MeTTa atoms without changing them. `ParseResult`
deliberately keeps its two fields — SENF lives in the wrapper parser's own state
and reaches the atomspace only as bridge atoms, so parsers that never produce it
do not have to reason about it.

Identity, weave, and exemplar types are added by the commits that first use them.
"""

from dataclasses import dataclass, field
from typing import Literal as TypingLiteral, Optional, Union

MentionType = TypingLiteral["proper", "common", "pronoun", "nominal"]


@dataclass(frozen=True)
class Mention:
    """One reference to an entity, as it appeared at a specific place in the text.

    `surface` is what the text said and `canonical_symbol` is what the atom says;
    they differ whenever normalization did anything ("Fish Eaters" -> fish_eater).
    Keeping both is what lets identity resolution weigh surface evidence without
    re-deriving symbols.
    """

    surface: str
    canonical_symbol: str
    sentence_id: str
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


@dataclass
class SENFFrame:
    """One predication: a predicate head plus its filled roles.

    `polarity` False marks negation, carried here rather than as a wrapper frame
    so that a negated frame still aligns with its positive counterpart during
    weave — the alignment is the point, the disagreement is the signal.
    """

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
    """All frames and mentions derived from one sentence.

    `kinds` maps a symbol to its inferred kind (from IsA atoms), which is the
    cheapest identity signal available and the reason IsA gets special handling
    in the extractor.
    """

    senf_id: str
    sentence_id: str
    frames: list[SENFFrame] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.frames and not self.mentions

    def symbols(self) -> set[str]:
        return {mention.canonical_symbol for mention in self.mentions}
