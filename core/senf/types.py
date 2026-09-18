"""SENF (Semantic Entity-Network Frame) data structures and persistence."""

from dataclasses import dataclass, field
import math
from typing import Literal as TypingLiteral, NamedTuple, Optional, Union

MentionType = TypingLiteral["proper", "common", "pronoun", "nominal"]
ClauseRole = TypingLiteral["fact", "premise", "conclusion"]
Definiteness = TypingLiteral["definite", "indefinite", "demonstrative", "pronoun", "unknown"]
BranchType = TypingLiteral["actual", "counterfactual", "projected"]
PersistenceType = TypingLiteral["rigid", "flexible", "contingent", "temporal"]
EntityStatus = TypingLiteral["realized", "ghost", "unfulfilled"]
ACTUAL_BRANCH_ID = "actual_root"


class SourceSpan(NamedTuple):
    """Half-open source offsets, kept tuple-compatible for existing callers."""

    start: int
    end: int


@dataclass(frozen=True)
class ContextGuard:
    source_unit_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    speakers: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    time_refs: tuple[str, ...] = ()
    location_refs: tuple[str, ...] = ()
    branch_ids: tuple[str, ...] = ()
    validity_interval_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceUnit:
    source_unit_id: str
    sentence_id: str
    text: str
    char_span: SourceSpan


@dataclass(frozen=True)
class Context:
    source_unit_id: str
    speaker: Optional[str] = None
    modality: Optional[str] = None
    time_ref: Optional[str] = None
    location_ref: Optional[str] = None
    branch_id: str = ACTUAL_BRANCH_ID
    validity_interval_id: Optional[str] = None


@dataclass(frozen=True)
class BranchContext:
    branch_id: str
    parent_id: Optional[str]
    branch_type: BranchType
    probability: float


@dataclass(frozen=True)
class ValidityInterval:
    interval_id: str
    start: Optional[str]
    end: Optional[str]
    start_inclusive: bool = True
    end_inclusive: bool = True


@dataclass(frozen=True)
class EntityPersistence:
    entity_id: str
    persistence_type: PersistenceType
    status: EntityStatus = "realized"
    branch_id: str = ACTUAL_BRANCH_ID
    validity_interval_id: Optional[str] = None


@dataclass(frozen=True)
class Constraint:
    kind: str
    value: str
    frame_id: str
    source_unit_id: str


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_symbol: str


@dataclass(frozen=True)
class Mention:
    """One source occurrence of an entity."""

    surface: str
    canonical_symbol: str
    sentence_id: str
    entity_id: str = ""
    mention_id: str = ""
    char_span: Optional[SourceSpan] = None
    mention_type: MentionType = "common"
    head_lemma: str = ""
    source_unit_id: str = ""
    definiteness: Definiteness = "unknown"


@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    mention_id: str


@dataclass(frozen=True)
class KindRef:
    canonical_symbol: str


@dataclass(frozen=True)
class ValueRef:
    value: str
    value_type: str = "unknown"

@dataclass(frozen=True)
class FrameRef:
    frame_id: str


Filler = Union[EntityRef, KindRef, ValueRef, FrameRef]


@dataclass(frozen=True)
class Role:
    name: str
    filler: Filler
    position: int = 0


@dataclass(frozen=True)
class KindAssertion:
    entity_id: str
    kind: KindRef
    polarity: bool
    source_frame_id: str


@dataclass(frozen=True)
class ExemplarAlternative:
    kind: str
    exemplar: str
    distance: float
    reasons: tuple[str, ...] = ()
    guard: Optional[ContextGuard] = None


# The old name remains valid for callers that treat alternatives as scores.
ExemplarScore = ExemplarAlternative


@dataclass
class SENFFrame:
    frame_id: str
    predicate_head: str
    roles: list[Role] = field(default_factory=list)
    polarity: bool = True
    modality: Optional[str] = None
    time_ref: Optional[str] = None
    location_ref: Optional[str] = None
    source_sentence_id: str = ""
    source_text: str = ""
    source_atom_id: str = ""
    clause_role: ClauseRole = "fact"
    context: Optional[Context] = None
    frame_span: Optional[SourceSpan] = None
    clause_span: Optional[SourceSpan] = None

    def role(self, name: str) -> Optional[Role]:
        return next((role for role in self.roles if role.name == name), None)

@dataclass
class SENF:
    senf_id: str
    sentence_id: str
    frames: list[SENFFrame] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    kind_assertions: list[KindAssertion] = field(default_factory=list)
    exemplar_scores: dict[str, list[ExemplarAlternative]] = field(default_factory=dict)
    nearest_exemplars: dict[str, str] = field(default_factory=dict)
    active_exemplars: dict[str, list[str]] = field(default_factory=dict)
    source_units: list[SourceUnit] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    branches: list[BranchContext] = field(default_factory=lambda: [
        BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0)
    ])
    validity_intervals: list[ValidityInterval] = field(default_factory=list)
    entity_persistence: list[EntityPersistence] = field(default_factory=list)
    source_atoms: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.frames and not self.mentions

    def entity(self, entity_id: str) -> Optional[Entity]:
        return next((entity for entity in self.entities if entity.entity_id == entity_id), None)

    def symbols(self) -> set[str]:
        return {entity.canonical_symbol for entity in self.entities}

    def kind_for(self, mention: Mention) -> Optional[str]:
        return next(
            (
                assertion.kind.canonical_symbol
                for assertion in self.kind_assertions
                if assertion.entity_id == mention.entity_id and assertion.polarity
            ),
            None,
        )

    def nearest_exemplar_for(self, mention: Mention) -> Optional[str]:
        active = self.active_exemplars.get(mention.mention_id, ())
        if len(active) == 1:
            return active[0]
        return self.nearest_exemplars.get(mention.mention_id)

    def active_exemplars_for(self, mention: Mention) -> tuple[str, ...]:
        return tuple(self.active_exemplars.get(mention.mention_id, ()))

    @property
    def exemplar_alternatives(self) -> dict[str, list[ExemplarAlternative]]:
        """Guarded alternatives; legacy active-name maps remain a projection."""
        return {
            mention_id: [score for score in self.exemplar_scores.get(mention_id, ()) if score.exemplar in names]
            for mention_id, names in self.active_exemplars.items()
        }

    def exemplar_alternatives_for(self, mention: Mention) -> tuple[ExemplarAlternative, ...]:
        return tuple(self.exemplar_alternatives.get(mention.mention_id, ()))


SENF_PAYLOAD_VERSION = 6
SENF_PAYLOAD_KEY = "senf"


def _span_to_payload(span: Optional[SourceSpan]) -> Optional[dict[str, int]]:
    return {"start": span[0], "end": span[1]} if span is not None else None


def _span_from_payload(blob: object) -> Optional[SourceSpan]:
    if blob is None:
        return None
    if (
        not isinstance(blob, dict)
        or set(blob) != {"start", "end"}
        or type(blob.get("start")) is not int
        or type(blob.get("end")) is not int
        or blob["start"] < 0
        or blob["start"] > blob["end"]
    ):
        raise ValueError("invalid source span")
    return SourceSpan(blob["start"], blob["end"])


def _guard_to_payload(guard: Optional[ContextGuard]) -> Optional[dict]:
    if guard is None:
        return None
    return {
        "source_unit_ids": list(guard.source_unit_ids),
        "sentence_ids": list(guard.sentence_ids),
        "speakers": list(guard.speakers),
        "modalities": list(guard.modalities),
        "time_refs": list(guard.time_refs),
        "location_refs": list(guard.location_refs),
        "branch_ids": list(guard.branch_ids),
        "validity_interval_ids": list(guard.validity_interval_ids),
    }


def _guard_from_payload(blob: object) -> Optional[ContextGuard]:
    if blob is None:
        return None
    fields = (
        "source_unit_ids", "sentence_ids", "speakers", "modalities", "time_refs",
        "location_refs", "branch_ids", "validity_interval_ids",
    )
    if not isinstance(blob, dict) or set(blob) != set(fields):
        raise ValueError("invalid context guard")
    values = []
    for name in fields:
        value = blob.get(name)
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("invalid context guard values")
        values.append(tuple(value))
    if not values[0] or not values[1]:
        raise ValueError("context guard lacks provenance")
    return ContextGuard(*values)


def _filler_to_payload(filler: Filler) -> dict:
    if isinstance(filler, EntityRef):
        return {
            "type": "entity_ref",
            "entity_id": filler.entity_id,
            "mention_id": filler.mention_id,
        }
    if isinstance(filler, KindRef):
        return {"type": "kind_ref", "canonical_symbol": filler.canonical_symbol}
    if isinstance(filler, FrameRef):
        return {"type": "frame_ref", "frame_id": filler.frame_id}
    return {"type": "value_ref", "value": filler.value, "value_type": filler.value_type}


def _filler_from_payload(blob: object) -> Filler:
    if not isinstance(blob, dict):
        raise ValueError("filler must be an object")
    filler_type = blob.get("type")
    if (
        filler_type == "entity_ref"
        and isinstance(blob.get("entity_id"), str)
        and blob["entity_id"]
        and isinstance(blob.get("mention_id"), str)
        and blob["mention_id"]
    ):
        return EntityRef(blob["entity_id"], blob["mention_id"])
    if filler_type == "kind_ref" and isinstance(blob.get("canonical_symbol"), str) and blob["canonical_symbol"]:
        return KindRef(blob["canonical_symbol"])
    if filler_type == "frame_ref" and isinstance(blob.get("frame_id"), str) and blob["frame_id"]:
        return FrameRef(blob["frame_id"])
    if filler_type == "value_ref" and isinstance(blob.get("value"), str) and isinstance(blob.get("value_type", "unknown"), str):
        return ValueRef(blob["value"], blob.get("value_type", "unknown"))
    raise ValueError("unknown or malformed filler")


def senf_to_payload(senf: SENF) -> dict:
    """Serialize a SENF into the current JSON-safe payload."""
    return {
        "senf_version": SENF_PAYLOAD_VERSION,
        "senf_id": senf.senf_id,
        "sentence_id": senf.sentence_id,
        "entities": [
            {"entity_id": entity.entity_id, "canonical_symbol": entity.canonical_symbol}
            for entity in senf.entities
        ],
        "mentions": [
            {
                "mention_id": mention.mention_id,
                "entity_id": mention.entity_id,
                "surface": mention.surface,
                "canonical_symbol": mention.canonical_symbol,
                "sentence_id": mention.sentence_id,
                "char_span": _span_to_payload(mention.char_span),
                "mention_type": mention.mention_type,
                "head_lemma": mention.head_lemma,
                "source_unit_id": mention.source_unit_id,
                "definiteness": mention.definiteness,
            }
            for mention in senf.mentions
        ],
        "frames": [
            {
                "frame_id": frame.frame_id,
                "predicate_head": frame.predicate_head,
                "roles": [
                    {
                        "name": role.name,
                        "position": role.position,
                        "filler": _filler_to_payload(role.filler),
                    }
                    for role in frame.roles
                ],
                "polarity": frame.polarity,
                "modality": frame.modality,
                "time_ref": frame.time_ref,
                "location_ref": frame.location_ref,
                "source_sentence_id": frame.source_sentence_id,
                "source_text": frame.source_text,
                "source_atom_id": frame.source_atom_id,
                "clause_role": frame.clause_role,
                "frame_span": _span_to_payload(frame.frame_span),
                "clause_span": _span_to_payload(frame.clause_span),
                "context": {
                    "source_unit_id": frame.context.source_unit_id,
                    "speaker": frame.context.speaker,
                    "modality": frame.context.modality,
                    "time_ref": frame.context.time_ref,
                    "location_ref": frame.context.location_ref,
                    "branch_id": frame.context.branch_id,
                    "validity_interval_id": frame.context.validity_interval_id,
                } if frame.context else None,
            }
            for frame in senf.frames
        ],
        "kind_assertions": [
            {
                "entity_id": assertion.entity_id,
                "kind": assertion.kind.canonical_symbol,
                "polarity": assertion.polarity,
                "source_frame_id": assertion.source_frame_id,
            }
            for assertion in senf.kind_assertions
        ],
        "exemplar_scores": {
            key: [
                {
                    "kind": score.kind,
                    "exemplar": score.exemplar,
                    "distance": score.distance,
                    "reasons": list(score.reasons),
                    "guard": _guard_to_payload(score.guard),
                }
                for score in scores
            ]
            for key, scores in senf.exemplar_scores.items()
        },
        "nearest_exemplars": dict(senf.nearest_exemplars),
        "active_exemplars": {key: list(values) for key, values in senf.active_exemplars.items()},
        "source_units": [
            {
                "source_unit_id": unit.source_unit_id,
                "sentence_id": unit.sentence_id,
                "text": unit.text,
                "char_span": _span_to_payload(unit.char_span),
            }
            for unit in senf.source_units
        ],
        "constraints": [
            {
                "kind": constraint.kind,
                "value": constraint.value,
                "frame_id": constraint.frame_id,
                "source_unit_id": constraint.source_unit_id,
            }
            for constraint in senf.constraints
        ],
        "branches": [
            {
                "branch_id": branch.branch_id,
                "parent_id": branch.parent_id,
                "branch_type": branch.branch_type,
                "probability": branch.probability,
            }
            for branch in senf.branches
        ],
        "validity_intervals": [
            {
                "interval_id": interval.interval_id,
                "start": interval.start,
                "end": interval.end,
                "start_inclusive": interval.start_inclusive,
                "end_inclusive": interval.end_inclusive,
            }
            for interval in senf.validity_intervals
        ],
        "entity_persistence": [
            {
                "entity_id": item.entity_id,
                "persistence_type": item.persistence_type,
                "status": item.status,
                "branch_id": item.branch_id,
                "validity_interval_id": item.validity_interval_id,
            }
            for item in senf.entity_persistence
        ],
        "source_atoms": list(senf.source_atoms),
    }


def migrate_v5_payload(blob: object) -> Optional[dict]:
    """Convert a legacy v5 payload to v6 shape without weakening validation."""
    if (
        not isinstance(blob, dict)
        or type(blob.get("senf_version")) is not int
        or blob["senf_version"] != 5
    ):
        return None
    migrated = dict(blob)
    try:
        migrated["senf_version"] = SENF_PAYLOAD_VERSION
        migrated["source_units"] = [dict(item) for item in blob["source_units"]]
        for item in migrated["source_units"]:
            span = item.get("char_span")
            if not isinstance(span, list) or len(span) != 2:
                return None
            item["char_span"] = {"start": span[0], "end": span[1]}
        migrated["mentions"] = [dict(item) for item in blob["mentions"]]
        for item in migrated["mentions"]:
            span = item.get("char_span")
            if span is not None:
                if not isinstance(span, list) or len(span) != 2:
                    return None
                item["char_span"] = {"start": span[0], "end": span[1]}
        migrated["frames"] = [dict(item) for item in blob["frames"]]
        for item in migrated["frames"]:
            item["frame_span"] = None
            item["clause_span"] = None
        mentions = {
            item.get("mention_id"): item
            for item in migrated["mentions"]
            if isinstance(item, dict)
        }
        migrated["exemplar_scores"] = {}
        for key, scores in blob["exemplar_scores"].items():
            mention = mentions.get(key, {})
            contexts = [
                frame.get("context")
                for frame in migrated["frames"]
                if isinstance(frame, dict) and isinstance(frame.get("context"), dict)
                if any(
                    isinstance(role, dict)
                    and isinstance(role.get("filler"), dict)
                    and role["filler"].get("type") == "entity_ref"
                    and role["filler"].get("mention_id") == key
                    for role in frame.get("roles", ())
                )
            ]

            def unique_context_values(name: str) -> list[str]:
                return list(dict.fromkeys(
                    value for context in contexts
                    if isinstance((value := context.get(name)), str) and value
                ))

            source_unit_ids = list(dict.fromkeys(
                [mention.get("source_unit_id")]
                + [context.get("source_unit_id") for context in contexts]
            ))
            guard = {
                "source_unit_ids": [value for value in source_unit_ids if value],
                "sentence_ids": [mention.get("sentence_id")]
                if mention.get("sentence_id") else [],
                "speakers": unique_context_values("speaker"),
                "modalities": unique_context_values("modality"),
                "time_refs": unique_context_values("time_ref"),
                "location_refs": unique_context_values("location_ref"),
                "branch_ids": unique_context_values("branch_id"),
                "validity_interval_ids": unique_context_values("validity_interval_id"),
            }
            migrated["exemplar_scores"][key] = [
                dict(score, guard=guard) for score in scores
            ]
        return migrated
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def senf_from_payload(blob: object) -> Optional[SENF]:
    """Read strict v6 or migrate strict v5; all other payloads fail closed."""
    if not isinstance(blob, dict) or type(blob.get("senf_version")) is not int:
        return None
    legacy_v5 = blob["senf_version"] == 5
    if legacy_v5:
        blob = migrate_v5_payload(blob)
        if blob is None:
            return None
    elif blob["senf_version"] != SENF_PAYLOAD_VERSION:
        return None
    try:
        if not isinstance(blob.get("senf_id"), str) or not blob["senf_id"]:
            raise ValueError("invalid SENF id")
        if not isinstance(blob.get("sentence_id"), str) or not blob["sentence_id"]:
            raise ValueError("invalid SENF ids")
        for key in (
            "entities", "mentions", "frames", "kind_assertions", "source_units",
            "exemplar_scores", "nearest_exemplars", "active_exemplars", "constraints",
            "branches", "validity_intervals", "entity_persistence",
            "source_atoms",
        ):
            expected = dict if key in ("exemplar_scores", "nearest_exemplars", "active_exemplars") else list
            if not isinstance(blob.get(key), expected):
                raise ValueError(f"{key} must be a list")

        entities = []
        for item in blob["entities"]:
            if not isinstance(item, dict) or not isinstance(item.get("entity_id"), str) or not isinstance(item.get("canonical_symbol"), str):
                raise ValueError("malformed entity")
            entities.append(Entity(item["entity_id"], item["canonical_symbol"]))
        entity_ids = {entity.entity_id for entity in entities}
        if len(entity_ids) != len(entities) or any(not value for value in entity_ids) or any(not entity.canonical_symbol for entity in entities):
            raise ValueError("invalid entity")
        entity_symbols = {entity.entity_id: entity.canonical_symbol for entity in entities}

        branches = []
        for item in blob["branches"]:
            if not isinstance(item, dict):
                raise ValueError("malformed branch")
            probability = item.get("probability")
            if (
                not isinstance(item.get("branch_id"), str)
                or not item["branch_id"]
                or item.get("parent_id") is not None
                and (not isinstance(item["parent_id"], str) or not item["parent_id"])
                or item.get("branch_type") not in ("actual", "counterfactual", "projected")
                or type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError("malformed branch")
            branches.append(BranchContext(
                item["branch_id"], item.get("parent_id"), item["branch_type"],
                float(probability),
            ))
        branch_ids = {item.branch_id for item in branches}
        if len(branch_ids) != len(branches):
            raise ValueError("duplicate branch")

        intervals = []
        for item in blob["validity_intervals"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("interval_id"), str)
                or not item["interval_id"]
                or item.get("start") is not None
                and (not isinstance(item["start"], str) or not item["start"])
                or item.get("end") is not None
                and (not isinstance(item["end"], str) or not item["end"])
                or type(item.get("start_inclusive")) is not bool
                or type(item.get("end_inclusive")) is not bool
            ):
                raise ValueError("malformed validity interval")
            intervals.append(ValidityInterval(
                item["interval_id"], item.get("start"), item.get("end"),
                item["start_inclusive"], item["end_inclusive"],
            ))
        interval_ids = {item.interval_id for item in intervals}
        if len(interval_ids) != len(intervals):
            raise ValueError("duplicate validity interval")

        source_units = []
        for item in blob["source_units"]:
            if not isinstance(item, dict):
                raise ValueError("malformed source unit")
            span = _span_from_payload(item.get("char_span"))
            if (
                not isinstance(item.get("source_unit_id"), str) or not item["source_unit_id"]
                or item.get("sentence_id") != blob["sentence_id"]
                or not isinstance(item.get("text"), str)
                or span is None
            ):
                raise ValueError("malformed source unit")
            source_units.append(SourceUnit(item["source_unit_id"], item["sentence_id"], item["text"], span))
        source_unit_ids = {unit.source_unit_id for unit in source_units}
        if len(source_unit_ids) != len(source_units):
            raise ValueError("duplicate source unit")

        mentions = []
        for item in blob["mentions"]:
            if not isinstance(item, dict) or item.get("entity_id") not in entity_ids:
                raise ValueError("dangling mention")
            for key in ("surface", "canonical_symbol", "sentence_id", "mention_id", "entity_id", "head_lemma", "source_unit_id"):
                if not isinstance(item.get(key), str):
                    raise ValueError("malformed mention")
            if (
                not item["mention_id"]
                or not item["sentence_id"]
                or item["sentence_id"] != blob["sentence_id"]
                or not item["canonical_symbol"]
                or item["canonical_symbol"] != entity_symbols[item["entity_id"]]
            ):
                raise ValueError("inconsistent mention")
            mention_type = item.get("mention_type")
            if mention_type not in ("proper", "common", "pronoun", "nominal"):
                raise ValueError("invalid mention type")
            definiteness = item.get("definiteness")
            if definiteness not in ("definite", "indefinite", "demonstrative", "pronoun", "unknown"):
                raise ValueError("invalid definiteness")
            if item["source_unit_id"] not in source_unit_ids:
                raise ValueError("invalid mention source unit")
            span = _span_from_payload(item.get("char_span"))
            mentions.append(Mention(
                surface=item["surface"], canonical_symbol=item["canonical_symbol"],
                sentence_id=item["sentence_id"], entity_id=item["entity_id"],
                mention_id=item["mention_id"],
                char_span=span,
                mention_type=mention_type, head_lemma=item["head_lemma"],
                source_unit_id=item["source_unit_id"], definiteness=definiteness,
            ))
        if len({mention.mention_id for mention in mentions}) != len(mentions):
            raise ValueError("duplicate mention id")
        if {mention.entity_id for mention in mentions} != entity_ids:
            raise ValueError("orphan entity")

        frames = []
        for item in blob["frames"]:
            if not isinstance(item, dict) or not isinstance(item.get("roles"), list):
                raise ValueError("malformed frame")
            if "frame_span" not in item or "clause_span" not in item:
                raise ValueError("missing frame provenance span")
            for key in ("frame_id", "predicate_head", "source_sentence_id", "source_atom_id"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ValueError("malformed frame")
            if not isinstance(item.get("source_text"), str):
                raise ValueError("malformed frame source text")
            if item["source_sentence_id"] != blob["sentence_id"]:
                raise ValueError("frame belongs to another sentence")
            for key in ("modality", "time_ref", "location_ref"):
                if item.get(key) is not None and (
                    not isinstance(item[key], str) or not item[key]
                ):
                    raise ValueError(f"invalid {key}")
            if type(item.get("polarity")) is not bool:
                raise ValueError("invalid frame polarity")
            clause_role = item.get("clause_role")
            if clause_role not in ("fact", "premise", "conclusion"):
                raise ValueError("invalid clause role")
            frame_span = _span_from_payload(item.get("frame_span"))
            clause_span = _span_from_payload(item.get("clause_span"))
            if frame_span is not None and clause_span is not None and not (
                clause_span.start <= frame_span.start <= frame_span.end <= clause_span.end
            ):
                raise ValueError("frame span lies outside clause")
            raw_context = item.get("context")
            if not isinstance(raw_context, dict) or raw_context.get("source_unit_id") not in source_unit_ids:
                raise ValueError("invalid context")
            for key in ("speaker", "modality", "time_ref", "location_ref", "validity_interval_id"):
                if raw_context.get(key) is not None and (
                    not isinstance(raw_context[key], str) or not raw_context[key]
                ):
                    raise ValueError("invalid context value")
            context = Context(
                raw_context["source_unit_id"], raw_context.get("speaker"),
                raw_context.get("modality"), raw_context.get("time_ref"),
                raw_context.get("location_ref"), raw_context.get("branch_id", ""),
                raw_context.get("validity_interval_id"),
            )
            if context.branch_id not in branch_ids:
                raise ValueError("frame references missing branch")
            context_unit = next(
                unit for unit in source_units
                if unit.source_unit_id == context.source_unit_id
            )
            if clause_span is not None and not (
                context_unit.char_span.start <= clause_span.start
                and clause_span.end <= context_unit.char_span.end
            ):
                raise ValueError("clause span lies outside source unit")
            if (
                context.validity_interval_id is not None
                and context.validity_interval_id not in interval_ids
            ):
                raise ValueError("frame references missing validity interval")
            if (
                context.modality != item.get("modality")
                or context.time_ref != item.get("time_ref")
                or context.location_ref != item.get("location_ref")
            ):
                raise ValueError("frame context disagrees with frame metadata")
            roles = []
            for role in item["roles"]:
                if not isinstance(role, dict) or not isinstance(role.get("name"), str) or not role["name"] or type(role.get("position")) is not int or role["position"] < 0:
                    raise ValueError("malformed role")
                roles.append(Role(role["name"], _filler_from_payload(role.get("filler")), role["position"]))
            if [role.position for role in roles] != list(range(len(roles))):
                raise ValueError("role positions do not preserve arity")
            frames.append(SENFFrame(
                frame_id=item["frame_id"], predicate_head=item["predicate_head"], roles=roles,
                polarity=item["polarity"], modality=item.get("modality"), time_ref=item.get("time_ref"),
                location_ref=item.get("location_ref"), source_sentence_id=item.get("source_sentence_id", ""),
                source_text=item.get("source_text", ""), source_atom_id=item["source_atom_id"],
                clause_role=clause_role,
                context=context,
                frame_span=frame_span,
                clause_span=clause_span,
            ))
        frame_ids = {frame.frame_id for frame in frames}
        if len(frame_ids) != len(frames) or any(not value for value in frame_ids):
            raise ValueError("invalid frame ids")
        mention_by_id = {mention.mention_id: mention for mention in mentions}
        source_texts = {frame.source_text for frame in frames}
        if len(source_texts) > 1:
            raise ValueError("inconsistent source text")
        source_text = next(iter(source_texts), "")
        for unit in source_units:
            if unit.char_span[1] > len(source_text) or source_text[unit.char_span[0]:unit.char_span[1]] != unit.text:
                raise ValueError("source unit does not match source text")
        if source_text:
            for mention in mentions:
                if mention.char_span is not None:
                    start, end = mention.char_span
                    if end > len(source_text):
                        raise ValueError("span exceeds source text")
                    if source_text[start:end] != mention.surface:
                        raise ValueError("mention surface does not match its span")
                    unit = next(unit for unit in source_units if unit.source_unit_id == mention.source_unit_id)
                    if not (unit.char_span[0] <= start and end <= unit.char_span[1]):
                        raise ValueError("mention span crosses source unit")
        frame_by_id = {frame.frame_id: frame for frame in frames}
        frame_refs: dict[str, list[str]] = {frame.frame_id: [] for frame in frames}
        referenced_mentions: set[str] = set()
        source_unit_ids = {unit.source_unit_id for unit in source_units}
        for frame in frames:
            for role in frame.roles:
                if isinstance(role.filler, EntityRef):
                    mention = mention_by_id.get(role.filler.mention_id)
                    if mention is None or mention.entity_id != role.filler.entity_id:
                        raise ValueError("dangling entity or mention ref")
                    referenced_mentions.add(mention.mention_id)
                if isinstance(role.filler, FrameRef):
                    child = frame_by_id.get(role.filler.frame_id)
                    if child is None:
                        raise ValueError("dangling frame ref")
                    if (
                        child.source_atom_id != frame.source_atom_id
                        or child.clause_role != frame.clause_role
                    ):
                        raise ValueError("frame ref crosses provenance")
                    frame_refs[frame.frame_id].append(child.frame_id)
            if frame.context.source_unit_id not in source_unit_ids:
                raise ValueError("frame context references missing source unit")
        if referenced_mentions != set(mention_by_id):
            raise ValueError("orphan mention")

        visiting: set[str] = set()
        visited: set[str] = set()

        def validate_frame_refs(frame_id: str) -> None:
            if frame_id in visiting:
                raise ValueError("cyclic frame ref")
            if frame_id in visited:
                return
            visiting.add(frame_id)
            for child_id in frame_refs[frame_id]:
                validate_frame_refs(child_id)
            visiting.remove(frame_id)
            visited.add(frame_id)

        for frame_id in frame_ids:
            validate_frame_refs(frame_id)

        assertions = []
        for item in blob["kind_assertions"]:
            if not isinstance(item, dict) or item.get("entity_id") not in entity_ids or item.get("source_frame_id") not in frame_ids:
                raise ValueError("dangling kind assertion")
            if type(item.get("polarity")) is not bool or not isinstance(item.get("kind"), str) or not item["kind"]:
                raise ValueError("malformed kind assertion")
            assertion = KindAssertion(item["entity_id"], KindRef(item["kind"]), item["polarity"], item["source_frame_id"])
            source_frame = next(frame for frame in frames if frame.frame_id == assertion.source_frame_id)
            if not (
                source_frame.predicate_head == "IsA"
                and len(source_frame.roles) == 2
                and isinstance(source_frame.roles[0].filler, EntityRef)
                and source_frame.roles[0].filler.entity_id == assertion.entity_id
                and source_frame.roles[1].filler == assertion.kind
                and source_frame.polarity == assertion.polarity
            ):
                raise ValueError("kind assertion does not match its IsA frame")
            assertions.append(assertion)
        assertion_keys = {
            (
                assertion.source_frame_id,
                assertion.entity_id,
                assertion.kind.canonical_symbol,
                assertion.polarity,
            )
            for assertion in assertions
        }
        for frame in frames:
            if frame.predicate_head != "IsA":
                continue
            if not (
                len(frame.roles) == 2
                and isinstance(frame.roles[0].filler, EntityRef)
                and isinstance(frame.roles[1].filler, KindRef)
            ):
                raise ValueError("malformed IsA frame")
            expected = (
                frame.frame_id,
                frame.roles[0].filler.entity_id,
                frame.roles[1].filler.canonical_symbol,
                frame.polarity,
            )
            if expected not in assertion_keys:
                raise ValueError("IsA frame has no matching kind assertion")

        raw_scores = blob["exemplar_scores"]
        nearest = blob["nearest_exemplars"]
        active = blob["active_exemplars"]
        constraints = blob["constraints"]
        if not isinstance(raw_scores, dict) or not isinstance(nearest, dict) or not isinstance(active, dict) or not isinstance(constraints, list):
            raise ValueError("malformed annotations")
        mention_ids = set(mention_by_id)
        if any(not isinstance(key, str) or key not in mention_ids for key in raw_scores):
            raise ValueError("score key is not a mention")
        if any(
            not isinstance(key, str)
            or key not in mention_ids
            or not isinstance(value, str)
            or not value
            for key, value in nearest.items()
        ):
            raise ValueError("nearest exemplar is malformed")
        if any(
            key not in mention_ids or not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            for key, values in active.items()
        ):
            raise ValueError("active exemplars are malformed")
        scores = {}
        from core.senf.exemplars import DEFAULT_EXEMPLAR_REGISTRY
        from core.symbol_normalization import canonical_symbol

        for key, values in raw_scores.items():
            if not isinstance(values, list) or any(not isinstance(score, dict) for score in values):
                raise ValueError("malformed scores")
            parsed_scores = []
            for score in values:
                if "guard" not in score:
                    raise ValueError("missing exemplar guard")
                reasons = score.get("reasons")
                distance = score.get("distance")
                if (
                    not isinstance(score.get("kind"), str)
                    or not score["kind"]
                    or not isinstance(score.get("exemplar"), str)
                    or not score["exemplar"]
                    or type(distance) not in (int, float)
                    or not math.isfinite(distance)
                    or not 0.0 <= distance <= 1.0
                    or not isinstance(reasons, list)
                    or any(not isinstance(reason, str) for reason in reasons)
                ):
                    raise ValueError("malformed exemplar score")
                mention = mention_by_id[key]
                applicable_kinds = {
                    canonical_symbol(assertion.kind.canonical_symbol)
                    for assertion in assertions
                    if assertion.entity_id == mention.entity_id and assertion.polarity
                }
                applicable_kinds.update(
                    normalized
                    for value in (mention.canonical_symbol, mention.head_lemma)
                    if (normalized := canonical_symbol(value)) in DEFAULT_EXEMPLAR_REGISTRY
                )
                score_kind = canonical_symbol(score["kind"])
                registered_names = {
                    definition.name
                    for definition in DEFAULT_EXEMPLAR_REGISTRY.get(score_kind, ())
                }
                if score_kind not in applicable_kinds or score["exemplar"] not in registered_names:
                    raise ValueError("exemplar does not apply to mention kind")
                guard = _guard_from_payload(score.get("guard"))
                if guard is None:
                    raise ValueError("exemplar alternative is unguarded")
                mention_contexts = [
                    frame.context
                    for frame in frames
                    if any(
                        isinstance(role.filler, EntityRef)
                        and role.filler.mention_id == key
                        for role in frame.roles
                    )
                ]
                allowed_guard_values = {
                    "speakers": {context.speaker for context in mention_contexts if context.speaker},
                    "modalities": {context.modality for context in mention_contexts if context.modality},
                    "time_refs": {context.time_ref for context in mention_contexts if context.time_ref},
                    "location_refs": {context.location_ref for context in mention_contexts if context.location_ref},
                    "branch_ids": {context.branch_id for context in mention_contexts if context.branch_id},
                    "validity_interval_ids": {
                        context.validity_interval_id
                        for context in mention_contexts
                        if context.validity_interval_id
                    },
                }
                if (
                    key not in mention_ids
                    or mention_by_id[key].source_unit_id not in guard.source_unit_ids
                    or mention_by_id[key].sentence_id not in guard.sentence_ids
                    or any(value not in source_unit_ids for value in guard.source_unit_ids)
                    or any(value != blob["sentence_id"] for value in guard.sentence_ids)
                ):
                    raise ValueError("exemplar guard disagrees with mention")
                if any(
                    not set(getattr(guard, name)) <= allowed
                    for name, allowed in allowed_guard_values.items()
                ):
                    raise ValueError("exemplar guard invents mention context")
                parsed_scores.append(ExemplarAlternative(
                    score["kind"], score["exemplar"], float(distance), tuple(reasons), guard
                ))
            scores[key] = parsed_scores
            if len({score.exemplar for score in parsed_scores}) != len(parsed_scores):
                raise ValueError("duplicate exemplar score")
        for key, values in active.items():
            scored_names = {score.exemplar for score in scores.get(key, ())}
            if not set(values) <= scored_names:
                raise ValueError("active exemplar has no score")
        for key, value in nearest.items():
            if active.get(key) != [value]:
                raise ValueError("nearest exemplar must be the sole active alternative")
        parsed_constraints = []
        for item in constraints:
            if (
                not isinstance(item, dict)
                or item.get("kind") not in ("modality", "time_ref", "location_ref")
                or not isinstance(item.get("value"), str) or not item["value"]
                or item.get("frame_id") not in frame_ids
                or item.get("source_unit_id") not in source_unit_ids
            ):
                raise ValueError("malformed constraint")
            frame = frame_by_id[item["frame_id"]]
            if (
                getattr(frame, item["kind"]) != item["value"]
                or getattr(frame.context, item["kind"]) != item["value"]
                or frame.context.source_unit_id != item["source_unit_id"]
            ):
                raise ValueError("constraint disagrees with its frame")
            parsed_constraints.append(Constraint(
                item["kind"], item["value"], item["frame_id"], item["source_unit_id"]
            ))
        constraint_keys = {
            (item.kind, item.value, item.frame_id, item.source_unit_id)
            for item in parsed_constraints
        }
        if len(constraint_keys) != len(parsed_constraints):
            raise ValueError("duplicate constraint")
        expected_constraints = {
            (kind, value, frame.frame_id, frame.context.source_unit_id)
            for frame in frames
            for kind in ("modality", "time_ref", "location_ref")
            if (value := getattr(frame, kind)) is not None
        }
        if constraint_keys != expected_constraints:
            raise ValueError("constraints do not match frame metadata")
        persistence = []
        for item in blob["entity_persistence"]:
            if (
                not isinstance(item, dict)
                or item.get("entity_id") not in entity_ids
                or item.get("persistence_type") not in ("rigid", "flexible", "contingent", "temporal")
                or item.get("status") not in ("realized", "ghost", "unfulfilled")
                or item.get("branch_id") not in branch_ids
                or item.get("validity_interval_id") is not None
                and item["validity_interval_id"] not in interval_ids
            ):
                raise ValueError("malformed entity persistence")
            persistence.append(EntityPersistence(
                item["entity_id"], item["persistence_type"], item["status"],
                item["branch_id"], item.get("validity_interval_id"),
            ))
        if len({(item.entity_id, item.branch_id) for item in persistence}) != len(persistence):
            raise ValueError("duplicate entity persistence")
        source_atoms = blob["source_atoms"]
        if any(not isinstance(atom, str) or not atom for atom in source_atoms):
            raise ValueError("malformed source atoms")
        from core.statement_validation import parse_expression, validate_statements

        valid_atoms, rejected_atoms = validate_statements(source_atoms)
        if rejected_atoms or valid_atoms != source_atoms:
            raise ValueError("invalid source atoms")
        accepted_atom_ids = {
            parts[1]
            for atom in source_atoms
            if len(parts := parse_expression(atom)) == 4
        }
        if any(frame.source_atom_id not in accepted_atom_ids for frame in frames):
            raise ValueError("frame source is not retained")
        parsed = SENF(
            senf_id=blob["senf_id"], sentence_id=blob["sentence_id"], frames=frames,
            entities=entities, mentions=mentions, kind_assertions=assertions,
            exemplar_scores=scores, nearest_exemplars=dict(nearest),
            active_exemplars={key: list(values) for key, values in active.items()},
            source_units=source_units, constraints=parsed_constraints,
            branches=branches, validity_intervals=intervals,
            entity_persistence=persistence,
            source_atoms=list(source_atoms),
        )
        from core.senf.temporal import validate_temporal_model

        validate_temporal_model(parsed)
        if legacy_v5:
            from core.senf.extractor import infer_frame_spans

            infer_frame_spans(parsed)
        return parsed
    except (KeyError, TypeError, ValueError):
        return None
