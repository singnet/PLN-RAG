"""SENF (Semantic Entity-Network Frame) v3 data structures."""

from dataclasses import dataclass, field
import math
from typing import Literal as TypingLiteral, Optional, Union

MentionType = TypingLiteral["proper", "common", "pronoun", "nominal"]
ClauseRole = TypingLiteral["fact", "premise", "conclusion"]


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
    char_span: Optional[tuple[int, int]] = None
    mention_type: MentionType = "common"
    head_lemma: str = ""


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
class ExemplarScore:
    kind: str
    exemplar: str
    distance: float
    reasons: tuple[str, ...] = ()


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
    exemplar_scores: dict[str, list[ExemplarScore]] = field(default_factory=dict)
    nearest_exemplars: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

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
        return self.nearest_exemplars.get(mention.mention_id)


SENF_PAYLOAD_VERSION = 3
SENF_PAYLOAD_KEY = "senf"


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
    """Serialize a SENF v3 into a JSON-safe payload."""
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
                "char_span": list(mention.char_span) if mention.char_span else None,
                "mention_type": mention.mention_type,
                "head_lemma": mention.head_lemma,
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
                }
                for score in scores
            ]
            for key, scores in senf.exemplar_scores.items()
        },
        "nearest_exemplars": dict(senf.nearest_exemplars),
        "constraints": list(senf.constraints),
    }


def senf_from_payload(blob: object) -> Optional[SENF]:
    """Read only v3; absent, old, future, and malformed payloads fail closed."""
    if not isinstance(blob, dict) or type(blob.get("senf_version")) is not int:
        return None
    if blob["senf_version"] != SENF_PAYLOAD_VERSION:
        return None
    try:
        if not isinstance(blob.get("senf_id"), str) or not blob["senf_id"]:
            raise ValueError("invalid SENF id")
        if not isinstance(blob.get("sentence_id"), str) or not blob["sentence_id"]:
            raise ValueError("invalid SENF ids")
        for key in ("entities", "mentions", "frames", "kind_assertions"):
            if not isinstance(blob.get(key), list):
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

        mentions = []
        for item in blob["mentions"]:
            if not isinstance(item, dict) or item.get("entity_id") not in entity_ids:
                raise ValueError("dangling mention")
            for key in ("surface", "canonical_symbol", "sentence_id", "mention_id", "entity_id", "head_lemma"):
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
            span = item.get("char_span")
            if span is not None and (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)):
                raise ValueError("invalid span")
            if span is not None and (span[0] < 0 or span[0] > span[1]):
                raise ValueError("invalid span bounds")
            mentions.append(Mention(
                surface=item["surface"], canonical_symbol=item["canonical_symbol"],
                sentence_id=item["sentence_id"], entity_id=item["entity_id"],
                mention_id=item["mention_id"],
                char_span=(span[0], span[1]) if span is not None else None,
                mention_type=mention_type, head_lemma=item["head_lemma"],
            ))
        if len({mention.mention_id for mention in mentions}) != len(mentions):
            raise ValueError("duplicate mention id")

        frames = []
        for item in blob["frames"]:
            if not isinstance(item, dict) or not isinstance(item.get("roles"), list):
                raise ValueError("malformed frame")
            for key in ("frame_id", "predicate_head", "source_sentence_id", "source_atom_id"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ValueError("malformed frame")
            if not isinstance(item.get("source_text"), str):
                raise ValueError("malformed frame source text")
            if item["source_sentence_id"] != blob["sentence_id"]:
                raise ValueError("frame belongs to another sentence")
            for key in ("modality", "time_ref", "location_ref"):
                if item.get(key) is not None and not isinstance(item[key], str):
                    raise ValueError(f"invalid {key}")
            if type(item.get("polarity")) is not bool:
                raise ValueError("invalid frame polarity")
            clause_role = item.get("clause_role")
            if clause_role not in ("fact", "premise", "conclusion"):
                raise ValueError("invalid clause role")
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
            ))
        frame_ids = {frame.frame_id for frame in frames}
        if len(frame_ids) != len(frames) or any(not value for value in frame_ids):
            raise ValueError("invalid frame ids")
        mention_by_id = {mention.mention_id: mention for mention in mentions}
        source_texts = {frame.source_text for frame in frames}
        if len(source_texts) > 1:
            raise ValueError("inconsistent source text")
        source_text = next(iter(source_texts), "")
        if source_text:
            for mention in mentions:
                if mention.char_span is not None:
                    start, end = mention.char_span
                    if end > len(source_text):
                        raise ValueError("span exceeds source text")
                    if source_text[start:end] != mention.surface:
                        raise ValueError("mention surface does not match its span")
        frame_by_id = {frame.frame_id: frame for frame in frames}
        frame_refs: dict[str, list[str]] = {frame.frame_id: [] for frame in frames}
        for frame in frames:
            for role in frame.roles:
                if isinstance(role.filler, EntityRef):
                    mention = mention_by_id.get(role.filler.mention_id)
                    if mention is None or mention.entity_id != role.filler.entity_id:
                        raise ValueError("dangling entity or mention ref")
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

        raw_scores = blob.get("exemplar_scores", {})
        nearest = blob.get("nearest_exemplars", {})
        constraints = blob.get("constraints", [])
        if not isinstance(raw_scores, dict) or not isinstance(nearest, dict) or not isinstance(constraints, list):
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
        if any(not isinstance(value, str) for value in constraints):
            raise ValueError("constraint must be a string")
        scores = {}
        for key, values in raw_scores.items():
            if not isinstance(values, list) or any(not isinstance(score, dict) for score in values):
                raise ValueError("malformed scores")
            parsed_scores = []
            for score in values:
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
                parsed_scores.append(ExemplarScore(
                    score["kind"], score["exemplar"], float(distance), tuple(reasons)
                ))
            scores[key] = parsed_scores
        return SENF(
            senf_id=blob["senf_id"], sentence_id=blob["sentence_id"], frames=frames,
            entities=entities, mentions=mentions, kind_assertions=assertions,
            exemplar_scores=scores, nearest_exemplars=dict(nearest),
            constraints=list(constraints),
        )
    except (KeyError, TypeError, ValueError):
        return None
