from dataclasses import dataclass
from typing import Mapping, Sequence

from core.senf.types import ContextGuard, EntityRef, ExemplarScore, Mention, SENF
from core.symbol_normalization import canonical_symbol


@dataclass(frozen=True)
class ExemplarDefinition:
    name: str
    cues: tuple[str, ...] = ()


DEFAULT_EXEMPLAR_REGISTRY: Mapping[str, tuple[ExemplarDefinition, ...]] = {
    "camera": (
        ExemplarDefinition("professional_camera", ("nikon", "dslr", "professional")),
        ExemplarDefinition("consumer_camera", ("consumer", "family", "home camera")),
        ExemplarDefinition("phone_camera", ("phone", "smartphone", "mobile")),
        ExemplarDefinition("security_camera", ("security", "surveillance", "cctv")),
    ),
    "game": (
        ExemplarDefinition("chess_game", ("strategy", "strategic", "chess", "patience")),
        ExemplarDefinition("football_game", ("football", "physical", "exhausting", "exhausted")),
        ExemplarDefinition("childrens_game", ("children", "playground", "toy")),
        ExemplarDefinition("video_game", ("video", "console", "computer", "online")),
        ExemplarDefinition("generic_game"),
    ),
    "bird": (
        ExemplarDefinition("robin", ("robin", "songbird")),
        ExemplarDefinition("eagle", ("eagle", "raptor")),
        ExemplarDefinition("penguin", ("penguin", "antarctic", "swim")),
        ExemplarDefinition("ostrich", ("ostrich", "flightless", "savanna")),
    ),
    "treatment": (
        ExemplarDefinition("drug_treatment", ("drug", "medication", "dose", "pharmacological")),
        ExemplarDefinition("surgical_treatment", ("surgery", "surgical", "operation")),
        ExemplarDefinition("behavioral_treatment", ("behavioral", "therapy", "lifestyle")),
    ),
    "lens": (
        ExemplarDefinition("camera_lens", ("camera", "nikon", "cracked")),
        ExemplarDefinition("standalone_lens", ("borrowed", "borrow", "separate lens", "separately")),
    ),
}


def _mention_context(senf: SENF, mention: Mention, radius: int = 120) -> str:
    unit = next(
        (unit for unit in senf.source_units if unit.source_unit_id == mention.source_unit_id),
        None,
    )
    if unit is None:
        return mention.surface.lower()
    text = unit.text
    if mention.char_span is None:
        return text.lower()
    start, end = mention.char_span
    local_start = start - unit.char_span[0]
    local_end = end - unit.char_span[0]
    return text[max(0, local_start - radius) : min(len(text), local_end + radius)].lower()


def _mention_kinds(
    senf: SENF,
    mention: Mention,
    registry: Mapping[str, Sequence[ExemplarDefinition]] = DEFAULT_EXEMPLAR_REGISTRY,
) -> tuple[str, ...]:
    """Positive asserted kinds plus bounded lexical registry applicability."""
    kinds = {
        canonical_symbol(assertion.kind.canonical_symbol): assertion.kind.canonical_symbol
        for assertion in senf.kind_assertions
        if assertion.entity_id == mention.entity_id and assertion.polarity
    }
    lexical_candidates = (mention.canonical_symbol, mention.head_lemma)
    for candidate in lexical_candidates:
        key = canonical_symbol(candidate)
        if key in registry:
            kinds.setdefault(key, candidate)
    return tuple(sorted(kinds.values()))


def _context_guard(senf: SENF, mention: Mention) -> ContextGuard:
    contexts = [
        frame.context
        for frame in senf.frames
        if frame.context is not None
        if any(
            isinstance(role.filler, EntityRef)
            and role.filler.mention_id == mention.mention_id
            for role in frame.roles
        )
    ]

    def values(name: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            value for context in contexts if (value := getattr(context, name))
        ))

    return ContextGuard(
        source_unit_ids=tuple(dict.fromkeys(
            [mention.source_unit_id]
            + [context.source_unit_id for context in contexts]
        )),
        sentence_ids=(mention.sentence_id,),
        speakers=values("speaker"),
        modalities=values("modality"),
        time_refs=values("time_ref"),
        location_refs=values("location_ref"),
        branch_ids=values("branch_id"),
        validity_interval_ids=values("validity_interval_id"),
    )


def score_exemplars(
    senf: SENF,
    registry: Mapping[str, Sequence[ExemplarDefinition]] = DEFAULT_EXEMPLAR_REGISTRY,
    alternative_margin: float = 0.1,
) -> SENF:
    """Attach deterministic exemplar distances to typed mention occurrences."""
    senf.exemplar_scores.clear()
    senf.nearest_exemplars.clear()
    senf.active_exemplars.clear()

    for mention in senf.mentions:
        kinds_and_definitions = [
            (kind, definition)
            for kind in _mention_kinds(senf, mention, registry)
            for definition in registry.get(canonical_symbol(kind), ())
        ]
        if not kinds_and_definitions:
            continue
        context = _mention_context(senf, mention)
        guard = _context_guard(senf, mention)
        scored: list[ExemplarScore] = []
        any_cue = False
        for kind, definition in kinds_and_definitions:
            matched = tuple(cue for cue in definition.cues if cue in context)
            any_cue = any_cue or bool(matched)
            if matched:
                distance = max(0.05, 0.18 - 0.03 * (len(matched) - 1))
            elif definition.name.startswith("generic_"):
                distance = 0.45
            else:
                distance = 0.65
            scored.append(
                ExemplarScore(
                    kind=kind,
                    exemplar=definition.name,
                    distance=round(distance, 4),
                    reasons=matched,
                    guard=guard,
                )
            )

        if any_cue:
            scored = [
                score
                if score.reasons or score.exemplar.startswith("generic_")
                else ExemplarScore(
                    score.kind, score.exemplar, 0.85, score.reasons, score.guard
                )
                for score in scored
            ]
        scored.sort(key=lambda score: (score.distance, score.exemplar))
        senf.exemplar_scores[mention.mention_id] = scored
        if scored:
            minimum = scored[0].distance
            active = [
                score
                for score in scored
                if score.distance <= minimum + alternative_margin
            ]
            senf.active_exemplars[mention.mention_id] = [score.exemplar for score in active]
            if len(active) == 1:
                senf.nearest_exemplars[mention.mention_id] = active[0].exemplar
    return senf


def exemplar_distance(
    left_senf: SENF,
    left: Mention,
    right_senf: SENF,
    right: Mention,
) -> float:
    left_exemplars = set(left_senf.active_exemplars_for(left))
    right_exemplars = set(right_senf.active_exemplars_for(right))
    if not left_exemplars or not right_exemplars:
        return 0.5
    left_kinds = set(_mention_kinds(left_senf, left))
    right_kinds = set(_mention_kinds(right_senf, right))
    if not left_kinds & right_kinds:
        return 1.0
    if left_exemplars & right_exemplars:
        return 0.0
    if any(value.startswith("generic_") for value in left_exemplars | right_exemplars):
        return 0.35
    return 0.8
