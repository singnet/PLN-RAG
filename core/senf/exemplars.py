from dataclasses import dataclass
from typing import Mapping, Sequence

from core.senf.types import ExemplarScore, Mention, SENF
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
        ExemplarDefinition("standalone_lens", ("borrowed", "borrow", "separate lens")),
    ),
}


def _mention_context(senf: SENF, mention: Mention, radius: int = 120) -> str:
    text = next((frame.source_text for frame in senf.frames if frame.source_text), "")
    if not text:
        return mention.surface.lower()
    if mention.char_span is None:
        return text.lower()
    start, end = mention.char_span
    return text[max(0, start - radius) : min(len(text), end + radius)].lower()


def score_exemplars(
    senf: SENF,
    registry: Mapping[str, Sequence[ExemplarDefinition]] = DEFAULT_EXEMPLAR_REGISTRY,
    alternative_margin: float = 0.1,
) -> SENF:
    """Attach deterministic exemplar distances to typed mention occurrences."""
    senf.exemplar_scores.clear()
    senf.nearest_exemplars.clear()

    for mention in senf.mentions:
        kind = senf.kind_for(mention)
        definitions = registry.get(canonical_symbol(kind or ""), ())
        if not definitions:
            continue
        context = _mention_context(senf, mention)
        scored: list[ExemplarScore] = []
        any_cue = False
        for definition in definitions:
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
                    kind=canonical_symbol(kind or ""),
                    exemplar=definition.name,
                    distance=round(distance, 4),
                    reasons=matched,
                )
            )

        if any_cue:
            scored = [
                score
                if score.reasons or score.exemplar.startswith("generic_")
                else ExemplarScore(
                    score.kind, score.exemplar, 0.85, score.reasons
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
            senf.nearest_exemplars[mention.mention_id] = active[0].exemplar
    return senf


def exemplar_distance(
    left_senf: SENF,
    left: Mention,
    right_senf: SENF,
    right: Mention,
) -> float:
    left_exemplar = left_senf.nearest_exemplar_for(left)
    right_exemplar = right_senf.nearest_exemplar_for(right)
    if not left_exemplar or not right_exemplar:
        return 0.5
    if left_exemplar == right_exemplar:
        return 0.0
    left_kind = left_senf.kind_for(left)
    right_kind = right_senf.kind_for(right)
    if left_kind and left_kind == right_kind:
        if left_exemplar.startswith("generic_") or right_exemplar.startswith("generic_"):
            return 0.35
        return 0.8
    return 1.0
