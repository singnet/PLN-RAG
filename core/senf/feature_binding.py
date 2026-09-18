"""Bind provider evidence to existing mentions without mutating SENF state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from core.senf.features import (
    CoreferenceEvidence,
    FeatureBatch,
    FeatureRejection,
    SpanFeature,
    validate_feature_batch,
)


class MentionLike(Protocol):
    mention_id: str
    entity_id: str
    char_span: Optional[tuple[int, int]]
    surface: str


@dataclass(frozen=True)
class BoundFeature:
    mention_id: str
    entity_id: str
    feature: SpanFeature


@dataclass(frozen=True)
class BoundCoreferenceEvidence:
    anaphor_mention_id: str
    antecedent_mention_id: str
    evidence: CoreferenceEvidence


@dataclass(frozen=True)
class FeatureBindings:
    features: tuple[BoundFeature, ...] = ()
    coreferences: tuple[BoundCoreferenceEvidence, ...] = ()
    rejected: tuple[FeatureRejection, ...] = ()


def bind_features(
    source_text: str,
    mentions: Iterable[MentionLike],
    batch: FeatureBatch,
) -> FeatureBindings:
    """Bind only identical `(start, end, surface)` mentions and spans.

    Coreference output is deliberately a bound evidence edge, not an entity ID
    rewrite or a canonical-symbol election.
    """
    validation = validate_feature_batch(source_text, batch)
    by_span: dict[tuple[int, int, str], list[MentionLike]] = {}
    for mention in mentions:
        span = mention.char_span
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or span[0] < 0
            or span[1] <= span[0]
            or span[1] > len(source_text)
            or source_text[span[0]:span[1]] != mention.surface
        ):
            continue
        by_span.setdefault((span[0], span[1], mention.surface), []).append(mention)

    bound_features: list[BoundFeature] = []
    bound_coreferences: list[BoundCoreferenceEvidence] = []
    rejected = list(validation.rejected)
    for index, feature in enumerate(validation.accepted.features):
        matches = by_span.get((feature.span.start, feature.span.end, feature.span.text), ())
        if not matches:
            rejected.append(FeatureRejection("feature", index, "no exact mention span"))
            continue
        if len(matches) != 1:
            rejected.append(FeatureRejection("feature", index, "ambiguous exact mention span"))
            continue
        mention = matches[0]
        bound_features.append(BoundFeature(mention.mention_id, mention.entity_id, feature))

    for index, link in enumerate(validation.accepted.coreferences):
        anaphors = by_span.get((link.anaphor.start, link.anaphor.end, link.anaphor.text), ())
        antecedents = by_span.get(
            (link.antecedent.start, link.antecedent.end, link.antecedent.text), ()
        )
        if not anaphors or not antecedents:
            rejected.append(FeatureRejection("coreference", index, "no exact mention span"))
            continue
        if len(anaphors) != 1 or len(antecedents) != 1:
            rejected.append(FeatureRejection("coreference", index, "ambiguous exact mention span"))
            continue
        bound_coreferences.append(BoundCoreferenceEvidence(
            anaphors[0].mention_id, antecedents[0].mention_id, link
        ))
    return FeatureBindings(
        tuple(bound_features), tuple(bound_coreferences), tuple(rejected)
    )
