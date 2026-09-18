"""Bind provider evidence to existing mentions without mutating SENF state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Protocol

from core.senf.features import (
    CoreferenceEvidence,
    FeatureBatch,
    FeatureRejection,
    SpanFeature,
    validate_feature_batch,
)
from core.senf.types import (
    AppliedCoreferenceEvidence,
    AppliedMentionFeature,
    Constraint,
    EntityRef,
    SENF,
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


_MENTION_TYPES = frozenset({"proper", "common", "pronoun", "nominal"})
_DEFINITENESS = frozenset({"definite", "indefinite", "demonstrative", "pronoun", "unknown"})
_CONTEXT_HINTS = frozenset({"modality", "time_ref", "location_ref"})


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


def apply_feature_bindings(
    senf: SENF,
    bindings: FeatureBindings,
    *,
    provider: str,
) -> tuple[FeatureRejection, ...]:
    """Apply only allowlisted advisory fields and retain an immutable audit trail."""
    rejected = list(bindings.rejected)
    mention_index = {mention.mention_id: index for index, mention in enumerate(senf.mentions)}
    seen = {
        (item.mention_id, item.name, item.value, item.provider)
        for item in senf.applied_mention_features
    }
    for index, bound in enumerate(bindings.features):
        feature = bound.feature
        if bound.mention_id not in mention_index:
            rejected.append(FeatureRejection("feature", index, "bound mention is missing"))
            continue
        mention = senf.mentions[mention_index[bound.mention_id]]
        if bound.entity_id != mention.entity_id:
            rejected.append(FeatureRejection("feature", index, "bound entity is not mention owner"))
            continue
        if feature.name == "mention_type" and feature.value not in _MENTION_TYPES:
            rejected.append(FeatureRejection("feature", index, "invalid mention type"))
            continue
        if feature.name == "definiteness" and feature.value not in _DEFINITENESS:
            rejected.append(FeatureRejection("feature", index, "invalid definiteness"))
            continue
        if feature.name == "exemplar_cue" and feature.value.lower() not in _registered_exemplar_cues(
            senf, mention
        ):
            rejected.append(FeatureRejection("feature", index, "unregistered exemplar cue"))
            continue
        if (
            feature.name == "exemplar_cue"
            and feature.value.lower() not in feature.span.text.lower()
        ):
            rejected.append(FeatureRejection(
                "feature", index, "exemplar cue is not present in its exact source span"
            ))
            continue
        if feature.name not in {"mention_type", "definiteness", "exemplar_cue"} | _CONTEXT_HINTS:
            rejected.append(FeatureRejection("feature", index, "feature cannot augment SENF"))
            continue

        key = (bound.mention_id, feature.name, feature.value, provider)
        if key in seen:
            continue
        seen.add(key)
        if feature.confidence > 0.0 and feature.name in ("mention_type", "definiteness"):
            senf.mentions[mention_index[bound.mention_id]] = replace(
                mention, **{feature.name: feature.value}
            )
        elif feature.confidence > 0.0 and feature.name in _CONTEXT_HINTS:
            frames = [
                frame for frame in senf.frames
                if any(
                    isinstance(role.filler, EntityRef)
                    and role.filler.mention_id == bound.mention_id
                    for role in frame.roles
                )
            ]
            if any(
                getattr(frame, feature.name) not in (None, feature.value)
                for frame in frames
            ):
                rejected.append(FeatureRejection(
                    "feature", index, "context hint conflicts with parser-owned context"
                ))
                continue
            for frame in frames:
                if getattr(frame, feature.name) == feature.value:
                    continue
                setattr(frame, feature.name, feature.value)
                if frame.context is not None:
                    frame.context = replace(frame.context, **{feature.name: feature.value})
                    constraint = Constraint(
                        feature.name, feature.value, frame.frame_id,
                        frame.context.source_unit_id,
                    )
                    if constraint not in senf.constraints:
                        senf.constraints.append(constraint)
        senf.applied_mention_features.append(AppliedMentionFeature(
            bound.mention_id,
            bound.entity_id,
            feature.name,
            feature.value,
            feature.confidence,
            provider,
            (feature.span.start, feature.span.end),
            feature.span.text,
            feature.evidence,
        ))

    seen_links = {
        (item.anaphor_mention_id, item.antecedent_mention_id, item.polarity, item.provider)
        for item in senf.coreference_evidence
    }
    for bound in bindings.coreferences:
        link = bound.evidence
        key = (
            bound.anaphor_mention_id, bound.antecedent_mention_id,
            link.polarity, provider,
        )
        if key in seen_links:
            continue
        seen_links.add(key)
        senf.coreference_evidence.append(AppliedCoreferenceEvidence(
            bound.anaphor_mention_id,
            bound.antecedent_mention_id,
            link.polarity,
            link.confidence,
            provider,
            link.evidence,
        ))
    return tuple(rejected)


def _registered_exemplar_cues(senf: SENF, mention: MentionLike) -> frozenset[str]:
    from core.senf.exemplars import DEFAULT_EXEMPLAR_REGISTRY
    from core.symbol_normalization import canonical_symbol

    kinds = {
        canonical_symbol(assertion.kind.canonical_symbol)
        for assertion in senf.kind_assertions
        if assertion.entity_id == mention.entity_id and assertion.polarity
    }
    kinds.update(
        normalized
        for value in (mention.canonical_symbol, mention.head_lemma)
        if (normalized := canonical_symbol(value)) in DEFAULT_EXEMPLAR_REGISTRY
    )
    return frozenset(
        cue.lower()
        for kind in kinds
        for definition in DEFAULT_EXEMPLAR_REGISTRY.get(kind, ())
        for cue in definition.cues
    )
