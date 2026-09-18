"""Provider-neutral, source-grounded evidence for enriching SENF mentions.

These values intentionally have no canonical-symbol or PLN representation.  A
feature provider describes source text; the canonical parser remains the sole
owner of symbols and logical statements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class ExactSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("span bounds must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must be non-empty and ordered")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("span text must be non-empty")
        if len(self.text) != self.end - self.start:
            raise ValueError("span text length must equal its bounds")

    def matches(self, source_text: str) -> bool:
        return self.end <= len(source_text) and source_text[self.start:self.end] == self.text


@dataclass(frozen=True)
class SpanFeature:
    """One advisory feature attached to an exact source span."""

    span: ExactSpan
    name: str
    value: str
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("feature name must be non-empty")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("feature value must be non-empty")
        if type(self.confidence) not in (int, float) or not math.isfinite(self.confidence):
            raise ValueError("feature confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("feature confidence must be between zero and one")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, str) or not item for item in self.evidence
        ):
            raise ValueError("feature evidence must contain non-empty strings")


@dataclass(frozen=True)
class CoreferenceEvidence:
    """Advisory evidence only; this value never requests or performs a merge."""

    anaphor: ExactSpan
    antecedent: ExactSpan
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()
    polarity: str = "positive"

    def __post_init__(self) -> None:
        if self.anaphor == self.antecedent:
            raise ValueError("coreference endpoints must be distinct")
        if self.polarity not in ("positive", "negative"):
            raise ValueError("coreference polarity must be positive or negative")
        if type(self.confidence) not in (int, float) or not math.isfinite(self.confidence):
            raise ValueError("coreference confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("coreference confidence must be between zero and one")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, str) or not item for item in self.evidence
        ):
            raise ValueError("coreference evidence must contain non-empty strings")


@dataclass(frozen=True)
class FeatureBatch:
    features: tuple[SpanFeature, ...] = ()
    coreferences: tuple[CoreferenceEvidence, ...] = ()

    @classmethod
    def empty(cls) -> FeatureBatch:
        return cls()


@dataclass(frozen=True)
class FeatureRejection:
    category: str
    index: int
    reason: str


@dataclass(frozen=True)
class FeatureValidation:
    accepted: FeatureBatch
    rejected: tuple[FeatureRejection, ...] = ()


def validate_feature_batch(source_text: str, batch: FeatureBatch) -> FeatureValidation:
    """Keep only evidence whose stated text exactly equals its source slice."""
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string")
    if not isinstance(batch, FeatureBatch):
        raise TypeError("provider must return FeatureBatch")

    features: list[SpanFeature] = []
    coreferences: list[CoreferenceEvidence] = []
    rejected: list[FeatureRejection] = []
    for index, feature in enumerate(batch.features):
        if not isinstance(feature, SpanFeature):
            rejected.append(FeatureRejection("feature", index, "malformed feature"))
        elif not feature.span.matches(source_text):
            rejected.append(FeatureRejection("feature", index, "span does not exactly match source"))
        else:
            features.append(feature)
    for index, link in enumerate(batch.coreferences):
        if not isinstance(link, CoreferenceEvidence):
            rejected.append(FeatureRejection("coreference", index, "malformed coreference"))
        elif not link.anaphor.matches(source_text) or not link.antecedent.matches(source_text):
            rejected.append(FeatureRejection("coreference", index, "span does not exactly match source"))
        else:
            coreferences.append(link)
    return FeatureValidation(FeatureBatch(tuple(features), tuple(coreferences)), tuple(rejected))


def feature_batch(
    features: Iterable[SpanFeature] = (),
    coreferences: Iterable[CoreferenceEvidence] = (),
) -> FeatureBatch:
    return FeatureBatch(tuple(features), tuple(coreferences))
