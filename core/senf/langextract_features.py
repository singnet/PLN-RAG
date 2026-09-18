"""LangExtract adapter for feature-only SENF evidence.

The adapter consumes aligned extraction objects and never translates them to
PLN or derives canonical symbols.  `extractor` injection keeps the optional
dependency out of imports and makes provider behavior independently testable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional

from core.senf.features import (
    CoreferenceEvidence,
    ExactSpan,
    FeatureBatch,
    FeatureRejection,
    FeatureValidation,
    SpanFeature,
    validate_feature_batch,
)

FEATURE_EXTRACTION_CLASS = "senf_feature"
COREFERENCE_EXTRACTION_CLASS = "senf_coreference"
EXACT_ALIGNMENT = "MATCH_EXACT"
_RESERVED_FIELDS = frozenset({
    "canonical_symbol", "canonicalsymbol", "symbol", "pln", "statement", "statements",
})


class LangExtractFeatureProvider:
    def __init__(
        self,
        extractor: Callable[[str], Iterable[Any]],
        *,
        allowed_features: Optional[Iterable[str]] = None,
    ):
        self._extractor = extractor
        self._allowed = frozenset(allowed_features) if allowed_features is not None else None

    def provide(self, source_text: str) -> FeatureBatch:
        return features_from_langextract(
            source_text,
            self._extractor(source_text),
            allowed_features=self._allowed,
        ).accepted


def features_from_langextract(
    source_text: str,
    extractions: Iterable[Any],
    *,
    allowed_features: Optional[Iterable[str]] = None,
) -> FeatureValidation:
    """Convert only exactly aligned, feature-only LangExtract output."""
    allowed = frozenset(allowed_features) if allowed_features is not None else None
    features: list[SpanFeature] = []
    grouped: dict[str, list[tuple[ExactSpan, float]]] = {}
    rejected: list[FeatureRejection] = []

    for index, extraction in enumerate(extractions):
        extraction_class = str(getattr(extraction, "extraction_class", "")).strip()
        if extraction_class not in (FEATURE_EXTRACTION_CLASS, COREFERENCE_EXTRACTION_CLASS):
            rejected.append(FeatureRejection("langextract", index, "unsupported extraction class"))
            continue
        alignment = getattr(extraction, "alignment_status", None)
        alignment_name = getattr(alignment, "name", None) or str(alignment).split(".")[-1]
        if alignment_name != EXACT_ALIGNMENT:
            rejected.append(FeatureRejection("langextract", index, "non-exact alignment"))
            continue
        try:
            span = _span_from_extraction(extraction)
            attrs = _attributes(extraction)
            normalized_keys = {
                str(key).strip().lower().replace("-", "_") for key in attrs
            }
            if _RESERVED_FIELDS & normalized_keys:
                raise ValueError("canonical symbol and PLN fields are forbidden")
            confidence = _confidence(attrs.get("confidence", 1.0))
            if extraction_class == FEATURE_EXTRACTION_CLASS:
                name = _required(attrs, "name")
                value = _required(attrs, "value")
                if name.lower().replace("-", "_") in _RESERVED_FIELDS:
                    raise ValueError("canonical symbol and PLN features are forbidden")
                if allowed is not None and name not in allowed:
                    raise ValueError("feature is not allowed")
                features.append(SpanFeature(span, name, value, confidence, ("langextract",)))
            else:
                group = _required(attrs, "group")
                grouped.setdefault(group, []).append((span, confidence))
        except (TypeError, ValueError) as exc:
            rejected.append(FeatureRejection("langextract", index, str(exc)))

    coreferences: list[CoreferenceEvidence] = []
    for group, members in grouped.items():
        if len(members) < 2:
            rejected.append(FeatureRejection("coreference_group", 0, f"group {group!r} has no pair"))
            continue
        antecedent, antecedent_confidence = members[0]
        for anaphor, confidence in members[1:]:
            coreferences.append(CoreferenceEvidence(
                anaphor,
                antecedent,
                min(antecedent_confidence, confidence),
                ("langextract", f"group:{group}"),
            ))

    validation = validate_feature_batch(
        source_text, FeatureBatch(tuple(features), tuple(coreferences))
    )
    return FeatureValidation(validation.accepted, tuple(rejected) + validation.rejected)


def _span_from_extraction(extraction: Any) -> ExactSpan:
    interval = getattr(extraction, "char_interval", None)
    start = getattr(interval, "start_pos", None)
    end = getattr(interval, "end_pos", None)
    text = getattr(extraction, "extraction_text", None)
    if type(start) is not int or type(end) is not int or not isinstance(text, str):
        raise ValueError("missing exact character interval")
    return ExactSpan(start, end, text)


def _attributes(extraction: Any) -> dict[str, Any]:
    attrs = getattr(extraction, "attributes", None)
    if not isinstance(attrs, dict):
        raise ValueError("attributes must be an object")
    return attrs


def _required(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {key}")
    return value.strip()


def _confidence(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("confidence must be numeric")
    return float(value)
