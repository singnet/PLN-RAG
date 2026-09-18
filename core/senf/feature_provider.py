"""Optional SENF feature-provider boundary."""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from core.senf.features import (
    FeatureBatch,
    FeatureRejection,
    FeatureValidation,
    validate_feature_batch,
)

logger = logging.getLogger(__name__)

_REJECTION_CATEGORIES = frozenset({
    "provider", "feature", "coreference", "langextract", "coreference_group",
})
_REJECTION_REASONS = frozenset({
    "feature provider failed",
    "feature limit exceeded",
    "unsupported extraction class",
    "non-exact alignment",
    "missing exact character interval",
    "attributes must be an object",
    "canonical symbol and PLN fields are forbidden",
    "canonical symbol and PLN features are forbidden",
    "missing name",
    "missing value",
    "missing group",
    "confidence must be numeric",
    "feature is not allowed",
    "invalid coreference polarity",
    "group has no pair",
    "span bounds must be integers",
    "span must be non-empty and ordered",
    "span text must be non-empty",
    "span text length must equal its bounds",
    "feature name must be non-empty",
    "feature value must be non-empty",
    "feature confidence must be finite",
    "feature confidence must be between zero and one",
    "feature evidence must contain non-empty strings",
    "coreference endpoints must be distinct",
    "coreference polarity must be positive or negative",
    "coreference confidence must be finite",
    "coreference confidence must be between zero and one",
    "coreference evidence must contain non-empty strings",
    "malformed feature",
    "malformed coreference",
    "span does not exactly match source",
})
_REDACTED_REJECTION = "provider rejection details redacted"


@runtime_checkable
class FeatureProvider(Protocol):
    """Produces advisory, source-grounded features without producing PLN."""

    def provide(self, source_text: str) -> FeatureBatch | FeatureValidation:
        ...


class NoOpFeatureProvider:
    def provide(self, source_text: str) -> FeatureBatch:
        return FeatureBatch.empty()


def provide_features(
    source_text: str,
    provider: Optional[FeatureProvider] = None,
    *,
    max_rejections: int = 64,
) -> FeatureValidation:
    """Run an optional provider fail-open and revalidate every accepted span."""
    if provider is None:
        return FeatureValidation(FeatureBatch.empty())
    try:
        provided = provider.provide(source_text)
        if isinstance(provided, FeatureValidation):
            validation = validate_feature_batch(source_text, provided.accepted)
            provider_rejected = _safe_provider_rejections(
                provided.rejected, max_rejections
            )
            remaining = max(0, max_rejections - len(provider_rejected))
            rejected = provider_rejected + validation.rejected[:remaining]
            return FeatureValidation(
                validation.accepted, rejected
            )
        return validate_feature_batch(source_text, provided)
    except Exception as exc:
        if getattr(exc, "fail_closed", False):
            raise
        logger.warning(
            "SENF feature provider failed open (%s)", type(exc).__name__
        )

        return FeatureValidation(FeatureBatch.empty(), (
            FeatureRejection("provider", 0, "feature provider failed"),
        ))


def _safe_provider_rejections(
    rejected: object, max_rejections: int
) -> tuple[FeatureRejection, ...]:
    if not isinstance(rejected, tuple) or max_rejections <= 0:
        return ()
    safe: list[FeatureRejection] = []
    for item in rejected:
        if len(safe) >= max_rejections:
            break
        if (
            not isinstance(item, FeatureRejection)
            or not isinstance(item.category, str)
            or not item.category.strip()
            or type(item.index) is not int
            or item.index < 0
            or not isinstance(item.reason, str)
        ):
            continue
        category = item.category.strip().lower().replace("-", "_")
        if category not in _REJECTION_CATEGORIES:
            category = "provider"
        reason = (
            item.reason if item.reason in _REJECTION_REASONS else _REDACTED_REJECTION
        )
        safe.append(FeatureRejection(category, item.index, reason))
    return tuple(safe)


class FailOpenFeatureProvider:
    """Composable provider wrapper that retains bounded provider diagnostics."""

    def __init__(self, provider: Optional[FeatureProvider] = None):
        self._provider = provider

    def provide(self, source_text: str) -> FeatureValidation:
        return provide_features(source_text, self._provider)


def create_feature_provider(settings: Any) -> FeatureProvider:
    """Build the configured provider without importing optional dependencies."""
    name = settings.senf_feature_provider
    if name == "none":
        return NoOpFeatureProvider()
    if name == "langextract":
        from core.senf.langextract_features import LangExtractFeatureProvider

        return LangExtractFeatureProvider.from_settings(settings)
    raise ValueError(f"Unsupported SENF feature provider: {name}")
