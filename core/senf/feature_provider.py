"""Optional SENF feature-provider boundary."""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from core.senf.features import FeatureBatch, FeatureValidation, validate_feature_batch

logger = logging.getLogger(__name__)


@runtime_checkable
class FeatureProvider(Protocol):
    """Produces advisory, source-grounded features without producing PLN."""

    def provide(self, source_text: str) -> FeatureBatch:
        ...


class NoOpFeatureProvider:
    def provide(self, source_text: str) -> FeatureBatch:
        return FeatureBatch.empty()


def provide_features(
    source_text: str,
    provider: Optional[FeatureProvider] = None,
) -> FeatureValidation:
    """Run an optional provider fail-open and validate every returned span."""
    if provider is None:
        return FeatureValidation(FeatureBatch.empty())
    try:
        return validate_feature_batch(source_text, provider.provide(source_text))
    except Exception as exc:
        logger.warning("SENF feature provider failed open: %s", exc)
        return FeatureValidation(FeatureBatch.empty())


class FailOpenFeatureProvider:
    """Composable provider wrapper for callers that only need accepted evidence."""

    def __init__(self, provider: Optional[FeatureProvider] = None):
        self._provider = provider

    def provide(self, source_text: str) -> FeatureBatch:
        return provide_features(source_text, self._provider).accepted
