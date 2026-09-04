"""Optional document-level coreference resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ResolvedDocument:
    """Original and parser-facing text plus optional resolution diagnostics."""

    original: str
    resolved: str
    mentions: list[dict[str, Any]] = field(default_factory=list)


class CoreferenceResolver:
    """Resolve text with fastcoref when enabled, otherwise leave it unchanged."""

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "biu-nlp/f-coref",
        min_confidence: float = 0.65,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.min_confidence = min(1.0, max(0.0, min_confidence))
        self._model: Any | None = None
        self._load_failed = False

    def resolve(self, text: str) -> ResolvedDocument:
        """Return resolved text, failing open to the original input."""
        original = text or ""
        if not self.enabled or not original.strip():
            return ResolvedDocument(original=original, resolved=original)

        try:
            return self._resolve_with_fastcoref(original)
        except Exception as exc:  # Optional dependency/model must not break ingest.
            if not self._load_failed:
                logger.warning("Coreference resolver unavailable; using original text: %s", exc)
                self._load_failed = True
            return ResolvedDocument(original=original, resolved=original)

    def _resolve_with_fastcoref(self, text: str) -> ResolvedDocument:
        if self._model is None:
            from fastcoref import FCoref
            from fastcoref.coref_models.modeling_fcoref import FCorefModel

            # Patch for compatibility with newer versions of transformers
            FCorefModel.all_tied_weights_keys = {}

            self._model = FCoref(model_name_or_path=self.model_name)

        prediction = self._model.predict(texts=[text])[0]
        clusters = prediction.get_clusters(as_strings=False)
        replacements: list[tuple[int, int, str]] = []
        mentions: list[dict[str, Any]] = []

        for cluster_id, cluster in enumerate(clusters or []):
            if len(cluster) < 2:
                continue
            antecedent_start, antecedent_end = self._span(cluster[0])
            antecedent = text[antecedent_start:antecedent_end]
            if not antecedent.strip():
                continue
            for span in cluster[1:]:
                start, end = self._span(span)
                mention = text[start:end]
                if not mention.strip() or mention.casefold() == antecedent.casefold():
                    continue
                replacements.append((start, end, antecedent))
                mentions.append(
                    {
                        "cluster_id": cluster_id,
                        "mention": mention,
                        "antecedent": antecedent,
                        "start": start,
                        "end": end,
                        "confidence": 1.0,
                    }
                )

        resolved = self._replace_spans(text, replacements)
        return ResolvedDocument(original=text, resolved=resolved, mentions=mentions)

    @staticmethod
    def _span(span: Any) -> tuple[int, int]:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"Invalid coreference span: {span!r}")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid coreference span: {span!r}")
        return start, end

    @staticmethod
    def _replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text
