"""Optional document-level coreference resolution."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


logger = logging.getLogger(__name__)

ResolverState = Literal["disabled", "not_loaded", "ready", "degraded"]
ResolutionState = Literal["disabled", "unchanged", "resolved", "failed_open"]


@dataclass(frozen=True)
class CoreferenceMention:
    start: int
    end: int
    text: str
    score: float | None = None


@dataclass(frozen=True)
class CoreferenceCluster:
    cluster_id: int
    mentions: tuple[CoreferenceMention, ...]


@dataclass(frozen=True)
class CoreferencePrediction:
    backend: str
    model: str | None
    clusters: tuple[CoreferenceCluster, ...]
    score_available: bool = False


@dataclass(frozen=True)
class CoreferenceReplacement:
    cluster_id: int
    start: int
    end: int
    mention: str
    antecedent: str
    replacement: str
    score: float | None = None


@dataclass(frozen=True)
class ResolvedDocument:
    """Original and parser-facing text plus resolution diagnostics."""

    original: str
    resolved: str
    backend: str = "none"
    model: str | None = None
    status: ResolutionState = "unchanged"
    replacements: tuple[CoreferenceReplacement, ...] = ()
    duration_seconds: float = 0.0
    score_available: bool = False
    error: str | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def mentions(self) -> list[dict[str, Any]]:
        """Compatibility view for the original experimental API."""
        return [asdict(replacement) for replacement in self.replacements]


@dataclass(frozen=True)
class CoreferenceStatus:
    enabled: bool
    fail_open: bool
    backend: str
    model: str | None
    state: ResolverState
    score_available: bool
    documents_processed: int
    documents_changed: int
    replacements: int
    failures: int
    total_duration_seconds: float
    last_duration_seconds: float
    last_error: str | None


class CoreferenceBackend(Protocol):
    name: str
    model_name: str | None

    @property
    def loaded(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    def predict(self, text: str) -> CoreferencePrediction: ...


class NoneCoreferenceBackend:
    name = "none"
    model_name = None
    loaded = True
    load_error = None

    def predict(self, text: str) -> CoreferencePrediction:
        return CoreferencePrediction(backend=self.name, model=None, clusters=())


class FastCorefBackend:
    """Thread-safe adapter for the FCoref and LingMess architectures."""

    def __init__(
        self,
        architecture: Literal["fcoref", "lingmess"] = "fcoref",
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        if architecture not in {"fcoref", "lingmess"}:
            raise ValueError(f"Unsupported coreference backend: {architecture}")
        if device != "cpu":
            raise ValueError("Only the CPU coreference runtime is currently supported")
        self.name = architecture
        self.model_name = model_name or (
            "biu-nlp/f-coref" if architecture == "fcoref" else "biu-nlp/lingmess-coref"
        )
        self.device = device
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import spacy
            from fastcoref import FCoref, LingMessCoref

            model_class = FCoref if self.name == "fcoref" else LingMessCoref
            if self.name == "fcoref":
                from fastcoref.coref_models.modeling_fcoref import FCorefModel

                # fastcoref 2.1.6 predates the tied-weight API in Transformers 5.
                FCorefModel.all_tied_weights_keys = {}
            else:
                from fastcoref.coref_models.modeling_lingmess import LingMessModel

                # Transformers 5 selects SDPA for Longformer even though that
                # architecture still requires eager attention.
                LingMessModel.all_tied_weights_keys = {}
                if not getattr(LingMessModel, "_pln_rag_eager_attention", False):
                    original_init = LingMessModel.__init__

                    def eager_init(instance: Any, config: Any) -> None:
                        config._attn_implementation = "eager"
                        original_init(instance, config)

                    LingMessModel.__init__ = eager_init
                    LingMessModel._pln_rag_eager_attention = True

            self._model = model_class(
                model_name_or_path=self.model_name,
                device=self.device,
                nlp=spacy.blank("en"),
                enable_progress_bar=False,
            )
            self._load_error = None
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def predict(self, text: str) -> CoreferencePrediction:
        with self._lock:
            model = self._load()
            prediction = model.predict(texts=[text])[0]
            raw_clusters = prediction.get_clusters(as_strings=False) or []
            clusters: list[CoreferenceCluster] = []
            score_available = False

            for cluster_id, raw_cluster in enumerate(raw_clusters):
                spans = [self._normalize_span(span) for span in raw_cluster]
                mentions: list[CoreferenceMention] = []
                score_anchor = spans[0] if spans else None
                for span in spans:
                    start, end = span
                    score = None
                    if score_anchor is not None and span != score_anchor:
                        try:
                            score = float(prediction.get_logit(score_anchor, span))
                            score_available = True
                        except (AttributeError, TypeError, ValueError, KeyError):
                            pass
                    mentions.append(
                        CoreferenceMention(start=start, end=end, text=text[start:end], score=score)
                    )
                clusters.append(CoreferenceCluster(cluster_id, tuple(mentions)))

            return CoreferencePrediction(
                backend=self.name,
                model=self.model_name,
                clusters=tuple(clusters),
                score_available=score_available,
            )

    @staticmethod
    def _normalize_span(span: Any) -> tuple[int, int]:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"Invalid coreference span: {span!r}")
        return int(span[0]), int(span[1])


class CoreferenceResolver:
    """Resolve pronouns conservatively using a configurable model backend."""

    _plain_pronouns = {"he", "him", "she", "it", "they", "them"}
    _possessive_pronouns = {"his", "hers", "its", "their", "theirs"}
    _all_pronouns = _plain_pronouns | _possessive_pronouns | {"her"}
    _possessive_attachment_words = {"at", "from", "in", "of", "on", "with"}
    _possessive_terminal_noise = {
        "also",
        "closely",
        "gradually",
        "however",
        "respectively",
        "thus",
    }

    def __init__(
        self,
        enabled: bool = False,
        model_name: str | None = None,
        min_confidence: float = 0.65,
        *,
        backend_name: Literal["none", "fcoref", "lingmess"] = "fcoref",
        device: str = "cpu",
        fail_open: bool = True,
        backend: CoreferenceBackend | None = None,
    ) -> None:
        self.enabled = enabled and backend_name != "none"
        self.min_confidence = min(1.0, max(0.0, min_confidence))
        self.fail_open = fail_open
        self._backend: CoreferenceBackend = backend or (
            FastCorefBackend(backend_name, model_name, device)
            if self.enabled
            else NoneCoreferenceBackend()
        )
        self._status_lock = threading.Lock()
        self._documents_processed = 0
        self._documents_changed = 0
        self._replacement_count = 0
        self._failures = 0
        self._total_duration = 0.0
        self._last_duration = 0.0
        self._last_error: str | None = None
        self._score_available = False

    def resolve(self, text: str) -> ResolvedDocument:
        """Return resolved text, optionally failing open to the original input."""
        original = text or ""
        if not self.enabled or not original.strip():
            return ResolvedDocument(
                original=original,
                resolved=original,
                backend=self._backend.name,
                model=self._backend.model_name,
                status="disabled" if not self.enabled else "unchanged",
            )

        started = time.perf_counter()
        try:
            prediction = self._backend.predict(original)
            resolved, replacements, diagnostics = self._rewrite(original, prediction.clusters)
            duration = time.perf_counter() - started
            result = ResolvedDocument(
                original=original,
                resolved=resolved,
                backend=prediction.backend,
                model=prediction.model,
                status="resolved" if resolved != original else "unchanged",
                replacements=replacements,
                duration_seconds=duration,
                score_available=prediction.score_available,
                diagnostics=diagnostics,
            )
            self._record(result)
            return result
        except Exception as exc:
            duration = time.perf_counter() - started
            error = self._sanitize_error(exc)
            if not self.fail_open:
                self._record_failure(duration, error)
                raise
            logger.warning("Coreference resolution failed; using original text: %s", error)
            result = ResolvedDocument(
                original=original,
                resolved=original,
                backend=self._backend.name,
                model=self._backend.model_name,
                status="failed_open",
                duration_seconds=duration,
                error=error,
            )
            self._record(result)
            return result

    def _record_failure(self, duration: float, error: str) -> None:
        with self._status_lock:
            self._documents_processed += 1
            self._failures += 1
            self._total_duration += duration
            self._last_duration = duration
            self._last_error = error

    def _record(self, result: ResolvedDocument) -> None:
        with self._status_lock:
            self._documents_processed += 1
            self._documents_changed += int(result.resolved != result.original)
            self._replacement_count += len(result.replacements)
            self._failures += int(result.status == "failed_open")
            self._total_duration += result.duration_seconds
            self._last_duration = result.duration_seconds
            self._last_error = result.error
            self._score_available = self._score_available or result.score_available

    def status(self) -> CoreferenceStatus:
        with self._status_lock:
            if not self.enabled:
                state: ResolverState = "disabled"
            elif self._backend.load_error or self._last_error:
                state = "degraded"
            elif self._backend.loaded:
                state = "ready"
            else:
                state = "not_loaded"
            return CoreferenceStatus(
                enabled=self.enabled,
                fail_open=self.fail_open,
                backend=self._backend.name,
                model=self._backend.model_name,
                state=state,
                score_available=self._score_available,
                documents_processed=self._documents_processed,
                documents_changed=self._documents_changed,
                replacements=self._replacement_count,
                failures=self._failures,
                total_duration_seconds=round(self._total_duration, 6),
                last_duration_seconds=round(self._last_duration, 6),
                last_error=self._last_error,
            )

    def _rewrite(
        self,
        text: str,
        clusters: tuple[CoreferenceCluster, ...],
    ) -> tuple[str, tuple[CoreferenceReplacement, ...], tuple[str, ...]]:
        candidates: list[CoreferenceReplacement] = []
        diagnostics: list[str] = []

        for cluster in clusters:
            valid_mentions = []
            for mention in sorted(cluster.mentions, key=lambda item: (item.start, item.end)):
                error = self._validate_mention(text, mention)
                if error:
                    diagnostics.append(f"cluster_{cluster.cluster_id}:{error}")
                else:
                    valid_mentions.append(mention)

            antecedent = next(
                (
                    mention
                    for mention in valid_mentions
                    if mention.text.strip().casefold() not in self._all_pronouns
                ),
                None,
            )
            if antecedent is None:
                continue
            antecedent_text = antecedent.text.strip().rstrip(",;:")
            if not antecedent_text:
                diagnostics.append(f"cluster_{cluster.cluster_id}:invalid_antecedent")
                continue

            for mention in valid_mentions:
                pronoun = mention.text.strip().casefold()
                if mention.start < antecedent.end or pronoun == "her":
                    continue
                if pronoun in self._plain_pronouns:
                    replacement = antecedent_text
                elif pronoun in self._possessive_pronouns:
                    unsafe_reason = self._unsafe_possessive_antecedent(antecedent_text)
                    if unsafe_reason:
                        diagnostics.append(
                            f"cluster_{cluster.cluster_id}:unsafe_possessive_antecedent:"
                            f"{unsafe_reason}:{mention.start}:{mention.end}"
                        )
                        continue
                    replacement = self._possessive(antecedent_text)
                else:
                    continue
                candidates.append(
                    CoreferenceReplacement(
                        cluster_id=cluster.cluster_id,
                        start=mention.start,
                        end=mention.end,
                        mention=mention.text,
                        antecedent=antecedent_text,
                        replacement=replacement,
                        score=mention.score,
                    )
                )

        accepted: list[CoreferenceReplacement] = []
        seen_spans: set[tuple[int, int]] = set()
        for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
            span = (candidate.start, candidate.end)
            if span in seen_spans:
                diagnostics.append(f"duplicate_span:{candidate.start}:{candidate.end}")
                continue
            if accepted and candidate.start < accepted[-1].end:
                diagnostics.append(f"overlapping_span:{candidate.start}:{candidate.end}")
                continue
            seen_spans.add(span)
            accepted.append(candidate)

        resolved = text
        for replacement in reversed(accepted):
            resolved = (
                resolved[: replacement.start]
                + replacement.replacement
                + resolved[replacement.end :]
            )
        return resolved, tuple(accepted), tuple(diagnostics)

    @staticmethod
    def _validate_mention(text: str, mention: CoreferenceMention) -> str | None:
        if not isinstance(mention.start, int) or not isinstance(mention.end, int):
            return "non_integer_span"
        if mention.start < 0 or mention.end <= mention.start or mention.end > len(text):
            return f"invalid_span:{mention.start}:{mention.end}"
        if text[mention.start : mention.end] != mention.text:
            return f"text_mismatch:{mention.start}:{mention.end}"
        return None

    @staticmethod
    def _possessive(antecedent: str) -> str:
        if antecedent.endswith(("'s", "'")):
            return antecedent
        return antecedent + ("'" if antecedent.casefold().endswith("s") else "'s")

    @classmethod
    def _unsafe_possessive_antecedent(cls, antecedent: str) -> str | None:
        """Reject structurally risky noun phrases without imposing a length cap."""
        if re.search(r"[,;:()[\]{}]", antecedent):
            return "internal_punctuation"

        normalized = " ".join(antecedent.casefold().split())
        tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", normalized)
        if not tokens:
            return "empty"
        if normalized.startswith(("a total of ", "an aggregate of ", "a group of ")):
            return "quantified_phrase"
        if "other" in tokens:
            return "comparative_group"
        if any(token in {"that", "which", "who", "whom", "whose"} for token in tokens):
            return "relative_clause"
        if sum(token in cls._possessive_attachment_words for token in tokens) > 1:
            return "multiple_attachments"
        if tokens[-1] in cls._possessive_terminal_noise:
            return "terminal_adverb"
        return None

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        return " ".join(str(exc).split())[:500] or exc.__class__.__name__
