"""LangExtract adapter for feature-only SENF evidence.

The adapter consumes aligned extraction objects and never translates them to
PLN or derives canonical symbols.  `extractor` injection keeps the optional
dependency out of imports and makes provider behavior independently testable.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
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
        extractor: Optional[Callable[[str], Iterable[Any]]] = None,
        *,
        allowed_features: Optional[Iterable[str]] = None,
        exact_only: bool = True,
        max_features: int = 64,
        timeout: float = 20.0,
        model_id: Optional[str] = None,
        model_url: Optional[str] = None,
        api_key: Optional[str] = None,
        examples_path: Optional[str] = None,
    ):
        self._extractor = extractor
        self._allowed = frozenset(allowed_features) if allowed_features is not None else None
        self._exact_only = exact_only
        self._max_features = max_features
        self._timeout = timeout
        self._model_id = model_id
        self._model_url = model_url
        self._api_key = api_key
        self._examples_path = examples_path
        self._backend = None

    @classmethod
    def from_settings(cls, settings: Any) -> "LangExtractFeatureProvider":
        return cls(
            exact_only=settings.senf_feature_exact_only,
            max_features=settings.senf_feature_max_features,
            timeout=settings.senf_feature_timeout,
            model_id=settings.senf_feature_model or settings.langextract_model_id,
            model_url=settings.langextract_model_url,
            api_key=settings.langextract_api_key or settings.openai_api_key,
            examples_path=settings.senf_feature_examples_path,
            allowed_features=(
                "mention_type", "definiteness", "exemplar_cue", "modality",
                "time_ref", "location_ref",
            ),
        )

    @property
    def name(self) -> str:
        return "langextract"

    def set_backend(self, backend: Any) -> None:
        self._backend = backend

    def provide(self, source_text: str) -> FeatureBatch:
        examples_path = self._resolved_examples_path()
        request = {"text": source_text, "prompt": "senf-feature-only-v1"}
        config = {
            "exact_only": self._exact_only,
            "max_features": self._max_features,
            "timeout": self._timeout,
            "model_id": self._model_id,
            "model_url": self._model_url,
            "examples_path": str(examples_path),
            "examples_sha256": (
                hashlib.sha256(examples_path.read_bytes()).hexdigest()
                if examples_path.is_file() else None
            ),
        }
        live = lambda: self._extract(source_text)
        extractions = (
            self._backend.feature_call(request=request, config=config, live=live)
            if self._backend is not None and hasattr(self._backend, "feature_call")
            else live()
        )
        validation = features_from_langextract(
            source_text, list(extractions)[: self._max_features],
            allowed_features=self._allowed,
            exact_only=self._exact_only,
        )
        return validation.accepted

    def _extract(self, source_text: str) -> list[Any]:
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_extract_worker,
            args=(self, source_text, sender),
            name="senf-features",
            daemon=True,
        )
        process.start()
        sender.close()
        try:
            if not receiver.poll(self._timeout):
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                raise TimeoutError("LangExtract SENF feature call timed out")
            status, payload = receiver.recv()
            process.join()
            if status != "ok":
                raise RuntimeError(payload)
            return [_extraction_from_wire(item) for item in payload]
        finally:
            receiver.close()
            if process.is_alive():
                process.terminate()
                process.join()

    def _extract_live(self, source_text: str) -> list[Any]:
        if self._extractor is not None:
            return list(self._extractor(source_text))
        import langextract as lx

        path = self._resolved_examples_path()
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = [
            lx.data.ExampleData(
                text=str(example["text"]),
                extractions=[
                    lx.data.Extraction(
                        str(item["class"]), str(item["text"]),
                        attributes=dict(item.get("attributes", {})),
                    )
                    for item in example.get("extractions", ())
                ],
            )
            for example in payload.get("examples", ())
        ]
        result = lx.extract(
            text_or_documents=source_text,
            prompt_description=str(payload["prompt"]),
            examples=examples,
            model_id=self._model_id,
            model_url=self._model_url,
            api_key=self._api_key,
            extraction_passes=1,
            max_char_buffer=max(1000, len(source_text) + 1),
            max_workers=1,
            show_progress=False,
        )
        return list(getattr(result, "extractions", ()))

    def _resolved_examples_path(self) -> Path:
        path = Path(self._examples_path or "data/senf_feature_examples.json")
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def features_from_langextract(
    source_text: str,
    extractions: Iterable[Any],
    *,
    allowed_features: Optional[Iterable[str]] = None,
    exact_only: bool = True,
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
        if exact_only and alignment_name != EXACT_ALIGNMENT:
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
                polarity = str(attrs.get("polarity", "positive")).strip().lower()
                if polarity not in ("positive", "negative"):
                    raise ValueError("invalid coreference polarity")
                grouped.setdefault(f"{polarity}:{group}", []).append((span, confidence))
        except (TypeError, ValueError) as exc:
            rejected.append(FeatureRejection("langextract", index, str(exc)))

    coreferences: list[CoreferenceEvidence] = []
    for grouped_key, members in grouped.items():
        polarity, group = grouped_key.split(":", 1)
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
                polarity,
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


def _extract_worker(provider: LangExtractFeatureProvider, source_text: str, sender: Any) -> None:
    try:
        sender.send((
            "ok",
            [_extraction_to_wire(item) for item in provider._extract_live(source_text)],
        ))
    except BaseException as exc:
        sender.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def _extraction_to_wire(extraction: Any) -> dict[str, Any]:
    interval = getattr(extraction, "char_interval", None)
    alignment = getattr(extraction, "alignment_status", None)
    return {
        "extraction_class": getattr(extraction, "extraction_class", None),
        "extraction_text": getattr(extraction, "extraction_text", None),
        "attributes": getattr(extraction, "attributes", None),
        "start": getattr(interval, "start_pos", None),
        "end": getattr(interval, "end_pos", None),
        "alignment": getattr(alignment, "name", None) or str(alignment).split(".")[-1],
    }


def _extraction_from_wire(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        extraction_class=payload.get("extraction_class"),
        extraction_text=payload.get("extraction_text"),
        attributes=payload.get("attributes"),
        char_interval=SimpleNamespace(
            start_pos=payload.get("start"), end_pos=payload.get("end")
        ),
        alignment_status=SimpleNamespace(name=payload.get("alignment")),
    )
