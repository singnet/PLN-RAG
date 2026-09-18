"""Capture and replay raw Canonical NL2PLN generations for controlled benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


TAPE_SCHEMA_VERSION = 2
_FEATURE_OVERFLOW_CLASS = "__senf_feature_overflow__"
_MISSING = object()


class GenerationTapeError(RuntimeError):
    """The replay no longer matches the captured benchmark call structure."""

    fail_closed = True


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


class GenerationTape:
    """One capture tape shared by all isolated benchmark cases and parser arms."""

    def __init__(
        self,
        mode: str,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ):
        if mode not in {"capture", "replay"}:
            raise ValueError(f"Unsupported generation tape mode: {mode}")
        self.mode = mode
        self.path = Path(path)
        self.schema_version = TAPE_SCHEMA_VERSION
        self.metadata = dict(metadata or {})
        self.calls: list[dict[str, Any]] = []
        self.feature_calls: list[dict[str, Any]] = []
        self._by_scope: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self._features_by_scope: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self._reports: list[dict[str, Any]] = []

        if mode == "replay":
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.schema_version = payload.get("schema_version")
            if self.schema_version not in (1, TAPE_SCHEMA_VERSION):
                raise GenerationTapeError(
                    f"Unsupported generation tape schema: {payload.get('schema_version')!r}"
                )
            captured_metadata = payload.get("metadata")
            calls = payload.get("calls")
            feature_calls = payload.get("feature_calls", [])
            if (
                not isinstance(captured_metadata, dict) or not isinstance(calls, list)
                or not isinstance(feature_calls, list)
            ):
                raise GenerationTapeError("Generation tape must contain metadata and calls")
            self._validate_metadata(captured_metadata)
            self.metadata = captured_metadata
            for call in calls:
                if not isinstance(call, dict):
                    raise GenerationTapeError("Generation tape contains a non-object call")
                case_id = str(call.get("case_id", ""))
                phase = str(call.get("phase", ""))
                if not case_id or not phase:
                    raise GenerationTapeError("Generation tape call is missing case_id or phase")
                parser = ""
                if self.schema_version >= 2:
                    parser = call.get("parser")
                    if not isinstance(parser, str) or not parser:
                        raise GenerationTapeError("Generation tape call is missing parser ownership")
                self.calls.append(call)
                self._by_scope[(parser, case_id, phase)].append(call)
            for call in feature_calls:
                if not isinstance(call, dict):
                    raise GenerationTapeError("Generation tape contains a non-object feature call")
                case_id = str(call.get("case_id", ""))
                phase = str(call.get("phase", ""))
                if not case_id or not phase:
                    raise GenerationTapeError("Feature call is missing case_id or phase")
                parser = ""
                if self.schema_version >= 2:
                    parser = call.get("parser")
                    if not isinstance(parser, str) or not parser:
                        raise GenerationTapeError("Feature call is missing parser ownership")
                self.feature_calls.append(call)
                self._features_by_scope[(parser, case_id, phase)].append(call)

    def _validate_metadata(self, captured: dict[str, Any]) -> None:
        for key, current in self.metadata.items():
            if current is None:
                continue
            # Schema 1 predates feature-provider capture. Preserve replay of
            # frozen generation-only tapes while validating its recorded fields.
            if self.schema_version == 1 and key not in captured:
                continue
            previous = captured.get(key)
            if previous != current:
                raise GenerationTapeError(
                    f"Generation tape metadata mismatch for {key}: "
                    f"captured={previous!r} current={current!r}"
                )

    def backend(self, parser_name: str, case_id: str) -> "GenerationBackend":
        return GenerationBackend(self, parser_name, case_id)

    def save(self) -> None:
        if self.mode != "capture":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TAPE_SCHEMA_VERSION,
            "metadata": self.metadata,
            "calls": self.calls,
            "feature_calls": self.feature_calls,
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(self.path)

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "path": str(self.path),
            "sha256": file_hash(self.path) if self.path.is_file() else None,
            "captured_calls": len(self.calls),
            "captured_feature_calls": len(self.feature_calls),
            "metadata": self.metadata,
            "parser_runs": list(self._reports),
        }


class GenerationBackend:
    """Parser-local cursor over one case's capture or replay calls."""

    fail_closed = True

    def __init__(self, tape: GenerationTape, parser_name: str, case_id: str):
        self.tape = tape
        self.parser_name = parser_name
        self.case_id = case_id
        self.phase = "unscoped"
        self._indices: dict[str, int] = defaultdict(int)
        self._feature_indices: dict[str, int] = defaultdict(int)
        self._context_mismatches = 0
        self._repeated_calls = 0

    def set_phase(self, phase: str) -> None:
        if not phase:
            raise ValueError("Generation phase cannot be empty")
        self.phase = phase

    def _scope(self, phase: str) -> tuple[str, str, str]:
        parser = self.parser_name if self.tape.schema_version >= 2 else ""
        return parser, self.case_id, phase

    @property
    def _strict_generation(self) -> bool:
        return self.tape.schema_version >= 2 or self.parser_name == "canonical_pln"

    def generate(
        self,
        *,
        sentences: list[str],
        context: list[str],
        pln_spec: Any,
        live: Callable[..., Any],
    ) -> Any:
        if self.phase == "unscoped":
            raise GenerationTapeError("Generation call has no benchmark phase")
        index = self._indices[self.phase]
        self._indices[self.phase] += 1

        if self.tape.mode == "capture":
            result = live(sentences=sentences, context=context, pln_spec=pln_spec)
            call = {
                "parser": self.parser_name,
                "case_id": self.case_id,
                "phase": self.phase,
                "call_index": index,
                "sentences": list(sentences),
                "context": list(context),
                "context_sha256": content_hash(context),
                "pln_spec_sha256": content_hash(str(pln_spec)),
                "statements": list(result.statements or []),
                "queries": list(result.queries or []),
            }
            self.tape.calls.append(call)
            self.tape._by_scope[self._scope(self.phase)].append(call)
            return result

        entries = self.tape._by_scope.get(self._scope(self.phase), [])
        if not entries:
            raise GenerationTapeError(
                f"No generation calls captured for {self.case_id}/{self.phase}"
            )
        repeated = index >= len(entries)
        if repeated and self._strict_generation:
            raise GenerationTapeError(
                f"Unexpected extra generation call for {self.case_id}/{self.phase}/{index}"
            )
        entry = entries[-1] if repeated else entries[index]
        if repeated:
            self._repeated_calls += 1
        if entry.get("call_index") != min(index, len(entries) - 1):
            raise GenerationTapeError(
                f"Unexpected call index for {self.case_id}/{self.phase}: {index}"
            )
        if entry.get("sentences") != list(sentences):
            raise GenerationTapeError(
                f"Sentence mismatch for {self.case_id}/{self.phase}/{index}"
            )
        context_matches = entry.get("context_sha256") == content_hash(context)
        if not context_matches:
            if self._strict_generation:
                raise GenerationTapeError(
                    f"Context mismatch for {self.case_id}/{self.phase}/{index}"
                )
            self._context_mismatches += 1
        if entry.get("pln_spec_sha256") != content_hash(str(pln_spec)):
            raise GenerationTapeError(
                f"PLN specification mismatch for {self.case_id}/{self.phase}/{index}"
            )
        return SimpleNamespace(
            statements=list(entry.get("statements") or []),
            queries=list(entry.get("queries") or []),
        )

    def feature_call(
        self,
        *,
        request: dict[str, Any],
        config: dict[str, Any],
        live: Callable[[], Any],
    ) -> list[Any]:
        """Capture/replay feature extraction with a strict parser-local cursor."""
        if self.phase == "unscoped":
            raise GenerationTapeError("Feature call has no benchmark phase")
        index = self._feature_indices[self.phase]
        self._feature_indices[self.phase] += 1
        request_hash = content_hash(request)
        config_hash = content_hash(config)
        text_hash = content_hash(request.get("text"))
        if self.tape.mode == "capture":
            try:
                cap = config.get("max_features")
                if type(cap) is not int or cap <= 0:
                    raise ValueError("feature config requires a positive max_features")
                iterator = iter(live())
                result = list(islice(iterator, cap))
                if next(iterator, _MISSING) is not _MISSING:
                    result.append(_feature_overflow_sentinel())
            except Exception as exc:
                raise GenerationTapeError(
                    f"Feature capture failed for {self.case_id}/{self.phase}/{index}"
                ) from exc
            call = {
                "parser": self.parser_name,
                "case_id": self.case_id,
                "phase": self.phase,
                "call_index": index,
                "text_sha256": text_hash,
                "request_sha256": request_hash,
                "config_sha256": config_hash,
                "extractions": [_extraction_to_payload(item) for item in result],
            }
            self.tape.feature_calls.append(call)
            self.tape._features_by_scope[self._scope(self.phase)].append(call)
            return result

        entries = self.tape._features_by_scope.get(self._scope(self.phase), [])
        if index >= len(entries):
            raise GenerationTapeError(
                f"Unexpected extra feature call for {self.case_id}/{self.phase}/{index}"
            )
        entry = entries[index]
        if entry.get("call_index") != index:
            raise GenerationTapeError(
                f"Unexpected feature call index for {self.case_id}/{self.phase}: {index}"
            )
        checks = {
            "text": (entry.get("text_sha256"), text_hash),
            "request": (entry.get("request_sha256"), request_hash),
            "config": (entry.get("config_sha256"), config_hash),
        }
        for name, (captured, current) in checks.items():
            if captured != current:
                raise GenerationTapeError(
                    f"Feature {name} mismatch for {self.case_id}/{self.phase}/{index}"
                )
        extractions = entry.get("extractions")
        if not isinstance(extractions, list):
            raise GenerationTapeError("Feature call has malformed extractions")
        return [_extraction_from_payload(item) for item in extractions]

    def finish(self) -> None:
        expected = {
            phase: len(entries)
            for (parser, case_id, phase), entries in self.tape._by_scope.items()
            if case_id == self.case_id
            and (self.tape.schema_version == 1 or parser == self.parser_name)
        }
        unconsumed = {
            phase: count - self._indices.get(phase, 0)
            for phase, count in expected.items()
            if count > self._indices.get(phase, 0)
        }
        expected_features = {
            phase: len(entries)
            for (parser, case_id, phase), entries in self.tape._features_by_scope.items()
            if case_id == self.case_id
            and (self.tape.schema_version == 1 or parser == self.parser_name)
        }
        unconsumed_features = {
            phase: count - self._feature_indices.get(phase, 0)
            for phase, count in expected_features.items()
            if count > self._feature_indices.get(phase, 0)
        }
        if self.tape.mode == "replay" and self._strict_generation and unconsumed:
            raise GenerationTapeError(
                f"Unconsumed generation calls for {self.case_id}: {unconsumed}"
            )
        if self.tape.mode == "replay" and unconsumed_features:
            raise GenerationTapeError(
                f"Unconsumed feature calls for {self.case_id}: {unconsumed_features}"
            )
        self.tape._reports.append(
            {
                "parser": self.parser_name,
                "case_id": self.case_id,
                "calls": sum(self._indices.values()),
                "context_mismatches": self._context_mismatches,
                "repeated_calls": self._repeated_calls,
                "unconsumed_calls": unconsumed,
                "feature_calls": sum(self._feature_indices.values()),
                "unconsumed_feature_calls": unconsumed_features,
            }
        )
        if self.tape.mode == "capture":
            self.tape.save()


def _extraction_to_payload(extraction: Any) -> dict[str, Any]:
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


def _feature_overflow_sentinel() -> Any:
    return SimpleNamespace(
        extraction_class=_FEATURE_OVERFLOW_CLASS,
        extraction_text=None,
        attributes=None,
        char_interval=SimpleNamespace(start_pos=None, end_pos=None),
        alignment_status=SimpleNamespace(name=None),
    )


def _extraction_from_payload(payload: Any) -> Any:
    fields = {
        "extraction_class", "extraction_text", "attributes",
        "start", "end", "alignment",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise GenerationTapeError("Feature extraction has an invalid schema")
    if (
        payload["extraction_class"] is not None
        and not isinstance(payload["extraction_class"], str)
        or payload["extraction_text"] is not None
        and not isinstance(payload["extraction_text"], str)
        or payload["attributes"] is not None
        and not isinstance(payload["attributes"], dict)
        or payload["start"] is not None
        and type(payload["start"]) is not int
        or payload["end"] is not None
        and type(payload["end"]) is not int
        or not isinstance(payload["alignment"], str)
    ):
        raise GenerationTapeError("Feature extraction has invalid field types")
    return SimpleNamespace(
        extraction_class=payload.get("extraction_class"),
        extraction_text=payload.get("extraction_text"),
        attributes=payload.get("attributes"),
        char_interval=SimpleNamespace(
            start_pos=payload.get("start"), end_pos=payload.get("end")
        ),
        alignment_status=SimpleNamespace(name=payload.get("alignment")),
    )
