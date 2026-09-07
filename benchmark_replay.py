"""Capture and replay raw Canonical NL2PLN generations for controlled benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


TAPE_SCHEMA_VERSION = 1


class GenerationTapeError(RuntimeError):
    """The replay no longer matches the captured benchmark call structure."""


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
        self.metadata = dict(metadata or {})
        self.calls: list[dict[str, Any]] = []
        self._by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._reports: list[dict[str, Any]] = []

        if mode == "replay":
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != TAPE_SCHEMA_VERSION:
                raise GenerationTapeError(
                    f"Unsupported generation tape schema: {payload.get('schema_version')!r}"
                )
            captured_metadata = payload.get("metadata")
            calls = payload.get("calls")
            if not isinstance(captured_metadata, dict) or not isinstance(calls, list):
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
                self.calls.append(call)
                self._by_scope[(case_id, phase)].append(call)

    def _validate_metadata(self, captured: dict[str, Any]) -> None:
        for key, current in self.metadata.items():
            if current is None:
                continue
            previous = captured.get(key)
            if previous != current:
                raise GenerationTapeError(
                    f"Generation tape metadata mismatch for {key}: "
                    f"captured={previous!r} current={current!r}"
                )

    def backend(self, parser_name: str, case_id: str) -> "GenerationBackend":
        if self.mode == "capture" and parser_name != "canonical_pln":
            raise GenerationTapeError("Capture supports canonical_pln only")
        return GenerationBackend(self, parser_name, case_id)

    def save(self) -> None:
        if self.mode != "capture":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TAPE_SCHEMA_VERSION,
            "metadata": self.metadata,
            "calls": self.calls,
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
        self._context_mismatches = 0
        self._repeated_calls = 0

    def set_phase(self, phase: str) -> None:
        if not phase:
            raise ValueError("Generation phase cannot be empty")
        self.phase = phase

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
            self.tape._by_scope[(self.case_id, self.phase)].append(call)
            return result

        entries = self.tape._by_scope.get((self.case_id, self.phase), [])
        if not entries:
            raise GenerationTapeError(
                f"No generation calls captured for {self.case_id}/{self.phase}"
            )
        repeated = index >= len(entries)
        if repeated and self.parser_name == "canonical_pln":
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
            if self.parser_name == "canonical_pln":
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

    def finish(self) -> None:
        expected = {
            phase: len(entries)
            for (case_id, phase), entries in self.tape._by_scope.items()
            if case_id == self.case_id
        }
        unconsumed = {
            phase: count - self._indices.get(phase, 0)
            for phase, count in expected.items()
            if count > self._indices.get(phase, 0)
        }
        if self.tape.mode == "replay" and self.parser_name == "canonical_pln" and unconsumed:
            raise GenerationTapeError(
                f"Unconsumed generation calls for {self.case_id}: {unconsumed}"
            )
        self.tape._reports.append(
            {
                "parser": self.parser_name,
                "case_id": self.case_id,
                "calls": sum(self._indices.values()),
                "context_mismatches": self._context_mismatches,
                "repeated_calls": self._repeated_calls,
                "unconsumed_calls": unconsumed,
            }
        )
        if self.tape.mode == "capture":
            self.tape.save()
