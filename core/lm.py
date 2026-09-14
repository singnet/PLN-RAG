import hashlib
import json
import os
import tempfile
import threading
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import dspy

from config import get_settings


_CASSETTE_VERSION = 1
_MODES = {"off", "capture", "replay"}
_SECRET_KEYS = {
    "api_key",
    "api_token",
    "authorization",
    "headers",
    "password",
    "secret",
    "token",
}


class CassetteError(RuntimeError):
    """Raised when a cassette cannot satisfy a deterministic replay."""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if callable(value):
        return {
            "type": "callable",
            "name": getattr(
                value,
                "__qualname__",
                getattr(value, "__name__", type(value).__name__),
            ),
        }
    if hasattr(value, "model_dump"):
        return _canonical_value(value.model_dump(mode="json"))
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def _request_hash(lm: "CassetteLM", prompt: Any, messages: Any, kwargs: dict) -> str:
    request = {
        "model": lm.model,
        "model_type": lm.model_type,
        "purpose": lm.purpose,
        "prompt": prompt,
        "messages": messages,
        "lm_kwargs": lm.kwargs,
        "forward_kwargs": kwargs,
    }
    encoded = json.dumps(
        _canonical_value(request),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_response(response: Any) -> dict[str, Any]:
    from litellm.types.utils import ModelResponse

    if isinstance(response, ModelResponse):
        return {
            "type": "litellm.ModelResponse",
            "value": response.model_dump(mode="json"),
        }
    value = _canonical_value(response)
    # Fail during capture rather than produce a cassette that cannot be replayed.
    json.dumps(value, sort_keys=True)
    return {"type": "json", "value": value}


def _deserialize_response(response: dict[str, Any]) -> Any:
    response_type = response.get("type")
    value = response.get("value")
    if response_type == "litellm.ModelResponse":
        from litellm.types.utils import ModelResponse

        if not isinstance(value, dict):
            raise CassetteError("Cassette ModelResponse payload is not an object")
        return ModelResponse(**value)
    if response_type == "json":
        return value
    raise CassetteError(f"Unsupported cassette response type: {response_type!r}")


class _CassetteCoordinator:
    def __init__(self, mode: str, path: Path, scope: str):
        self.mode = mode
        self.path = path
        self.scope = scope
        self.lock = threading.RLock()
        self.hits = 0
        self.live_calls = 0
        self.misses = 0
        self.calls = 0
        self._occurrences: dict[str, int] = defaultdict(int)
        self._trace_cursor = 0
        self._data = self._load()
        scopes = self._data["scopes"]
        if mode == "capture":
            scopes[scope] = []
            self.replay_valid = True
        else:
            self.replay_valid = isinstance(scopes.get(scope), list)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": _CASSETTE_VERSION, "responses": {}, "scopes": {}}
        try:
            with self.path.open(encoding="utf-8") as cassette_file:
                data = json.load(cassette_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise CassetteError(f"Unable to read cassette {self.path}: {exc}") from exc
        if data.get("version") != _CASSETTE_VERSION:
            raise CassetteError(
                f"Unsupported cassette version {data.get('version')!r}; expected {_CASSETTE_VERSION}"
            )
        if not isinstance(data.get("responses"), dict) or not isinstance(
            data.get("scopes"), dict
        ):
            raise CassetteError("Cassette must contain response pools and scopes")
        return data

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as cassette_file:
                json.dump(self._data, cassette_file, indent=2, sort_keys=True)
                cassette_file.write("\n")
                cassette_file.flush()
                os.fsync(cassette_file.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def forward(
        self,
        lm: "CassetteLM",
        request_hash: str,
        prompt: Any,
        messages: Any,
        kwargs: dict,
    ) -> Any:
        with self.lock:
            self.calls += 1
            if self.mode == "replay":
                return self._replay(lm, request_hash)
            return self._capture(lm, request_hash, prompt, messages, kwargs)

    def _capture(
        self,
        lm: "CassetteLM",
        request_hash: str,
        prompt: Any,
        messages: Any,
        kwargs: dict,
    ) -> Any:
        occurrence = self._occurrences[request_hash]
        self._occurrences[request_hash] += 1
        pool = self._data["responses"].setdefault(request_hash, [])
        if occurrence < len(pool):
            response = _deserialize_response(pool[occurrence])
            self.hits += 1
        else:
            self.misses += 1
            self.live_calls += 1
            response = super(CassetteLM, lm).forward(
                prompt=prompt, messages=messages, **kwargs
            )
            pool.append(_serialize_response(response))

        self._data["scopes"][self.scope].append(
            {
                "request_hash": request_hash,
                "occurrence": occurrence,
                "purpose": lm.purpose,
            }
        )
        self._write()
        return response

    def _replay(self, lm: "CassetteLM", request_hash: str) -> Any:
        trace = self._data["scopes"].get(self.scope)
        if not isinstance(trace, list) or self._trace_cursor >= len(trace):
            return self._replay_failure(
                f"Cassette scope {self.scope!r} has no response for call {self._trace_cursor}"
            )

        entry = trace[self._trace_cursor]
        expected_hash = entry.get("request_hash") if isinstance(entry, dict) else None
        if expected_hash != request_hash:
            return self._replay_failure(
                f"Cassette request mismatch at call {self._trace_cursor}: "
                f"expected {expected_hash}, got {request_hash}"
            )
        if entry.get("purpose") != lm.purpose:
            return self._replay_failure(
                f"Cassette purpose mismatch at call {self._trace_cursor}: "
                f"expected {entry.get('purpose')!r}, got {lm.purpose!r}"
            )

        occurrence = entry.get("occurrence")
        pool = self._data["responses"].get(request_hash)
        expected_occurrence = self._occurrences[request_hash]
        if occurrence != expected_occurrence:
            return self._replay_failure(
                f"Cassette occurrence mismatch for hash {request_hash}: "
                f"expected {expected_occurrence}, got {occurrence!r}"
            )
        if (
            not isinstance(occurrence, int)
            or not isinstance(pool, list)
            or occurrence >= len(pool)
        ):
            return self._replay_failure(
                f"Cassette response missing for hash {request_hash}, occurrence {occurrence!r}"
            )

        try:
            response = _deserialize_response(pool[occurrence])
        except Exception:
            self.replay_valid = False
            self.misses += 1
            raise
        self._trace_cursor += 1
        self._occurrences[request_hash] += 1
        self.hits += 1
        return response

    def _replay_failure(self, message: str) -> Any:
        self.replay_valid = False
        self.misses += 1
        raise CassetteError(message)

    def status(self) -> dict[str, Any]:
        with self.lock:
            trace = self._data["scopes"].get(self.scope)
            trace_length = len(trace) if isinstance(trace, list) else 0
            return {
                "mode": self.mode,
                "path": str(self.path),
                "scope": self.scope,
                "hits": self.hits,
                "live_calls": self.live_calls,
                "misses": self.misses,
                "calls": self.calls,
                "replay_valid": self.replay_valid,
                "unconsumed": (
                    max(0, trace_length - self._trace_cursor)
                    if self.mode == "replay"
                    else 0
                ),
            }


_coordinator_lock = threading.Lock()
_coordinator: _CassetteCoordinator | None = None
_coordinator_key: tuple[str, str, str] | None = None


def _cassette_config() -> tuple[str, str | None, str]:
    mode = os.environ.get("LLM_CASSETTE_MODE", "off").strip().lower()
    if mode not in _MODES:
        raise ValueError(
            f"LLM_CASSETTE_MODE must be one of {sorted(_MODES)}, got {mode!r}"
        )
    path = os.environ.get("LLM_CASSETTE_PATH")
    scope = os.environ.get("LLM_CASSETTE_SCOPE", "default").strip() or "default"
    if mode != "off" and not path:
        raise ValueError(f"LLM_CASSETTE_PATH is required in {mode} mode")
    return mode, path, scope


def _get_coordinator(mode: str, path: str, scope: str) -> _CassetteCoordinator:
    global _coordinator, _coordinator_key
    key = (mode, str(Path(path).expanduser().resolve()), scope)
    with _coordinator_lock:
        if _coordinator is None or _coordinator_key != key:
            _coordinator = _CassetteCoordinator(mode, Path(key[1]), scope)
            _coordinator_key = key
        return _coordinator


class CassetteLM(dspy.LM):
    """DSPy LM that captures or deterministically replays forward calls."""

    def __init__(
        self,
        model: str,
        *,
        purpose: str,
        coordinator: _CassetteCoordinator,
        **kwargs,
    ):
        self.purpose = purpose
        self._cassette_coordinator = coordinator
        super().__init__(model, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        request_hash = _request_hash(self, prompt, messages, kwargs)
        return self._cassette_coordinator.forward(
            self, request_hash, prompt, messages, kwargs
        )


def cassette_status() -> dict[str, Any]:
    """Return process-local cassette counters and replay completeness."""
    mode, path, scope = _cassette_config()
    if mode == "off":
        return {
            "mode": mode,
            "path": path,
            "scope": scope,
            "hits": 0,
            "live_calls": 0,
            "misses": 0,
            "calls": 0,
            "replay_valid": True,
            "unconsumed": 0,
        }
    return _get_coordinator(mode, path, scope).status()  # type: ignore[arg-type]


def reset_cassette_state() -> None:
    """Reset shared cassette state and cursors; intended for isolated tests."""
    global _coordinator, _coordinator_key
    with _coordinator_lock:
        _coordinator = None
        _coordinator_key = None


def create_lm(purpose: str = "unspecified") -> dspy.LM:
    """Create the configured OpenAI-compatible DSPy language model."""
    cfg = get_settings()
    kwargs = {
        "api_key": cfg.openai_api_key,
        "cache": False,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url

    mode, path, scope = _cassette_config()
    if mode == "off":
        return dspy.LM(cfg.openai_model, **kwargs)
    coordinator = _get_coordinator(mode, path, scope)  # type: ignore[arg-type]
    return CassetteLM(
        cfg.openai_model, purpose=purpose, coordinator=coordinator, **kwargs
    )
