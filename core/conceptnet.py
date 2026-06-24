import json
import os
import subprocess
import sys
import threading
from typing import Any

from config import get_settings
from core.reasoner import Reasoner
from core.symbol_normalization import NORMALIZATION_VERSION
from storage.vector_store import VectorStore


class ConceptNetManager:
    def __init__(self):
        self._cfg = get_settings()
        self._lock = threading.Lock()
        self._index_thread: threading.Thread | None = None
        self._indexing = False
        self._indexed_count = 0
        self._expected_count = 0
        self._last_error = ""

    @property
    def enabled(self) -> bool:
        return self._cfg.conceptnet_enabled

    def ensure_loaded(self, reasoner: Reasoner, vector_store: VectorStore):
        if not self.enabled:
            return

        regenerated = self._ensure_artifacts_current()
        if regenerated:
            vector_store.delete_by_source("conceptnet")

        if self._cfg.conceptnet_autoload:
            self.ensure_atomspace_loaded(reasoner)
        if self._cfg.conceptnet_index_on_startup:
            self.ensure_vector_index(vector_store, force=regenerated, background=True)

    def ensure_atomspace_loaded(self, reasoner: Reasoner):
        path = self._cfg.conceptnet_atomspace_path
        if not os.path.exists(path):
            self._handle_missing(f"ConceptNet atomspace file not found: {path}")
            return
        reasoner.load_background_file(path)

    def ensure_vector_index(
        self, vector_store: VectorStore, force: bool = False, background: bool = False
    ):
        payload_path = self._cfg.conceptnet_vector_payload_path
        if not os.path.exists(payload_path):
            self._handle_missing(f"ConceptNet vector payload file not found: {payload_path}")
            return

        expected = self._expected_vector_count()
        existing = vector_store.count_by_source("conceptnet")
        self._set_status(indexed_count=existing, expected_count=expected)
        if not force and existing > 0 and (expected == 0 or existing >= expected):
            return

        if background:
            self._start_background_index(vector_store, force=force)
            return

        self._index_vectors(vector_store, force=force)

    def restore_after_reset(self, reasoner: Reasoner, vector_store: VectorStore, scope: str):
        if not self.enabled:
            return
        if scope in ("all", "atomspace") and self._cfg.conceptnet_autoload:
            self.ensure_atomspace_loaded(reasoner)
        if (
            scope in ("all", "vectordb")
            and self._cfg.conceptnet_index_on_startup
            and self._cfg.conceptnet_reindex_on_reset
        ):
            self.ensure_vector_index(vector_store, force=True, background=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "indexing": self._indexing,
                "indexed_count": self._indexed_count,
                "expected_count": self._expected_count,
                "last_error": self._last_error,
            }

    def _start_background_index(self, vector_store: VectorStore, force: bool):
        with self._lock:
            if self._index_thread is not None and self._index_thread.is_alive():
                return
            self._indexing = True
            self._last_error = ""
            self._index_thread = threading.Thread(
                target=self._index_vectors,
                args=(vector_store,),
                kwargs={"force": force},
                daemon=True,
                name="conceptnet-indexer",
            )
            self._index_thread.start()

    def _index_vectors(self, vector_store: VectorStore, force: bool = False):
        payload_path = self._cfg.conceptnet_vector_payload_path
        expected = self._expected_vector_count()
        indexed = vector_store.count_by_source("conceptnet")
        self._set_status(indexing=True, indexed_count=indexed, expected_count=expected, last_error="")

        try:
            if force and indexed > 0:
                vector_store.delete_by_source("conceptnet")
                indexed = 0
                self._set_status(indexed_count=indexed, expected_count=expected)
            elif expected > 0 and 0 < indexed < expected:
                print(
                    f"[ConceptNet] Partial vector index detected: {indexed}/{expected}. Reindexing."
                )
                vector_store.delete_by_source("conceptnet")
                indexed = 0
                self._set_status(indexed_count=indexed, expected_count=expected)

            batch: list[dict[str, Any]] = []
            for record in self._iter_records(payload_path):
                batch.append(record)
                if len(batch) >= 100:
                    indexed += vector_store.store_many(batch)
                    self._set_status(indexed_count=indexed, expected_count=expected)
                    print(f"[ConceptNet] Indexed {indexed}/{expected or '?'} vectors")
                    batch = []
            if batch:
                indexed += vector_store.store_many(batch)
                self._set_status(indexed_count=indexed, expected_count=expected)
                print(f"[ConceptNet] Indexed {indexed}/{expected or '?'} vectors")
            self._set_status(indexing=False, indexed_count=indexed, expected_count=expected)
        except Exception as exc:
            self._set_status(indexing=False, indexed_count=indexed, expected_count=expected, last_error=str(exc))
            print(f"[ConceptNet] Vector indexing failed: {exc}")
            return

    def _iter_records(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "nl" not in record or "pln" not in record:
                    continue
                yield record

    def _ensure_artifacts_current(self) -> bool:
        if not self._cfg.conceptnet_auto_rebuild_on_change:
            return False

        input_exists = os.path.exists(self._cfg.conceptnet_input_file)
        artifact_paths = [
            self._cfg.conceptnet_atomspace_path,
            self._cfg.conceptnet_vector_payload_path,
            self._cfg.conceptnet_manifest_path,
        ]
        artifacts_exist = all(os.path.exists(path) for path in artifact_paths)

        if not input_exists:
            self._handle_missing(
                "ConceptNet raw input file not found; using committed generated artifacts if available: "
                f"{self._cfg.conceptnet_input_file}"
            )
            return False

        required_paths = [
            self._cfg.conceptnet_input_file,
            self._cfg.conceptnet_atomspace_path,
            self._cfg.conceptnet_vector_payload_path,
            self._cfg.conceptnet_manifest_path,
        ]
        if not all(os.path.exists(path) for path in required_paths):
            if not artifacts_exist:
                self._rebuild_artifacts()
                return True
            return False

        manifest = self._load_manifest()
        if manifest is None or self._manifest_mismatch(manifest):
            self._rebuild_artifacts()
            return True
        return False

    def _load_manifest(self) -> dict[str, Any] | None:
        path = self._cfg.conceptnet_manifest_path
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _expected_vector_count(self) -> int:
        manifest = self._load_manifest()
        if manifest is None:
            return 0
        try:
            return int(manifest.get("vector_record_count", 0))
        except (TypeError, ValueError):
            return 0

    def _manifest_mismatch(self, manifest: dict[str, Any]) -> bool:
        expected = {
            "source_file": self._normalize_path(self._cfg.conceptnet_input_file),
            "normalization_version": NORMALIZATION_VERSION,
            "min_weight": float(self._cfg.conceptnet_min_weight),
            "coverage_percent": float(self._cfg.conceptnet_coverage_percent),
            "sample_seed": int(self._cfg.conceptnet_sample_seed),
        }
        for key, value in expected.items():
            current = manifest.get(key)
            if key == "source_file":
                current = self._normalize_path(current)
            if current != value:
                print(
                    f"[ConceptNet] Config change detected for {key}: "
                    f"manifest={manifest.get(key)!r}, current={value!r}"
                )
                return True
        return False

    def _normalize_path(self, path: Any) -> str:
        if not path:
            return ""
        return os.path.abspath(str(path))

    def _rebuild_artifacts(self):
        if not os.path.exists(self._cfg.conceptnet_input_file):
            self._handle_missing(
                f"ConceptNet raw input file not found: {self._cfg.conceptnet_input_file}"
            )
            return
        input_dir = os.path.dirname(self._cfg.conceptnet_atomspace_path)
        os.makedirs(input_dir, exist_ok=True)
        command = [
            sys.executable,
            "scripts/conceptnet/export_conceptnet.py",
            "--input",
            self._cfg.conceptnet_input_file,
            "--atomspace-output",
            self._cfg.conceptnet_atomspace_path,
            "--vector-output",
            self._cfg.conceptnet_vector_payload_path,
            "--manifest-output",
            self._cfg.conceptnet_manifest_path,
            "--min-weight",
            str(self._cfg.conceptnet_min_weight),
            "--coverage-percent",
            str(self._cfg.conceptnet_coverage_percent),
            "--sample-seed",
            str(self._cfg.conceptnet_sample_seed),
        ]
        print("[ConceptNet] Rebuilding background artifacts...")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            self._set_status(last_error=str(exc))
            if self._cfg.conceptnet_startup_fail_open:
                print(f"[ConceptNet] Warning: artifact rebuild failed: {exc}")
                return
            raise
        print("[ConceptNet] Background artifacts rebuilt.")

    def _handle_missing(self, message: str):
        if self._cfg.conceptnet_startup_fail_open:
            print(f"[ConceptNet] Warning: {message}")
            return
        raise FileNotFoundError(message)

    def _set_status(
        self,
        *,
        indexing: bool | None = None,
        indexed_count: int | None = None,
        expected_count: int | None = None,
        last_error: str | None = None,
    ):
        with self._lock:
            if indexing is not None:
                self._indexing = indexing
            if indexed_count is not None:
                self._indexed_count = indexed_count
            if expected_count is not None:
                self._expected_count = expected_count
            if last_error is not None:
                self._last_error = last_error
