import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Settings has required fields and reads .env; pin them so tests never depend on
# developer environment or a real key.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434/api/embeddings")


class FakeVectorStore:
    """In-memory stand-in for storage.VectorStore. No Qdrant, no Ollama.

    Mirrors only the surface the service and parsers actually touch.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.points: list[dict] = []
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Deterministic pseudo-embedding: stable across runs, no network.
        return [((hash((text, i)) % 1000) / 1000.0) for i in range(self._dim)]

    def store(self, sentence: str, atoms: list[str], vector: list[float], metadata: dict | None = None):
        payload = {"nl": sentence, "pln": atoms}
        if metadata:
            payload.update(metadata)
        self.points.append({"vector": vector, "payload": payload})

    def store_many(self, records: list[dict], batch_size: int = 100) -> int:
        for record in records:
            self.points.append({"vector": self.embed(record["nl"]), "payload": record})
        return len(records)

    def retrieve_context(self, text: str, top_k: int) -> tuple[list[str], list[float]]:
        vector = self.embed(text)
        context: list[str] = []
        for point in self.points[-top_k:]:
            pln = point["payload"].get("pln", [])
            if isinstance(pln, list):
                context.extend(pln)
        return context, vector

    def retrieve_senf_context(self, text: str, top_k: int) -> list[dict]:
        from core.senf import SENF_PAYLOAD_KEY

        return [
            point["payload"][SENF_PAYLOAD_KEY]
            for point in self.points[-top_k:]
            if isinstance(point["payload"].get(SENF_PAYLOAD_KEY), dict)
        ]

    def lookup_sources_by_atoms(self, atoms: list[str], max_atoms: int = 30, score_threshold: float = 0.6) -> list[str]:
        return []

    def count_by_source(self, source: str) -> int:
        return sum(1 for p in self.points if p["payload"].get("source") == source)

    def delete_by_source(self, source: str):
        self.points = [p for p in self.points if p["payload"].get("source") != source]

    def reset(self):
        self.points.clear()

    @property
    def count(self) -> int:
        return len(self.points)

    def close(self):
        pass


@pytest.fixture
def fake_vector_store():
    return FakeVectorStore()


@pytest.fixture
def temp_atomspace(tmp_path, monkeypatch):
    """Point ATOMSPACE_PATH at a temp file and clear the settings cache."""
    path = tmp_path / "atomspace" / "kb.metta"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ATOMSPACE_PATH", str(path))

    from config import get_settings

    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def reasoner(temp_atomspace):
    """Real Reasoner over a temp atomspace. Skips if PeTTaChainer is unavailable."""
    pytest.importorskip("pettachainer", reason="PeTTaChainer only present in the container image")
    from core.reasoner import Reasoner

    return Reasoner()
