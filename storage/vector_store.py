import uuid
import httpx
from typing import List, Tuple
from config import get_settings


class VectorStore:
    """
    Manages NL ↔ PLN atom mappings in Qdrant.

    Stores: { nl: sentence, pln: [atoms] } per ingested sentence.
    Retrieves: relevant PLN atoms to use as parser context.
    """

    def __init__(self):
        cfg = get_settings()
        self._qdrant = cfg.qdrant_url
        self._ollama = cfg.ollama_url
        self._ollama_model = cfg.ollama_model
        self._collection = cfg.qdrant_collection
        self._client = httpx.Client(timeout=30)
        self._vector_size: int | None = None

    def embed(self, text: str) -> List[float]:
        resp = self._client.post(self._ollama, json={
            "model": self._ollama_model,
            "prompt": text
        })
        resp.raise_for_status()
        return resp.json()["embedding"]

    def is_qdrant_available(self) -> tuple[bool, str]:
        try:
            resp = self._client.get(f"{self._qdrant}/collections")
            resp.raise_for_status()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def is_ollama_available(self) -> tuple[bool, str]:
        base_url = self._ollama
        if "/api/embeddings" in base_url:
            base_url = base_url.rsplit("/api/embeddings", 1)[0]
        try:
            resp = self._client.get(base_url)
            resp.raise_for_status()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def _ensure_collection(self, vector_size: int):
        if self._vector_size == vector_size:
            return
        try:
            self._client.get(f"{self._qdrant}/collections/{self._collection}").raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._client.put(
                    f"{self._qdrant}/collections/{self._collection}",
                    json={"vectors": {"size": vector_size, "distance": "Cosine"}}
                ).raise_for_status()
        self._vector_size = vector_size

    def store(self, sentence: str, atoms: List[str], vector: List[float]):
        self._ensure_collection(len(vector))
        self._client.put(
            f"{self._qdrant}/collections/{self._collection}/points?wait=true",
            json={"points": [{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {"nl": sentence, "pln": atoms}
            }]}
        ).raise_for_status()

    def store_many(self, records: List[dict], batch_size: int = 100) -> int:
        if not records:
            return 0
        stored = 0
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            points = []
            vector_size: int | None = None
            for record in chunk:
                vector = self.embed(record["nl"])
                vector_size = len(vector)
                points.append(
                    {
                        "id": str(uuid.uuid4()),
                        "vector": vector,
                        "payload": record,
                    }
                )
            if vector_size is None:
                continue
            self._ensure_collection(vector_size)
            self._client.put(
                f"{self._qdrant}/collections/{self._collection}/points?wait=true",
                json={"points": points},
            ).raise_for_status()
            stored += len(points)
        return stored

    def retrieve_context(self, text: str, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Returns (context_atoms, embedding_vector).
        context_atoms: flat list of PLN atom strings from top-k similar sentences.
        """
        vector = self.embed(text)
        self._ensure_collection(len(vector))

        resp = self._client.post(
            f"{self._qdrant}/collections/{self._collection}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True}
        )
        if resp.status_code != 200:
            return [], vector

        context: List[str] = []
        for item in resp.json().get("result", []):
            pln = item.get("payload", {}).get("pln", [])
            if isinstance(pln, list):
                context.extend(pln)

        return context, vector

    def reset(self):
        try:
            self._client.delete(f"{self._qdrant}/collections/{self._collection}")
        except Exception:
            pass
        self._vector_size = None

    @property
    def count(self) -> int:
        try:
            resp = self._client.get(f"{self._qdrant}/collections/{self._collection}")
            return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            return 0

    def count_by_source(self, source: str) -> int:
        try:
            resp = self._client.post(
                f"{self._qdrant}/collections/{self._collection}/points/count",
                json={
                    "filter": {
                        "must": [
                            {"key": "source", "match": {"value": source}}
                        ]
                    }
                },
            )
            if resp.status_code != 200:
                return 0
            return resp.json().get("result", {}).get("count", 0)
        except Exception:
            return 0

    def delete_by_source(self, source: str):
        try:
            self._client.post(
                f"{self._qdrant}/collections/{self._collection}/points/delete?wait=true",
                json={
                    "filter": {
                        "must": [
                            {"key": "source", "match": {"value": source}}
                        ]
                    }
                },
            ).raise_for_status()
        except Exception:
            pass
