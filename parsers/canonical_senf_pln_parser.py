import logging
import re
from typing import List

from config import get_settings
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.types import SENF, senf_from_payload
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)

# A leading $ or ? marks a MeTTa variable; its name may collide with a mention symbol.
_TOKEN_RE = re.compile(r"[$?]?[A-Za-z_][A-Za-z0-9_]*")


class CanonicalSENFPLNParser(CanonicalPLNParser):
    """canonical_pln with symbols unified across sentences by SENF identity resolution."""

    def __init__(self):
        super().__init__()
        cfg = get_settings()
        self._threshold = cfg.senf_identity_threshold
        self._context_top_k = cfg.senf_context_top_k
        self._max_frames = cfg.senf_session_max_frames
        self._use_vector_context = cfg.senf_use_vector_context
        self._vector_store = None
        self.reset()

    def reset(self) -> None:
        self._session: List[SENF] = []
        self._sentence_counter = 0

    def _post_filter_hook(
        self,
        texts: List[str],
        statements: List[str],
        queries: List[str],
        context: List[str],
        is_query: bool,
    ) -> tuple[List[str], List[str]]:
        if not statements and not queries:
            return statements, queries

        try:
            text = " ".join(texts)
            self._sentence_counter += 1
            senf = extract_senf(f"s{self._sentence_counter}", text, statements + queries)
            if senf.is_empty:
                return statements, queries

            graph = resolve_identity(
                self._prior_senfs(text) + [senf], threshold=self._threshold
            )
            if not graph.representatives:
                self._remember(senf, is_query)
                return statements, queries

            statements = [self._rewrite(stmt, graph) for stmt in statements]
            queries = [self._rewrite(query, graph) for query in queries]
            self._remember(senf, is_query)
            return statements, queries
        except Exception:
            logger.exception("SENF identity resolution failed; using canonical_pln output")
            return statements, queries

    def _prior_senfs(self, text: str) -> List[SENF]:
        """Session SENF first, then anything the vector store recalls for this text."""
        prior = list(self._session)
        seen = {senf.sentence_id for senf in prior}
        for blob in self._retrieve_senf_blobs(text):
            recalled = senf_from_payload(blob)
            if recalled and not recalled.is_empty and recalled.sentence_id not in seen:
                seen.add(recalled.sentence_id)
                prior.append(recalled)
        return prior

    def _retrieve_senf_blobs(self, text: str) -> List[dict]:
        if not self._use_vector_context or self._context_top_k <= 0:
            return []
        try:
            store = self._store()
            return store.retrieve_senf_context(text, self._context_top_k) if store else []
        except Exception:
            logger.warning("SENF context retrieval failed; continuing session-only", exc_info=True)
            return []

    def _store(self):
        if self._vector_store is None:
            from storage.vector_store import VectorStore

            self._vector_store = VectorStore()
        return self._vector_store

    def _remember(self, senf: SENF, is_query: bool) -> None:
        """Queries are transient, so only ingested sentences join the session."""
        if is_query:
            return
        self._session.append(senf)
        frames = 0
        keep = 0
        for kept in reversed(self._session):
            frames += len(kept.frames)
            if frames > self._max_frames and keep:
                break
            keep += 1
        self._session = self._session[len(self._session) - keep :]

    @staticmethod
    def _rewrite(expression: str, graph) -> str:
        def replace(match: re.Match) -> str:
            token = match.group(0)
            if token[0] in "$?":
                return token
            return graph.resolve(token)

        return _TOKEN_RE.sub(replace, expression)
