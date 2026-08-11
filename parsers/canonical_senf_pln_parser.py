import logging
import re
from typing import List, Optional

from config import get_settings
from core import query_scoring
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.types import SENF, senf_from_payload
from core.senf.weave import WeaveResult, weave
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)

# A leading $ or ? marks a MeTTa variable; its name may collide with a mention symbol.
_TOKEN_RE = re.compile(r"[$?]?[A-Za-z_][A-Za-z0-9_]*")


def _telemetry(
    senf: SENF,
    graph=None,
    weave_result: Optional[WeaveResult] = None,
    rewritten: int = 0,
) -> dict:
    """Counts read off state the hook already built. Nothing is recomputed."""
    return {
        "frame_count": len(senf.frames),
        "mention_count": len(senf.mentions),
        "identity_edge_count": len(graph.edges) if graph else 0,
        "merge_count": graph.merge_count if graph else 0,
        "rewritten_atom_count": rewritten,
        "weave_distortion": (
            round(weave_result.distortion, 4) if weave_result else None
        ),
        "weave_pair_count": len(weave_result.pairs) if weave_result else 0,
    }


class CanonicalSENFPLNParser(CanonicalPLNParser):
    """canonical_pln with symbols unified across sentences by SENF identity resolution."""

    def __init__(self):
        super().__init__()
        cfg = get_settings()
        self._threshold = cfg.senf_identity_threshold
        self._context_top_k = cfg.senf_context_top_k
        self._max_frames = cfg.senf_session_max_frames
        self._use_vector_context = cfg.senf_use_vector_context
        self._source_grounding_weight = cfg.senf_source_grounding_weight
        self._role_compat_weight = cfg.senf_role_compat_weight
        self._distortion_weight = cfg.senf_distortion_weight
        self._vector_store = None
        self.reset()

    def reset(self) -> None:
        self._session: List[SENF] = []
        self._sentence_counter = 0
        self._weave: Optional[WeaveResult] = None
        self._telemetry: Optional[dict] = None

    def senf_telemetry(self) -> Optional[dict]:
        """Counts from the most recent hook call, or None if it never ran.

        Probed by the service through `hasattr`, so the base parser reporting
        nothing needs no coordination here.
        """
        return dict(self._telemetry) if self._telemetry else None

    def _post_filter_hook(
        self,
        texts: List[str],
        statements: List[str],
        queries: List[str],
        context: List[str],
        is_query: bool,
    ) -> tuple[List[str], List[str]]:
        self._telemetry = None
        if is_query:
            # Cleared before the empty-input guard: a question that yields no atoms
            # must not inherit the previous question's grounding.
            self._weave = None

        if not statements and not queries:
            return statements, queries

        try:
            text = " ".join(texts)
            self._sentence_counter += 1
            senf = extract_senf(f"s{self._sentence_counter}", text, statements + queries)
            if senf.is_empty:
                self._telemetry = _telemetry(senf)
                return statements, queries

            prior = self._prior_senfs(text)
            graph = resolve_identity(prior + [senf], threshold=self._threshold)
            if is_query:
                self._weave = weave(senf, prior, resolve=graph.resolve)
            # Only a question owns a weave; on ingest self._weave still holds the
            # previous question's, which is not this call's telemetry.
            reported = self._weave if is_query else None
            if not graph.representatives:
                self._telemetry = _telemetry(senf, graph, reported)
                self._remember(senf, is_query)
                return statements, queries

            rewritten_statements = [self._rewrite(stmt, graph) for stmt in statements]
            rewritten_queries = [self._rewrite(query, graph) for query in queries]
            changed = sum(
                1
                for before, after in zip(
                    statements + queries, rewritten_statements + rewritten_queries
                )
                if before != after
            )
            self._telemetry = _telemetry(senf, graph, reported, changed)
            self._remember(senf, is_query)
            return rewritten_statements, rewritten_queries
        except Exception:
            logger.exception("SENF identity resolution failed; using canonical_pln output")
            return statements, queries

    def _score_query_candidate(
        self,
        query: dict,
        facts: list[dict],
        conclusions: list[dict],
        is_yes_no: bool,
    ) -> int | None:
        return query_scoring.score_query_candidate(
            query, facts, conclusions, is_yes_no, senf=self._senf_signals()
        )

    def _senf_signals(self) -> Optional[query_scoring.SENFSignals]:
        if self._weave is None:
            return None
        return query_scoring.SENFSignals(
            grounded_symbols=self._weave.grounded_symbols,
            role_signatures=self._weave.role_signatures,
            distortion=self._weave.distortion,
            source_grounding_weight=self._source_grounding_weight,
            role_compat_weight=self._role_compat_weight,
            distortion_weight=self._distortion_weight,
        )

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
