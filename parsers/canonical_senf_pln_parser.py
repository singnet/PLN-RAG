import logging
import re
import uuid
from typing import List, Optional

from config import get_settings
from core import query_scoring
from core.senf.bridge import identity_bridge_atoms, transport_truth
from core.senf.exemplars import score_exemplars
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.types import SENF, SENF_PAYLOAD_KEY, senf_from_payload, senf_to_payload
from core.senf.weave import WeaveResult, build_weaves
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)

# A leading $ or ? marks a MeTTa variable; its name may collide with a mention symbol.
_TOKEN_RE = re.compile(r"[$?]?[A-Za-z_][A-Za-z0-9_]*")
_STV_RE = re.compile(r"\(STV\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\)")


def _telemetry(
    senf: SENF,
    graph=None,
    weave_result: Optional[WeaveResult] = None,
    rewritten: int = 0,
    bridge_count: int = 0,
) -> dict:
    """Counts read off state the hook already built. Nothing is recomputed."""
    return {
        "frame_count": len(senf.frames),
        "mention_count": len(senf.mentions),
        "identity_edge_count": len(graph.edges) if graph else 0,
        "negative_identity_edge_count": (
            sum(1 for edge in graph.edges if edge.negative_evidence) if graph else 0
        ),
        "merge_count": graph.merge_count if graph else 0,
        "rewritten_atom_count": rewritten,
        "exemplar_count": sum(len(scores) for scores in senf.exemplar_scores.values()),
        "bridge_atom_count": bridge_count,
        "weave_distortion": (
            round(weave_result.distortion, 4) if weave_result else None
        ),
        "weave_pair_count": len(weave_result.pairs) if weave_result else 0,
        "weave_total_cost": round(weave_result.total_cost, 4) if weave_result else None,
    }


class CanonicalSENFPLNParser(CanonicalPLNParser):
    """Canonical PLN parser extended with SENF identity, exemplars, and weaves."""

    def __init__(self):
        super().__init__()
        cfg = get_settings()
        self._threshold = cfg.senf_identity_threshold
        self._context_top_k = cfg.senf_context_top_k
        self._max_frames = cfg.senf_session_max_frames
        self._use_vector_context = cfg.senf_use_vector_context
        self._exemplar_enabled = cfg.senf_exemplar_enabled
        self._emit_bridge_atoms = cfg.senf_emit_bridge_atoms
        self._transport_truth_values = cfg.senf_transport_truth_values
        self._weave_top_k = cfg.senf_weave_top_k
        self._source_grounding_weight = cfg.senf_source_grounding_weight
        self._role_compat_weight = cfg.senf_role_compat_weight
        self._distortion_weight = cfg.senf_distortion_weight
        self._identity_support_weight = cfg.senf_identity_support_weight
        self._exemplar_coherence_weight = cfg.senf_exemplar_coherence_weight
        self._conflict_weight = cfg.senf_conflict_weight
        self._transport_cost_weight = cfg.senf_transport_cost_weight
        self._vector_store = None
        self.reset()

    def reset(self) -> None:
        self._session: List[SENF] = []
        self._sentence_counter = 0
        self._session_nonce = uuid.uuid4().hex[:8]
        self._weave: Optional[WeaveResult] = None
        self._weaves: tuple[WeaveResult, ...] = ()
        self._telemetry: Optional[dict] = None
        self._latest_ingest_senf: Optional[SENF] = None

    def senf_telemetry(self) -> Optional[dict]:
        """Counts from the most recent hook call, or None if it never ran.

        Probed by the service through `hasattr`, so the base parser reporting
        nothing needs no coordination here.
        """
        return dict(self._telemetry) if self._telemetry else None

    def storage_metadata(self) -> Optional[dict]:
        if self._latest_ingest_senf is None:
            return None
        return {SENF_PAYLOAD_KEY: senf_to_payload(self._latest_ingest_senf)}

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
            self._weaves = ()
        else:
            self._latest_ingest_senf = None

        if not statements and not queries:
            return statements, queries

        try:
            text = " ".join(texts)
            self._sentence_counter += 1
            sentence_id = f"{self._session_nonce}:s{self._sentence_counter}"
            senf = extract_senf(sentence_id, text, statements + queries)
            if self._exemplar_enabled:
                score_exemplars(senf)
            if senf.is_empty:
                self._telemetry = _telemetry(senf)
                return statements, queries

            prior = self._prior_senfs(text)
            if self._exemplar_enabled:
                for prior_senf in prior:
                    if not prior_senf.exemplar_scores:
                        score_exemplars(prior_senf)
            graph = resolve_identity(prior + [senf], threshold=self._threshold)
            if is_query:
                self._weaves = build_weaves(
                    senf,
                    prior,
                    k=self._weave_top_k,
                    resolve=graph.resolve,
                    identity_graph=graph,
                )
                self._weave = self._weaves[0] if self._weaves else None
            # Only a question owns a weave; on ingest self._weave still holds the
            # previous question's, which is not this call's telemetry.
            reported = self._weave if is_query else None
            if not graph.representatives:
                self._telemetry = _telemetry(senf, graph, reported)
                self._remember(senf, is_query)
                if not is_query:
                    self._latest_ingest_senf = senf
                return statements, queries

            rewritten_statements = [
                self._rewrite_statement(stmt, graph) for stmt in statements
            ]
            rewritten_queries = [self._rewrite(query, graph) for query in queries]
            bridges = (
                identity_bridge_atoms(graph.merged, threshold=self._threshold)
                if self._emit_bridge_atoms and not is_query
                else []
            )
            rewritten_statements.extend(bridges)
            changed = sum(
                1
                for before, after in zip(
                    statements + queries, rewritten_statements + rewritten_queries
                )
                if before != after
            )
            self._telemetry = _telemetry(
                senf, graph, reported, changed, len(bridges)
            )
            self._remember(senf, is_query)
            if not is_query:
                self._latest_ingest_senf = senf
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
        identity_support: dict[str, float] = {}
        exemplar_coherence: dict[str, float] = {}
        conflict_penalty: dict[str, float] = {}
        for mapping in self._weave.entity_maps:
            identity_support[mapping.target_symbol] = max(
                identity_support.get(mapping.target_symbol, 0.0),
                1.0 - mapping.cost,
            )
            exemplar_coherence[mapping.target_symbol] = max(
                exemplar_coherence.get(mapping.target_symbol, 0.0),
                1.0 - self._weave.exemplar_cost,
            )
            conflict_penalty[mapping.target_symbol] = max(
                conflict_penalty.get(mapping.target_symbol, 0.0),
                self._weave.conflict_cost,
            )
        return query_scoring.SENFSignals(
            grounded_symbols=self._weave.grounded_symbols,
            role_signatures=self._weave.role_signatures,
            distortion=self._weave.distortion,
            source_grounding_weight=self._source_grounding_weight,
            role_compat_weight=self._role_compat_weight,
            distortion_weight=self._distortion_weight,
            identity_support=identity_support,
            exemplar_coherence=exemplar_coherence,
            conflict_penalty=conflict_penalty,
            transport_cost=self._weave.total_cost,
            identity_support_weight=self._identity_support_weight,
            exemplar_coherence_weight=self._exemplar_coherence_weight,
            conflict_weight=self._conflict_weight,
            transport_cost_weight=self._transport_cost_weight,
        )

    def _prior_senfs(self, text: str) -> List[SENF]:
        """Session SENF first, then anything the vector store recalls for this text."""
        prior = list(self._session)
        seen = {senf.senf_id for senf in prior}
        for blob in self._retrieve_senf_blobs(text):
            recalled = senf_from_payload(blob)
            if recalled and not recalled.is_empty and recalled.senf_id not in seen:
                seen.add(recalled.senf_id)
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

    def _rewrite_statement(self, statement: str, graph) -> str:
        changed_costs: list[float] = []

        def replace(match: re.Match) -> str:
            token = match.group(0)
            if token[0] in "$?":
                return token
            replacement = graph.resolve(token)
            if replacement != token:
                changed_costs.append(graph.transport_cost(token, replacement))
            return replacement

        rewritten = _TOKEN_RE.sub(replace, statement)
        if not changed_costs or not self._transport_truth_values:
            return rewritten
        cost = max(changed_costs)

        def degrade(match: re.Match) -> str:
            tv = transport_truth(float(match.group(1)), float(match.group(2)), cost)
            return f"(STV {tv.strength} {tv.weight})"

        return _STV_RE.sub(degrade, rewritten, count=1)
