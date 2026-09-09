import logging
import re
import uuid
from typing import List, Optional

from config import get_settings
from core import query_scoring
from core.parser import ParseResult
from core.senf.bridge import predicate_bridge_atoms
from core.senf.exemplars import score_exemplars
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.types import (
    EntityRef,
    FrameRef,
    KindRef,
    SENF,
    SENFFrame,
    SENF_PAYLOAD_KEY,
    ValueRef,
    senf_from_payload,
    senf_to_payload,
)
from core.senf.weave import WeaveResult, build_weaves
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)

_SENF_SKIP_HEADS = frozenset(
    {"SimilarityLink", "ContextLink", "PredicateBridge", "PredicateSimilarity"}
)


def _is_semantic_atom(atom: str) -> bool:
    match = re.search(r"\(\s*([A-Za-z][A-Za-z0-9_]*)\b", atom)
    return bool(match and match.group(1) not in _SENF_SKIP_HEADS)


def _senf_inputs(text: str, statements: list[str], queries: list[str]) -> list[str]:
    """Admit grounded facts and query-relevant atoms, excluding generated links."""
    kept: list[str] = []
    for index, atom in enumerate(statements):
        if not _is_semantic_atom(atom):
            continue
        probe = extract_senf(f"admit:{index}", text, [atom])
        if any(mention.char_span is not None for mention in probe.mentions):
            kept.append(atom)
    kept.extend(atom for atom in queries if _is_semantic_atom(atom))
    return kept


def _telemetry(
    senf: SENF,
    graph=None,
    weave_result: Optional[WeaveResult] = None,
    query_rewritten: int = 0,
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
        "query_rewritten_atom_count": query_rewritten,
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
        self._pending_ingest: Optional[tuple[str, str]] = None
        self._query_source_heads: frozenset[str] = frozenset()

    def senf_telemetry(self) -> Optional[dict]:
        """Counts from the most recent hook call, or None if it never ran.

        Probed by the service through `hasattr`, so the base parser reporting
        nothing needs no coordination here.
        """
        return dict(self._telemetry) if self._telemetry else None

    def _parse_many_with_mode(
        self, texts: List[str], context: List[str], is_query: bool
    ) -> ParseResult:
        self._pending_ingest = None
        result = super()._parse_many_with_mode(texts, context, is_query)
        result.diagnostics = self.senf_telemetry()
        if not is_query:
            result.parser_state = self._pending_ingest
        self._pending_ingest = None
        return result

    def prepare_ingest(
        self, parse_result: ParseResult, accepted: List[str]
    ) -> Optional[dict]:
        pending = parse_result.parser_state
        parse_result.parser_state = None
        if not pending or not accepted:
            parse_result.metadata = {}
            return None
        sentence_id, text = pending
        semantic_atoms = [atom for atom in accepted if _is_semantic_atom(atom)]
        senf = extract_senf(sentence_id, text, semantic_atoms)
        if self._exemplar_enabled:
            score_exemplars(senf)
        if senf.is_empty:
            parse_result.metadata = {}
            return None
        parse_result.parser_state = senf
        parse_result.metadata = {SENF_PAYLOAD_KEY: senf_to_payload(senf)}
        return parse_result.metadata

    def commit_ingest(self, parse_result: ParseResult) -> None:
        senf = parse_result.parser_state
        parse_result.parser_state = None
        if isinstance(senf, SENF):
            self._remember(senf, is_query=False)

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
            self._pending_ingest = None

        if not statements and not queries:
            return statements, queries

        try:
            text = " ".join(texts)
            self._sentence_counter += 1
            sentence_id = f"{self._session_nonce}:s{self._sentence_counter}"
            senf = extract_senf(
                sentence_id, text, _senf_inputs(text, statements, queries)
            )
            if not is_query:
                self._pending_ingest = (sentence_id, text)
            if self._exemplar_enabled:
                score_exemplars(senf)
            if senf.is_empty:
                self._telemetry = _telemetry(senf)
                return statements, queries

            prior = self._prior_senfs(text, is_query=is_query)
            if is_query:
                self._query_source_heads = frozenset(
                    frame.predicate_head for item in prior for frame in item.frames
                )
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
                    identity_graph=graph,
                )
                self._weave = self._weaves[0] if self._weaves else None
            # Only a question owns a weave; on ingest self._weave still holds the
            # previous question's, which is not this call's telemetry.
            reported = self._weave if is_query else None
            rewritten_queries = self._render_queries(queries, senf, graph) if is_query else queries
            bridges = []
            if self._emit_bridge_atoms and is_query and self._weave is not None:
                bridges.extend(predicate_bridge_atoms(self._weave))
            query_statements = statements + bridges
            changed = sum(
                1
                for before, after in zip(queries, rewritten_queries)
                if before != after
            )
            self._telemetry = _telemetry(
                senf, graph, reported, changed, len(bridges)
            )
            return query_statements, rewritten_queries
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

    def _plan_queries(
        self,
        question: str,
        queries: List[str],
        statements: List[str],
        context: List[str],
    ) -> List[str]:
        planned = super()._plan_queries(question, queries, statements, context)
        if not planned:
            return planned

        facts, conclusions = self._collect_available_signatures(statements, context)
        available = {(sig["head"], sig["arity"]) for sig in facts + conclusions}
        predicate_sources: dict[str, tuple[float, str]] = {}
        for weave in self._weaves:
            for mapping in weave.predicate_maps:
                current = predicate_sources.get(mapping.query_head)
                if current is None or mapping.cost < current[0]:
                    predicate_sources[mapping.query_head] = (
                        mapping.cost,
                        mapping.source_head,
                    )
        source_heads = self._query_source_heads or frozenset(
            source_head for _, source_head in predicate_sources.values()
        )

        constrained: list[str] = []
        for candidate in planned:
            parsed = self._parse_query_signature(candidate)
            if parsed is None:
                continue
            if (
                parsed["head"] in source_heads
                and (parsed["head"], parsed["arity"]) in available
            ):
                constrained.append(candidate)
                continue
            mapped = predicate_sources.get(parsed["head"])
            if (
                mapped
                and mapped[1] in source_heads
                and (mapped[1], parsed["arity"]) in available
            ):
                rewritten = dict(parsed)
                rewritten["head"] = mapped[1]
                constrained.append(self._signature_to_query(rewritten))
        return self._dedupe_preserve_order(constrained + planned)

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

    def _prior_senfs(self, text: str, is_query: bool = False) -> List[SENF]:
        """Use the current query window plus semantically relevant recalled SENFs."""
        prior = list(self._session)
        seen = {senf.senf_id for senf in prior}
        for blob in self._retrieve_senf_blobs(text):
            recalled = senf_from_payload(blob)
            if (
                recalled
                and not recalled.is_empty
                and recalled.senf_id not in seen
                and (not is_query or self._senf_overlaps_question(recalled, text))
            ):
                seen.add(recalled.senf_id)
                prior.append(recalled)
        return prior

    def _senf_overlaps_question(self, senf: SENF, question: str) -> bool:
        question_symbols = set(self._question_symbols(question))
        if question_symbols.intersection(senf.symbols()):
            return True
        source_text = " ".join(
            frame.source_text for frame in senf.frames if frame.source_text
        )
        source_symbols = set(self._question_symbols(source_text))
        return bool(question_symbols.intersection(source_symbols))

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
    def _render_queries(queries: list[str], query_senf: SENF, graph) -> list[str]:
        occurrences: dict[str, int] = {}
        rendered = []
        for query in queries:
            parts = CanonicalSENFPLNParser._top_level_parts(query)
            atom_id = parts[1] if len(parts) >= 2 else ""
            occurrence = occurrences.get(atom_id, 0)
            rendered.append(CanonicalSENFPLNParser._render_query(
                query, query_senf, graph, occurrence
            ))
            occurrences[atom_id] = occurrence + 1
        return rendered

    @staticmethod
    def _render_query(
        expression: str, query_senf: SENF, graph, occurrence_index: int = 0
    ) -> str:
        parts = CanonicalSENFPLNParser._top_level_parts(expression)
        if len(parts) < 4 or parts[0] != ":":
            return expression
        atom_id = parts[1]
        candidates = [
            frame for frame in query_senf.frames if frame.source_atom_id == atom_id
        ]
        if any(frame.clause_role != "fact" for frame in candidates):
            return expression
        referenced = {
            role.filler.frame_id
            for frame in candidates
            for role in frame.roles
            if isinstance(role.filler, FrameRef)
        }
        roots = [item for item in candidates if item.frame_id not in referenced]
        if occurrence_index >= len(roots):
            return expression
        frame = roots[occurrence_index]
        frames = {item.frame_id: item for item in query_senf.frames}
        body = CanonicalSENFPLNParser._render_frame(frame, query_senf, graph, frames)
        return f"(: {atom_id} {body} {' '.join(parts[3:])})"

    @staticmethod
    def _render_frame(
        frame: SENFFrame, query_senf: SENF, graph, frames: dict[str, SENFFrame],
        include_polarity: bool = True,
    ) -> str:
        rendered = []
        for role in sorted(frame.roles, key=lambda item: item.position):
            filler = role.filler
            if isinstance(filler, EntityRef):
                representative = graph.resolve_entity(filler.mention_id)
                entity = query_senf.entity(filler.entity_id)
                rendered.append(
                    graph.entity_symbols.get(
                        representative,
                        entity.canonical_symbol if entity else filler.entity_id,
                    )
                )
            elif isinstance(filler, KindRef):
                rendered.append(filler.canonical_symbol)
            elif isinstance(filler, ValueRef):
                rendered.append(filler.value)
            elif isinstance(filler, FrameRef) and filler.frame_id in frames:
                rendered.append(CanonicalSENFPLNParser._render_frame(
                    frames[filler.frame_id], query_senf, graph, frames
                ))
        args = f" {' '.join(rendered)}" if rendered else ""
        body = f"({frame.predicate_head}{args})"
        return f"(Not {body})" if include_polarity and not frame.polarity else body

    @staticmethod
    def _top_level_parts(expression: str) -> list[str]:
        text = expression.strip()
        if len(text) < 2 or text[0] != "(" or text[-1] != ")":
            return []
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in text[1:-1]:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return []
            if char.isspace() and depth == 0:
                if current:
                    parts.append("".join(current))
                    current = []
            else:
                current.append(char)
        if depth != 0:
            return []
        if current:
            parts.append("".join(current))
        return parts
