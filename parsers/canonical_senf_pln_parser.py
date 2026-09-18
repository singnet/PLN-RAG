import copy
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional

from config import get_settings
from core import query_scoring
from core.parser import ParseResult
from core.senf.bridge import executable_bridge_atoms
from core.senf.exemplars import score_exemplars
from core.senf.feature_binding import apply_feature_bindings, bind_features
from core.senf.feature_provider import create_feature_provider, provide_features
from core.senf.features import CoreferenceEvidence, ExactSpan, FeatureBatch, SpanFeature
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.types import (
    ACTUAL_BRANCH_ID,
    Context,
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
from core.senf.temporal import (
    BranchingContextTree,
    TransportDecision,
    compile_branch_theory,
    unwrap_contextual_query,
    wrap_contextual_statement,
)
from core.senf.weave import WeaveResult, build_weaves
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)

_SENF_SKIP_HEADS = frozenset(
    {"SimilarityLink", "ContextLink", "PredicateBridge", "PredicateSimilarity"}
)


@dataclass(frozen=True)
class _CandidatePlan:
    query: str
    senf: SENF
    graph: object
    weave: Optional[WeaveResult]
    source: Optional[SENF]
    score: query_scoring.CandidateScore
    adapters: tuple[str, ...] = ()
    query_context: Optional[Context] = None
    temporal_decisions: tuple[TransportDecision, ...] = ()
    weaves: tuple[WeaveResult, ...] = ()


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


def _bounded(items, limit: int) -> tuple[list, dict]:
    values = list(items)
    emitted = values[:limit]
    return emitted, {
        "total": len(values),
        "emitted": len(emitted),
        "omitted": len(values) - len(emitted),
    }


def _weave_summary(
    result: WeaveResult, max_items: int, max_evidence: int
) -> dict:
    """Serialize one weave under response-specific cardinality bounds."""
    costs = {
        name: round(getattr(result, f"{name}_cost"), 4)
        for name in (
            "structural", "exemplar", "identity", "conflict", "time",
            "location", "modality", "unmatched", "distortion", "branch",
            "temporal_decay", "persistence",
        )
    }
    truncation = {}

    def collection(name: str, items, serializer=lambda item: item) -> list:
        emitted, report = _bounded(items, max_items)
        truncation[name] = report
        return [serializer(item) for item in emitted]

    def pair(item) -> dict:
        payload = asdict(item)
        evidence, report = _bounded(item.evidence, max_evidence)
        payload["evidence"] = evidence
        payload["evidence_truncation"] = report
        return payload

    def alignment(item) -> dict:
        evidence, report = _bounded(item.evidence, max_evidence)
        return {
            "query_frame_id": item.query_frame_id,
            "source_id": item.source_frame.source_id,
            "source_frame_id": item.source_frame.frame_id,
            "score": item.score,
            "costs": asdict(item.costs),
            "evidence": evidence,
            "evidence_truncation": report,
        }

    polish_status = "not_applicable"
    if result.polish is not None:
        polish_status = (
            "fallback" if result.fallback_reason
            else "converged" if result.polish_converged
            else "not_converged"
        )
    summary = {
        "guard": result.guard,
        "aligned": result.aligned,
        "distortion": result.distortion,
        "total_cost": result.total_cost,
        "costs": costs,
        "matched_pair_count": len(result.pairs),
        "pairs": collection("pairs", result.pairs, pair),
        "rejected_pairs": collection("rejected_pairs", result.rejected_pairs, pair),
        "entity_maps": collection("entity_maps", result.entity_maps, asdict),
        "predicate_maps": collection("predicate_maps", result.predicate_maps, asdict),
        "kind_maps": collection("kind_maps", result.kind_maps, lambda item: {
            "source_kind": item[0], "query_kind": item[1]
        }),
        "exemplar_maps": collection("exemplar_maps", result.exemplar_maps, lambda item: {
            "source_exemplar": item[0], "query_exemplar": item[1]
        }),
        "role_maps": collection("role_maps", result.role_maps, lambda item: {
            "source_role": item[0], "query_role": item[1]
        }),
        "alignments": collection("alignments", result.alignments, alignment),
        "grounded_symbols": collection(
            "grounded_symbols", sorted(result.grounded_symbols)
        ),
        "grounded_entity_ids": collection(
            "grounded_entity_ids", sorted(result.grounded_entity_ids)
        ),
        "role_signatures": collection(
            "role_signatures", sorted(result.role_signatures), list
        ),
        "residuals": list(result.residuals) if result.residuals is not None else None,
        "global_coupling_objective": result.global_coupling_objective,
        "coupling_entropy": result.coupling_entropy,
        "matched_soft_mass": result.matched_soft_mass,
        "unmatched_soft_mass": result.unmatched_soft_mass,
        "polish_iterations": result.polish_iterations,
        "polish_converged": result.polish_converged,
        "polish_status": polish_status,
        "fallback_reason": result.fallback_reason,
        "rejection_reason": result.rejection_reason,
        "polish": asdict(result.polish) if result.polish else None,
        "transport_decisions": collection(
            "transport_decisions", result.transport_decisions, asdict
        ),
    }
    summary["truncation"] = truncation
    return summary


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
        configured_work = max(1, int(cfg.senf_query_max_candidate_work))
        execution_limit = int(cfg.query_candidate_max_tries or 0)
        self._candidate_limit = execution_limit if execution_limit > 0 else configured_work
        self._max_priors = max(1, int(cfg.senf_query_max_priors))
        self._max_source_frames = max(1, int(cfg.senf_query_max_source_frames))
        self._max_mentions = max(1, int(cfg.senf_query_max_mentions))
        self._candidate_work_limit = max(
            self._candidate_limit, configured_work
        )
        self._source_grounding_weight = cfg.senf_source_grounding_weight
        self._role_compat_weight = cfg.senf_role_compat_weight
        self._distortion_weight = cfg.senf_distortion_weight
        self._identity_support_weight = cfg.senf_identity_support_weight
        self._exemplar_coherence_weight = cfg.senf_exemplar_coherence_weight
        self._conflict_weight = cfg.senf_conflict_weight
        self._transport_cost_weight = cfg.senf_transport_cost_weight
        self._matched_soft_mass_weight = cfg.senf_matched_soft_mass_weight
        self._global_residual_ratio_weight = cfg.senf_global_residual_ratio_weight
        self._alignment_confidence_weight = cfg.senf_alignment_confidence_weight
        self._sinkhorn_tolerance = cfg.senf_weave_sinkhorn_tolerance
        self._diagnostics_max_weaves = cfg.senf_diagnostics_max_weaves
        self._diagnostics_max_items = cfg.senf_diagnostics_max_items
        self._diagnostics_max_evidence = cfg.senf_diagnostics_max_evidence
        self._counterfactual_enabled = cfg.senf_counterfactual_enabled
        self._branch_max_nodes = max(1, int(cfg.senf_branch_max_nodes))
        self._branch_max_depth = max(1, int(cfg.senf_branch_max_depth))
        self._branch_max_theory_statements = max(
            1, int(cfg.senf_branch_max_theory_statements)
        )
        self._temporal_decay_rate = max(0.0, float(cfg.senf_temporal_decay_rate))
        self._feature_provider = create_feature_provider(cfg)
        self._feature_provider_name = cfg.senf_feature_provider
        self._feature_diagnostics_limit = cfg.senf_feature_max_features
        self._vector_store = None
        self.reset()

    def reset(self) -> None:
        self._session: List[SENF] = []
        self._sentence_counter = 0
        self._session_nonce = uuid.uuid4().hex[:8]
        self._weave: Optional[WeaveResult] = None
        self._weaves: tuple[WeaveResult, ...] = ()
        self._telemetry: Optional[dict] = None
        self._pending_ingest = None
        self._query_source_heads: frozenset[str] = frozenset()
        self._query_prior: tuple[SENF, ...] = ()
        self._query_sentence_id = ""
        self._candidate_plans: tuple[_CandidatePlan, ...] = ()
        self._query_context = Context("query:u0")
        self._query_contexts: dict[str, Context] = {}
        self._query_feature_batch = FeatureBatch.empty()
        self._feature_diagnostics: tuple[object, ...] = ()

    def set_generation_backend(self, backend) -> None:
        super().set_generation_backend(backend)
        installer = getattr(self._feature_provider, "set_backend", None)
        if callable(installer):
            installer(backend)

    def senf_telemetry(self) -> Optional[dict]:
        """Counts from the most recent hook call, or None if it never ran.

        Probed by the service through `hasattr`, so the base parser reporting
        nothing needs no coordination here.
        """
        return copy.deepcopy(self._telemetry) if self._telemetry else None

    def _parse_many_with_mode(
        self, texts: List[str], context: List[str], is_query: bool
    ) -> ParseResult:
        self._pending_ingest = None
        result = super()._parse_many_with_mode(texts, context, is_query)
        result.diagnostics = self.senf_telemetry()
        if is_query and self._candidate_plans:
            result.candidate_trusted_transient_statements = [
                list(plan.adapters) for plan in self._candidate_plans
            ]
        elif not is_query:
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
        sentence_id, text, feature_batch = pending
        semantic_atoms = [atom for atom in accepted if _is_semantic_atom(atom)]
        senf = extract_senf(sentence_id, text, semantic_atoms)
        self._apply_features(senf, text, feature_batch)
        if self._exemplar_enabled:
            score_exemplars(senf)
        if senf.is_empty:
            parse_result.metadata = {}
            return None
        payload = senf_to_payload(senf)
        if senf_from_payload(payload) is None:
            parse_result.metadata = {}
            parse_result.parser_state = senf
            return parse_result.metadata
        parse_result.parser_state = senf
        parse_result.metadata = {
            SENF_PAYLOAD_KEY: payload,
            "senf_branch_ids": [branch.branch_id for branch in senf.branches],
        }
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
            self._query_prior = ()
            self._query_sentence_id = ""
            self._candidate_plans = ()
            self._query_context = Context("query:u0")
            self._query_contexts = {}
            self._query_feature_batch = FeatureBatch.empty()
        else:
            self._pending_ingest = None

        if not statements and not queries:
            return statements, queries

        try:
            text = " ".join(texts)
            feature_validation = provide_features(text, self._feature_provider)
            self._feature_diagnostics = feature_validation.rejected
            self._sentence_counter += 1
            sentence_id = f"{self._session_nonce}:s{self._sentence_counter}"
            senf = extract_senf(
                sentence_id, text, _senf_inputs(text, statements, queries)
            )
            if not is_query:
                self._pending_ingest = (sentence_id, text, feature_validation.accepted)
            else:
                self._query_feature_batch = feature_validation.accepted
                self._apply_features(senf, text, feature_validation.accepted)
            if self._exemplar_enabled:
                score_exemplars(senf)
            if senf.is_empty:
                self._telemetry = _telemetry(senf)
                self._telemetry.update({
                    "feature_provider": self._feature_provider_name,
                    "applied_feature_count": 0,
                    "provider_coreference_count": 0,
                    "feature_rejections": [
                        asdict(item)
                        for item in self._feature_diagnostics[:self._feature_diagnostics_limit]
                    ],
                })
                return statements, queries

            if is_query and self._counterfactual_enabled:
                explicit_contexts = {
                    frame.context
                    for frame in senf.frames
                    if frame.context is not None
                    and (
                        frame.context.branch_id != ACTUAL_BRANCH_ID
                        or frame.context.validity_interval_id is not None
                    )
                }
                if len(explicit_contexts) > 1:
                    self._telemetry = {
                        **_telemetry(senf),
                        "stage7_rejection": "query spans multiple contexts",
                    }
                    return statements, []
                if explicit_contexts:
                    self._query_context = next(iter(explicit_contexts))

            prior = self._prior_senfs(text, is_query=is_query)
            if is_query:
                self._query_source_heads = frozenset(
                    frame.predicate_head for item in prior for frame in item.frames
                )
                self._query_prior = tuple(prior)
                self._query_sentence_id = sentence_id
            if self._exemplar_enabled:
                for prior_senf in prior:
                    if not prior_senf.exemplar_scores:
                        score_exemplars(prior_senf)
            graph = resolve_identity(prior + [senf], threshold=self._threshold)
            # Only a question owns a weave; on ingest self._weave still holds the
            # previous question's, which is not this call's telemetry.
            reported = self._weave if is_query else None
            # Candidate binding is planned later, after the base parser has derived
            # all fallbacks. The canonical query remains untouched here.
            rewritten_queries = queries
            query_statements = statements
            changed = 0
            self._telemetry = _telemetry(
                senf, graph, reported, changed, 0
            )
            self._telemetry.update({
                "feature_provider": self._feature_provider_name,
                "applied_feature_count": len(senf.applied_mention_features),
                "provider_coreference_count": len(senf.coreference_evidence),
                "feature_rejections": [
                    asdict(item)
                    for item in self._feature_diagnostics[:self._feature_diagnostics_limit]
                ],
            })
            return query_statements, rewritten_queries
        except Exception as exc:
            if getattr(exc, "fail_closed", False):
                raise
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
        contextual = [unwrap_contextual_query(query) for query in queries]
        stage7_query = self._counterfactual_enabled and any(
            item.context.branch_id != ACTUAL_BRANCH_ID
            or item.context.validity_interval_id is not None
            for item in contextual
        )
        planner_queries = [item.query for item in contextual] if stage7_query else queries
        canonical = CanonicalPLNParser._plan_queries(
            self, question, planner_queries, [], context
        )
        if stage7_query:
            admitted = set(planner_queries)
            canonical = [query for query in canonical if query in admitted]
            canonical = canonical or self._dedupe_preserve_order(planner_queries)
            self._query_contexts = {
                item.query: item.context for item in contextual
            }
            self._query_context = next(
                item.context for item in contextual
                if item.context.branch_id != ACTUAL_BRANCH_ID
                or item.context.validity_interval_id is not None
            )
            branch_ids = {
                branch.branch_id for senf in self._query_prior for branch in senf.branches
            }
            interval_ids = {
                interval.interval_id
                for senf in self._query_prior
                for interval in senf.validity_intervals
            }
            if any(
                item.context.branch_id not in branch_ids
                or item.context.validity_interval_id is not None
                and item.context.validity_interval_id not in interval_ids
                for item in contextual
            ):
                if self._telemetry is not None:
                    self._telemetry["stage7_rejection"] = (
                        "query references unknown branch or validity interval"
                    )
                return []
        if not canonical:
            return canonical

        if not self._query_prior:
            # With no SENF evidence there is nothing to transport. Preserve the
            # base parser's candidates and ordering byte-for-byte.
            return canonical

        facts, conclusions = self._collect_available_signatures([], context)
        is_yes_no = self._is_yes_no_question(question)
        canonical_kept = canonical[: self._candidate_limit]
        variant_slots = self._candidate_limit - len(canonical_kept)
        generated: list[str] = []
        contexts: dict[str, tuple[SENF, object, tuple[WeaveResult, ...]]] = {}
        work = 0
        for candidate in canonical_kept:
            if work >= self._candidate_work_limit or len(generated) >= variant_slots:
                break
            candidate_context = self._candidate_context(candidate, question)
            contexts[candidate] = candidate_context
            work += 1
            for variant in self._candidate_variants(candidate, *candidate_context):
                if variant != candidate and variant not in generated:
                    generated.append(variant)
                    self._query_contexts[variant] = self._query_contexts.get(
                        candidate, self._query_context
                    )
                if len(generated) >= variant_slots:
                    break

        plans: list[_CandidatePlan] = []
        for candidate in generated:
            if work >= self._candidate_work_limit:
                break
            plans.append(self._build_candidate_plan(
                candidate, question, facts, conclusions, is_yes_no
            ))
            work += 1
        plans.sort(key=lambda item: (
            -(item.score.total if item.score.total is not None else -10**9),
            item.weave.total_cost if item.weave else float("inf"),
            item.query,
        ))

        canonical_plans = [
            self._build_candidate_plan(
                candidate,
                question,
                facts,
                conclusions,
                is_yes_no,
                candidate_context=contexts.get(candidate),
            )
            for candidate in canonical_kept
        ]
        selected = canonical_plans if self._emit_bridge_atoms else sorted(
            plans[:variant_slots] + canonical_plans,
            key=lambda item: (
                -(item.score.total if item.score.total is not None else -10**9),
                item.weave.total_cost if item.weave else float("inf"),
                item.query,
            ),
        )
        deduped: list[_CandidatePlan] = []
        seen: set[str] = set()
        for plan in selected:
            if plan.query not in seen:
                seen.add(plan.query)
                deduped.append(plan)
        self._candidate_plans = tuple(deduped)
        self._weaves = tuple(plan.weave for plan in deduped if plan.weave is not None)
        self._weave = self._weaves[0] if self._weaves else None

        adapters = self._dedupe_preserve_order(
            [adapter for plan in deduped for adapter in plan.adapters]
        )
        if self._telemetry is not None:
            self._telemetry.update({
                "candidate_count": len(deduped),
                "candidate_score_breakdown": [
                    {"query": plan.query, **plan.score.as_dict()} for plan in deduped
                ],
                "bridge_atom_count": sum(
                    "senf_adapter" in adapter for adapter in adapters
                ),
                "counterfactual_theory_statement_count": sum(
                    adapter.startswith("(: stage7_") for adapter in adapters
                ),
                "branch_count": len({
                    branch.branch_id
                    for senf in self._query_prior
                    for branch in senf.branches
                }),
                "candidate_temporal_plans": [
                    {
                        "query": plan.query,
                        "branch_id": (
                            plan.query_context.branch_id
                            if plan.query_context else ACTUAL_BRANCH_ID
                        ),
                        "validity_interval_id": (
                            plan.query_context.validity_interval_id
                            if plan.query_context else None
                        ),
                        "decisions": [asdict(item) for item in plan.temporal_decisions],
                    }
                    for plan in deduped
                ],
                "query_rewritten_atom_count": sum(
                    plan.query not in canonical for plan in deduped
                ),
                "weave_distortion": round(self._weave.distortion, 4) if self._weave else None,
                "weave_pair_count": len(self._weave.pairs) if self._weave else 0,
                "weave_total_cost": round(self._weave.total_cost, 4) if self._weave else None,
                "weave_summaries": [],
            })
            diagnostic_weaves = deduped[0].weaves if deduped else ()
            emitted_weaves, weave_truncation = _bounded(
                diagnostic_weaves, self._diagnostics_max_weaves
            )
            self._telemetry["weave_summaries"] = [
                _weave_summary(
                    item,
                    self._diagnostics_max_items,
                    self._diagnostics_max_evidence,
                )
                for item in emitted_weaves
            ]
            self._telemetry["weave_summaries_truncation"] = weave_truncation
        return [plan.query for plan in deduped]

    def _candidate_context(
        self, candidate: str, question: str
    ) -> tuple[SENF, object, tuple[WeaveResult, ...]]:
        query_context = self._query_contexts.get(candidate, self._query_context)
        candidate_atom = wrap_contextual_statement(candidate, query_context)
        candidate_senf = extract_senf(
            self._query_sentence_id or f"{self._session_nonce}:s{self._sentence_counter}",
            question,
            [candidate_atom],
        )
        self._apply_features(candidate_senf, question, self._query_feature_batch)
        tree = BranchingContextTree.from_senfs(
            self._query_prior,
            max_nodes=self._branch_max_nodes,
            max_depth=self._branch_max_depth,
        )
        branch_definitions = {
            branch.branch_id: branch for branch in tree.branches
        }
        interval_definitions = {
            interval.interval_id: interval
            for senf in self._query_prior
            for interval in senf.validity_intervals
        }
        if query_context.branch_id not in branch_definitions:
            raise ValueError("query references an unknown branch")
        candidate_senf.branches = list(branch_definitions.values())
        candidate_senf.validity_intervals = list(interval_definitions.values())
        if self._exemplar_enabled:
            score_exemplars(candidate_senf)
        graph = resolve_identity(
            list(self._query_prior) + [candidate_senf], threshold=self._threshold
        )
        source_weaves = build_weaves(
            candidate_senf,
            self._query_prior,
            k=self._weave_top_k,
            identity_graph=graph,
        )
        weaves = tuple(sorted(source_weaves, key=lambda item: (
            0 if item.aligned else 1,
            item.total_cost,
            item.guard,
        ))[: self._weave_top_k])
        return candidate_senf, graph, weaves

    def _candidate_variants(
        self,
        candidate: str,
        query_senf: SENF,
        graph,
        weaves: tuple[WeaveResult, ...],
    ) -> list[str]:
        # When executable adapters are enabled, keep the canonical target so the
        # proof consumes the cost-degraded bridge instead of bypassing it through
        # a direct source-symbol query.
        if self._emit_bridge_atoms:
            return []
        parsed = self._parse_query_signature(candidate)
        if parsed is None:
            return []
        variants: list[str] = []
        query_mentions = {mention.mention_id: mention for mention in query_senf.mentions}
        accepted_edges = {frozenset(edge.mention_ids) for edge in graph.merged}
        for edge in graph.merged:
            if edge.left.mention_id in query_mentions:
                query_mention, source_mention = edge.left, edge.right
            elif edge.right.mention_id in query_mentions:
                query_mention, source_mention = edge.right, edge.left
            else:
                continue
            if source_mention.mention_id in query_mentions:
                continue
            positions = {
                role.position
                for frame in query_senf.frames
                for role in frame.roles
                if isinstance(role.filler, EntityRef)
                and role.filler.mention_id == query_mention.mention_id
            }
            if positions:
                changed = dict(parsed)
                changed["args"] = [
                    source_mention.canonical_symbol if index in positions else arg
                    for index, arg in enumerate(parsed["args"])
                ]
                rendered = self._signature_to_query(changed)
                if rendered != candidate:
                    variants.append(rendered)
        for weave in weaves:
            for pair in weave.pairs:
                query_frame = next((
                    frame for frame in query_senf.frames
                    if frame.frame_id == pair.query_frame_id
                ), None)
                if query_frame is None or query_frame.predicate_head != parsed["head"]:
                    continue
                replacements: dict[int, str] = {}
                for mapping in weave.entity_maps:
                    if mapping.source_id and mapping.source_id != pair.source_id:
                        continue
                    if mapping.source_frame_id and (
                        mapping.source_frame_id != pair.source_frame_id
                        or mapping.query_frame_id != pair.query_frame_id
                    ):
                        continue
                    edge = next((
                        edge for edge in graph.edges
                        if {mapping.source_mention_id, mapping.target_mention_id}
                        == set(edge.mention_ids)
                    ), None)
                    if (
                        edge is None
                        or frozenset(edge.mention_ids) not in accepted_edges
                        or mapping.target_mention_id not in query_mentions
                    ):
                        continue
                    for role in query_frame.roles:
                        if (
                            isinstance(role.filler, EntityRef)
                            and role.filler.mention_id == mapping.target_mention_id
                        ):
                            replacements[role.position] = mapping.source_symbol

                source_heads = [
                    mapping.source_head for mapping in weave.predicate_maps
                    if mapping.source_head != mapping.query_head
                    and (not mapping.source_id or mapping.source_id == pair.source_id)
                    and (
                        not mapping.source_frame_id
                        or mapping.source_frame_id == pair.source_frame_id
                        and mapping.query_frame_id == pair.query_frame_id
                    )
                ]
                replacement_sets = [replacements] if replacements else [{}]
                if replacements and source_heads:
                    replacement_sets.append({})
                heads = source_heads + [parsed["head"]]
                for replacements in replacement_sets:
                    for head in heads:
                        changed = dict(parsed)
                        changed["head"] = head
                        changed["args"] = [
                            replacements.get(index, arg)
                            for index, arg in enumerate(parsed["args"])
                        ]
                        rendered = self._signature_to_query(changed)
                        if rendered != candidate:
                            variants.append(rendered)
        return self._dedupe_preserve_order(variants)

    def _build_candidate_plan(
        self,
        candidate: str,
        question: str,
        facts: list[dict],
        conclusions: list[dict],
        is_yes_no: bool,
        candidate_context: Optional[tuple[SENF, object, tuple[WeaveResult, ...]]] = None,
    ) -> _CandidatePlan:
        parsed = self._parse_query_signature(candidate)
        candidate_senf, graph, weaves = candidate_context or self._candidate_context(
            candidate, question
        )
        weave = weaves[0] if weaves else None
        source = self._source_for_weave(weave)
        adapters: tuple[str, ...] = ()
        temporal_decisions: tuple[TransportDecision, ...] = ()
        if self._emit_bridge_atoms:
            for option in weaves:
                option_source = self._source_for_weave(option)
                if option_source is None:
                    continue
                generated = tuple(executable_bridge_atoms(
                    option,
                    self._query_prior,
                    candidate_senf,
                    graph,
                ))
                if generated:
                    weave, source, adapters = option, option_source, generated
                    break
        query_context = self._query_contexts.get(candidate, self._query_context)
        if self._counterfactual_enabled and (
            query_context.branch_id != ACTUAL_BRANCH_ID
            or query_context.validity_interval_id is not None
        ):
            theory, temporal_decisions = compile_branch_theory(
                self._query_prior,
                query_context,
                max_statements=self._branch_max_theory_statements,
                temporal_decay_rate=self._temporal_decay_rate,
                max_branch_nodes=self._branch_max_nodes,
                max_branch_depth=self._branch_max_depth,
            )
            adapters = tuple(self._dedupe_preserve_order(list(adapters) + theory))
        signals = self._senf_signals(weave)
        score = query_scoring.score_query_candidate_breakdown(
            parsed, facts, conclusions, is_yes_no, senf=signals
        ) if parsed else query_scoring.CandidateScore(rejected="invalid_query")
        return _CandidatePlan(
            candidate, candidate_senf, graph, weave, source, score, adapters,
            query_context, temporal_decisions, weaves,
        )

    def _source_for_weave(self, weave: Optional[WeaveResult]) -> Optional[SENF]:
        if weave is None or not weave.pairs:
            return None
        source_ids = {pair.source_id for pair in weave.pairs if pair.source_id}
        if source_ids:
            return next((
                senf for senf in self._query_prior if senf.senf_id in source_ids
            ), None)
        source_frame_ids = {pair.source_frame_id for pair in weave.pairs}
        return next((
            senf for senf in self._query_prior
            if any(frame.frame_id in source_frame_ids for frame in senf.frames)
        ), None)

    def _senf_signals(
        self, weave: Optional[WeaveResult] = None
    ) -> Optional[query_scoring.SENFSignals]:
        selected = weave if weave is not None else self._weave
        if selected is None:
            return None
        identity_support: dict[str, float] = {}
        exemplar_coherence: dict[str, float] = {}
        conflict_penalty: dict[str, float] = {}
        for mapping in selected.entity_maps:
            identity_support[mapping.target_symbol] = max(
                identity_support.get(mapping.target_symbol, 0.0),
                1.0 - mapping.cost,
            )
            exemplar_coherence[mapping.target_symbol] = max(
                exemplar_coherence.get(mapping.target_symbol, 0.0),
                1.0 - selected.exemplar_cost,
            )
            conflict_penalty[mapping.target_symbol] = max(
                conflict_penalty.get(mapping.target_symbol, 0.0),
                selected.conflict_cost,
            )
        return query_scoring.SENFSignals(
            grounded_symbols=selected.grounded_symbols,
            role_signatures=selected.role_signatures,
            distortion=selected.distortion,
            source_grounding_weight=self._source_grounding_weight,
            role_compat_weight=self._role_compat_weight,
            distortion_weight=self._distortion_weight,
            identity_support=identity_support,
            exemplar_coherence=exemplar_coherence,
            conflict_penalty=conflict_penalty,
            transport_cost=selected.total_cost,
            identity_support_weight=self._identity_support_weight,
            exemplar_coherence_weight=self._exemplar_coherence_weight,
            conflict_weight=self._conflict_weight,
            transport_cost_weight=self._transport_cost_weight,
            matched_soft_mass=selected.matched_soft_mass,
            global_residual_ratio=(
                min(
                    1.0,
                    selected.polish.global_stage.global_residual
                    / self._sinkhorn_tolerance,
                )
                if selected.polish is not None
                and not selected.fallback_reason
                and selected.polish_converged is not None
                else None
            ),
            alignment_confidence=(
                selected.matched_soft_mass
                / (selected.matched_soft_mass + selected.unmatched_soft_mass)
                if selected.matched_soft_mass is not None
                and selected.unmatched_soft_mass is not None
                and selected.matched_soft_mass + selected.unmatched_soft_mass > 0.0
                else None
            ),
            matched_soft_mass_weight=self._matched_soft_mass_weight,
            global_residual_ratio_weight=self._global_residual_ratio_weight,
            alignment_confidence_weight=self._alignment_confidence_weight,
            distortion_candidate_specific=weave is not None,
        )

    def _prior_senfs(self, text: str, is_query: bool = False) -> List[SENF]:
        """Use the current query window plus semantically relevant recalled SENFs."""
        session = list(self._session[-self._max_priors :])
        recalled_prior: list[SENF] = []
        seen = {senf.senf_id for senf in session}
        for record in self._retrieve_senf_records(text):
            if len(session) + len(recalled_prior) >= self._max_priors:
                break
            recalled = self._validated_recalled_senf(record)
            if (
                recalled
                and not recalled.is_empty
                and recalled.senf_id not in seen
                and (
                    not is_query
                    or (
                        self._counterfactual_enabled
                        and self._query_context.branch_id != ACTUAL_BRANCH_ID
                        and any(
                            branch.branch_id == self._query_context.branch_id
                            for branch in recalled.branches
                        )
                    )
                    or self._senf_overlaps_question(recalled, text)
                )
            ):
                seen.add(recalled.senf_id)
                recalled_prior.append(recalled)
        # Recalled records are older context. Keeping them before the session
        # lets reverse-order resource bounding preserve recent local evidence.
        return self._bound_prior_work(recalled_prior + session)

    def _bound_prior_work(self, prior: List[SENF]) -> List[SENF]:
        bounded: list[SENF] = []
        frames_left = self._max_source_frames
        mentions_left = self._max_mentions
        for senf in reversed(prior[-self._max_priors :]):
            if frames_left <= 0 or mentions_left <= 0:
                break
            if len(senf.frames) <= frames_left and len(senf.mentions) <= mentions_left:
                bounded.append(senf)
                frames_left -= len(senf.frames)
                mentions_left -= len(senf.mentions)
        bounded.reverse()
        return bounded

    def _senf_overlaps_question(self, senf: SENF, question: str) -> bool:
        question_symbols = set(self._question_symbols(question))
        if question_symbols.intersection(senf.symbols()):
            return True
        source_text = " ".join(
            frame.source_text for frame in senf.frames if frame.source_text
        )
        source_symbols = set(self._question_symbols(source_text))
        return bool(question_symbols.intersection(source_symbols))

    def _validated_recalled_senf(self, record: object) -> Optional[SENF]:
        if not isinstance(record, dict):
            return None
        source_text = record.get("nl")
        accepted = record.get("pln")
        if not isinstance(source_text, str) or not isinstance(accepted, list):
            return None
        accepted_ids = set()
        for atom in accepted:
            if not isinstance(atom, str):
                return None
            parts = self._top_level_parts(atom)
            if len(parts) < 4 or parts[0] != ":" or not parts[1]:
                return None
            accepted_ids.add(parts[1])
        recalled = senf_from_payload(record.get(SENF_PAYLOAD_KEY))
        if recalled is None or any(
            frame.source_text != source_text or frame.source_atom_id not in accepted_ids
            for frame in recalled.frames
        ):
            return None
        expected = extract_senf(recalled.sentence_id, source_text, accepted)
        if not self._reapply_audit(expected, recalled, source_text):
            return None
        if self._exemplar_enabled:
            score_exemplars(expected)
        if (
            recalled.entities != expected.entities
            or recalled.mentions != expected.mentions
            or recalled.frames != expected.frames
            or recalled.kind_assertions != expected.kind_assertions
            or recalled.source_units != expected.source_units
            or recalled.constraints != expected.constraints
            or recalled.branches != expected.branches
            or recalled.validity_intervals != expected.validity_intervals
            or recalled.entity_persistence != expected.entity_persistence
            or recalled.source_atoms != expected.source_atoms
            or recalled.applied_mention_features != expected.applied_mention_features
            or recalled.coreference_evidence != expected.coreference_evidence
        ):
            return None
        if self._exemplar_enabled:
            score_exemplars(recalled)
        else:
            recalled.exemplar_scores.clear()
            recalled.nearest_exemplars.clear()
            recalled.active_exemplars.clear()
        return recalled

    def _apply_features(self, senf: SENF, text: str, batch: FeatureBatch) -> None:
        bindings = bind_features(text, senf.mentions, batch)
        rejected = apply_feature_bindings(
            senf, bindings, provider=self._feature_provider_name
        )
        if rejected:
            self._feature_diagnostics = tuple(self._feature_diagnostics) + tuple(rejected)

    @staticmethod
    def _reapply_audit(expected: SENF, recalled: SENF, source_text: str) -> bool:
        mentions = {mention.mention_id: mention for mention in recalled.mentions}
        by_provider: dict[str, tuple[list[SpanFeature], list[CoreferenceEvidence]]] = {}
        try:
            for item in recalled.applied_mention_features:
                features, _ = by_provider.setdefault(item.provider, ([], []))
                features.append(SpanFeature(
                    ExactSpan(item.source_span.start, item.source_span.end, item.source_text),
                    item.name, item.value, item.confidence, item.evidence,
                ))
            for item in recalled.coreference_evidence:
                anaphor = mentions[item.anaphor_mention_id]
                antecedent = mentions[item.antecedent_mention_id]
                if anaphor.char_span is None or antecedent.char_span is None:
                    return False
                _, links = by_provider.setdefault(item.provider, ([], []))
                links.append(CoreferenceEvidence(
                    ExactSpan(*anaphor.char_span, anaphor.surface),
                    ExactSpan(*antecedent.char_span, antecedent.surface),
                    item.confidence, item.evidence, item.polarity,
                ))
            for provider, (features, links) in by_provider.items():
                bindings = bind_features(
                    source_text, expected.mentions,
                    FeatureBatch(tuple(features), tuple(links)),
                )
                if apply_feature_bindings(expected, bindings, provider=provider):
                    return False
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def _retrieve_senf_records(self, text: str) -> List[dict]:
        if not self._use_vector_context or self._context_top_k <= 0:
            return []
        try:
            store = self._store()
            if not store:
                return []
            semantic_records = list(
                store.retrieve_senf_context(text, self._context_top_k)
            )
            branch_records: list[dict] = []
            if (
                self._counterfactual_enabled
                and self._query_context.branch_id != ACTUAL_BRANCH_ID
                and hasattr(store, "retrieve_senf_branch_context")
            ):
                branch_records.extend(store.retrieve_senf_branch_context(
                    self._query_context.branch_id, self._max_priors
                ))
                branch_senfs = [
                    parsed
                    for record in branch_records
                    if isinstance(record, dict)
                    if (parsed := senf_from_payload(record.get(SENF_PAYLOAD_KEY)))
                    is not None
                ]
                if branch_senfs:
                    tree = BranchingContextTree.from_senfs(
                        branch_senfs,
                        max_nodes=self._branch_max_nodes,
                        max_depth=self._branch_max_depth,
                    )
                    for branch_id in tree.lineage(self._query_context.branch_id):
                        if branch_id in (ACTUAL_BRANCH_ID, self._query_context.branch_id):
                            continue
                        branch_records.extend(store.retrieve_senf_branch_context(
                            branch_id, self._max_priors
                        ))
            records = branch_records + semantic_records
            deduped: dict[str, dict] = {}
            for record in records:
                payload = record.get(SENF_PAYLOAD_KEY) if isinstance(record, dict) else None
                key = payload.get("senf_id") if isinstance(payload, dict) else None
                if isinstance(key, str):
                    deduped.setdefault(key, record)
            return list(deduped.values())
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
