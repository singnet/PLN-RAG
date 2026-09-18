import asyncio
import logging
import re
import threading
import time
from typing import List

from config import get_settings
from core.chunker import Chunker
from core.parser import SemanticParser
from core.reasoner import Reasoner
from core.statement_validation import is_valid_statement, validate_statements
from core.answer_generator import AnswerGenerator
from core.conceptnet import ConceptNetManager
from storage.vector_store import VectorStore
from api.models import IngestItemResult, ReasonResponse


logger = logging.getLogger(__name__)


class PLNRAGService:
    """
    Orchestrates the full pipeline:
      Text → Chunker → Parser → Reasoner → AnswerGenerator

    This class is the only place that knows about all components.
    Each component only knows about its own interface.
    """

    def __init__(self, parser: SemanticParser):
        cfg = get_settings()
        self._parser = parser
        create_chunker = getattr(parser, "create_chunker", None)
        self._chunker = create_chunker() if callable(create_chunker) else Chunker()
        self._reasoner = Reasoner()
        self._vector_store = VectorStore()
        self._conceptnet = ConceptNetManager()
        self._answer_gen = AnswerGenerator()
        self._context_top_k = cfg.context_top_k
        self._query_fallback_enabled = cfg.query_fallback_enabled
        self._query_execution_policy = cfg.query_execution_policy or (
            "ranked_first_proof"
            if cfg.query_fallback_enabled
            else "ranked_first_only"
        )
        self._legacy_retry_first_candidate = (
            cfg.query_execution_policy is None and not cfg.query_fallback_enabled
        )
        self._operation_lock = threading.RLock()
        self._conceptnet.ensure_loaded(self._reasoner, self._vector_store)

    #  Ingest

    async def ingest_batch(self, texts: List[str]) -> List[IngestItemResult]:
        """
        Process texts sequentially so each sentence can see
        all previously ingested atoms as context.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ingest_batch_sync, texts)

    def _ingest_batch_sync(self, texts: List[str]) -> List[IngestItemResult]:
        with self._operation_lock:
            return [self._ingest_single(text) for text in texts]

    def _ingest_single(self, text: str) -> IngestItemResult:
        try:
            all_atoms: List[str] = []
            rejected: List[dict] = []
            supports_batch_parse = (
                self._parser.__class__.parse_batch is not SemanticParser.parse_batch
            )
            chunk_units = self._chunker.chunk(text)
            chunk_count = len(chunk_units)
            batch_count = 0
            batch_sizes: List[int] = []
            parser_calls = 0
            empty_results = 0

            if supports_batch_parse:
                batches = self._chunker.batch_chunks(
                    text,
                    max_sentences=get_settings().parser_batch_sentences,
                    max_chars=get_settings().parser_batch_max_chars,
                )
                batch_count = len(batches)
                batch_sizes = [len(batch) for batch in batches]
                parser_calls = batch_count
                for batch in batches:
                    batch_text = " ".join(batch)
                    context, vector = self._vector_store.retrieve_context(
                        batch_text, top_k=self._context_top_k
                    )
                    context = self._enrich_context(context)
                    parse_result = self._parser.parse_batch(batch, context)

                    if not parse_result.statements:
                        self._prepare_parser_ingest(parse_result, [])
                        empty_results += 1
                        logger.warning("No statements for batch preview %r", batch_text[:60])
                        continue

                    valid, rejected_local = self._validate_statements(parse_result.statements)
                    rejected.extend(rejected_local)
                    added, rejected_reasoner = self._reasoner.add_statements_report(valid)
                    rejected.extend(rejected_reasoner)
                    all_atoms.extend(added)
                    metadata, prepared = self._prepare_parser_ingest(parse_result, added)
                    if added:
                        self._vector_store.store(
                            batch_text,
                            added,
                            vector,
                            metadata=metadata,
                        )
                        if prepared:
                            self._commit_parser_ingest(parse_result)
            else:
                parser_calls = chunk_count

                for chunk in chunk_units:
                    # 2. Retrieve context from atomspace + vector store
                    context, vector = self._vector_store.retrieve_context(
                        chunk, top_k=self._context_top_k
                    )
                    # Also supplement with recent atoms from disk
                    context = self._enrich_context(context)

                    # 3. Parse chunk → PLN atoms
                    parse_result = self._parser.parse(chunk, context)

                    if not parse_result.statements:
                        self._prepare_parser_ingest(parse_result, [])
                        empty_results += 1
                        logger.warning("No statements for chunk preview %r", chunk[:60])
                        continue

                    # 4. Add to atomspace via reasoner
                    valid, rejected_local = self._validate_statements(parse_result.statements)
                    rejected.extend(rejected_local)
                    added, rejected_reasoner = self._reasoner.add_statements_report(valid)
                    rejected.extend(rejected_reasoner)
                    all_atoms.extend(added)
                    metadata, prepared = self._prepare_parser_ingest(parse_result, added)

                    # 5. Store in vector DB for future context retrieval
                    if added:
                        self._vector_store.store(
                            chunk,
                            added,
                            vector,
                            metadata=metadata,
                        )
                        if prepared:
                            self._commit_parser_ingest(parse_result)

            if not all_atoms:
                if parser_calls == 0:
                    error = "No parse units were generated for this input."
                elif empty_results == parser_calls:
                    error = "Parser produced no statements for any chunk or batch. Check parser logs and model configuration."
                else:
                    error = "All generated statements were rejected before storage."
                return IngestItemResult(
                    text=text,
                    atoms=[],
                    status="failed",
                    error=error,
                    chunk_count=chunk_count,
                    batch_count=batch_count,
                    batch_sizes=batch_sizes,
                    parser_calls=parser_calls,
                    rejected_count=len(rejected),
                    rejected_samples=[r.get("stmt", "") for r in rejected[:3] if r.get("stmt")],
                )

            return IngestItemResult(
                text=text,
                atoms=all_atoms,
                status="success",
                chunk_count=chunk_count,
                batch_count=batch_count,
                batch_sizes=batch_sizes,
                parser_calls=parser_calls,
                rejected_count=len(rejected),
                rejected_samples=[r.get("stmt", "") for r in rejected[:3] if r.get("stmt")],
            )

        except Exception as exc:
            if getattr(exc, "fail_closed", False):
                raise
            logger.exception("Ingest failed for preview %r", text[:80])
            return IngestItemResult(
                text=text,
                status="failed",
                error=str(exc),
                chunk_count=0,
                batch_count=0,
                batch_sizes=[],
                parser_calls=0,
                rejected_count=0,
                rejected_samples=[],
            )

    def _validate_statements(self, statements: List[str]) -> tuple[List[str], List[dict]]:
        """Drop malformed statements before sending them to the reasoner.

        We prefer to drop + count rather than attempt repairs, to keep benchmark
        results interpretable and avoid hiding parser bugs.
        """

        return validate_statements(statements)

    def _is_valid_statement(self, stmt: str) -> tuple[bool, str]:
        return is_valid_statement(stmt)

    def _enrich_context(self, rag_context: List[str], max_atoms: int = 50) -> List[str]:
        """
        Supplement RAG-retrieved context with the most recent atoms
        from the atomspace file, deduplicating and capping at max_atoms.
        This ensures the parser always has the full predicate vocabulary
        even when RAG similarity scores are low.
        """
        from config import get_settings
        import os

        cfg = get_settings()
        file_atoms: List[str] = []
        if os.path.exists(cfg.atomspace_path):
            with open(cfg.atomspace_path, "r") as f:
                file_atoms = [l.strip() for l in f if l.strip()]

        seen = set()
        merged = []
        for atom in file_atoms + rag_context:
            if atom not in seen:
                seen.add(atom)
                merged.append(atom)

        return merged[-max_atoms:]

    #  Query

    async def reason(self, query: str) -> ReasonResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._reason_serialized, query)

    def _reason_serialized(self, query: str) -> ReasonResponse:
        with self._operation_lock:
            return self._reason(query)

    def _reason(self, query: str) -> ReasonResponse:
        cfg = get_settings()
        execution_policy = getattr(self, "_query_execution_policy", None) or (
            "ranked_first_proof"
            if self._query_fallback_enabled
            else "ranked_first_only"
        )
        # 1. Retrieve context for translation
        t0 = time.perf_counter()
        context, _ = self._vector_store.retrieve_context(query, top_k=self._context_top_k)
        context = self._enrich_context(context)
        context_retrieval_seconds = time.perf_counter() - t0

        # 2. Parse query text → PLN query
        t1 = time.perf_counter()
        if hasattr(self._parser, "parse_query"):
            parse_result = self._parser.parse_query(query, context)
        else:
            parse_result = self._parser.parse(query, context)
        parse_query_seconds = time.perf_counter() - t1

        original_query = parse_result.original_query or (
            parse_result.queries[0] if parse_result.queries else ""
        )
        if not parse_result.queries:
            return ReasonResponse(
                query=query,
                pln_query="",
                original_query="",
                executed_query="",
                fallback_used=False,
                query_status="no_query",
                proof="",
                sources=[],
                answer="I couldn't translate this request into a logical query.",
                context_retrieval_seconds=round(context_retrieval_seconds, 4),
                parse_query_seconds=round(parse_query_seconds, 4),
                reasoning_seconds=0.0,
                source_lookup_seconds=0.0,
                answer_generation_seconds=0.0,
                candidate_count=0,
                candidate_count_tried=0,
                execution_policy=execution_policy,
                attempted_candidate_indices=[],
                attempted_queries=[],
                senf=parse_result.diagnostics,
            )

        # 4. Run reasoning via PeTTaChainer against ordered candidates
        t2 = time.perf_counter()
        proof_traces: List[str] = []
        executed_query = ""
        all_candidates = list(parse_result.queries)
        candidate_count_total = len(all_candidates)
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if execution_policy == "original_only":
            try:
                original_index = all_candidates.index(original_query)
            except ValueError:
                original_index = None
            candidates = (
                [(original_query, original_index)]
                if original_index is not None
                else []
            )
        elif execution_policy == "ranked_first_only":
            candidates = [(all_candidates[0], 0)]
        else:
            candidates = list(enumerate(all_candidates))
            candidates = [(candidate, index) for index, candidate in candidates]
            if max_tries > 0:
                candidates = candidates[:max_tries]
        candidate_count_tried = 0

        executed_candidate_index: int | None = None
        successful_candidate_index: int | None = None
        executed_temporal_index: int | None = None
        attempted_candidate_indices: list[int | None] = []
        attempted_queries: list[str] = []
        executed_diagnostics = parse_result.diagnostics
        retry_used = False
        for candidate, source_index in candidates:
            executed_query = candidate
            executed_candidate_index = source_index
            executed_temporal_index = source_index
            attempted_candidate_indices.append(source_index)
            attempted_queries.append(candidate)
            candidate_count_tried += 1
            proof_traces = self._query_candidate(
                candidate, parse_result, source_index if source_index is not None else -1
            )
            if proof_traces:
                successful_candidate_index = source_index
                break

        # Parser-specific query retry hook (used for hybrid fast-query mode).
        if (
            not proof_traces
            and (
                execution_policy == "ranked_first_proof"
                or getattr(self, "_legacy_retry_first_candidate", False)
            )
            and hasattr(self._parser, "retry_parse_query")
        ):
            try:
                retry = getattr(self._parser, "retry_parse_query")
                retry_result = retry(query, context, executed_query)
                if retry_result and retry_result.queries:
                    retry_used = True
                    more = list(retry_result.queries)
                    legacy_retry = getattr(
                        self, "_legacy_retry_first_candidate", False
                    )
                    candidate_count_total += len(more)
                    if legacy_retry:
                        more = more[:1]
                    elif max_tries > 0:
                        remaining = max_tries - candidate_count_tried
                        if remaining <= 0:
                            more = []
                        else:
                            more = more[:remaining]
                    retry_start = len(all_candidates)
                    for retry_index, candidate in enumerate(more):
                        idx = retry_start + retry_index
                        executed_query = candidate
                        executed_candidate_index = idx
                        executed_temporal_index = retry_index
                        attempted_candidate_indices.append(idx)
                        attempted_queries.append(candidate)
                        candidate_count_tried += 1
                        executed_diagnostics = retry_result.diagnostics
                        proof_traces = self._query_candidate(
                            candidate, retry_result, retry_index
                        )
                        if proof_traces:
                            successful_candidate_index = idx
                            break
            except Exception as exc:
                if getattr(exc, "fail_closed", False):
                    raise
                logger.warning("retry_parse_query failed: %s", exc)

        reasoning_seconds = time.perf_counter() - t2

        senf_diagnostics = dict(executed_diagnostics or {})
        temporal_plans = senf_diagnostics.get("candidate_temporal_plans")
        if (
            isinstance(temporal_plans, list)
            and executed_temporal_index is not None
            and executed_temporal_index < len(temporal_plans)
        ):
            senf_diagnostics["executed_temporal_plan"] = temporal_plans[
                executed_temporal_index
            ]

        proof = str(proof_traces)
        fallback_used = bool(executed_query and original_query and executed_query != original_query)
        query_status = self._classify_query_status(query, original_query, fallback_used)

        # 5. Reverse-lookup NL sources from proof atoms
        t3 = time.perf_counter()
        sources: List[str] = []
        if cfg.source_lookup_max_atoms > 0:
            sources = self._extract_sources(
                proof_traces, max_atoms=cfg.source_lookup_max_atoms
            )
        source_lookup_seconds = time.perf_counter() - t3

        # 6. Generate natural language answer
        t4 = time.perf_counter()
        if cfg.answer_generation_enabled:
            answer = self._answer_gen.generate(query, proof_traces)
            answer_generation_seconds = time.perf_counter() - t4
        else:
            # Benchmarks often disable answer generation; avoid emitting a misleading
            # "no proof" answer when we may have a proof.
            answer = ""
            answer_generation_seconds = 0.0
        if not proof_traces and query_status == "weakly_aligned":
            answer = (
                "No proof was found. The generated query is only weakly aligned with the current "
                "knowledge base, so the failure may come from query shape mismatch or missing witness facts."
            )

        return ReasonResponse(
            query=query,
            pln_query=executed_query,
            original_query=original_query,
            executed_query=executed_query,
            fallback_used=fallback_used,
            query_status=query_status,
            proof=proof,
            sources=sources,
            answer=answer,
            candidate_count=candidate_count_total,
            candidate_count_tried=candidate_count_tried,
            executed_candidate_index=executed_candidate_index,
            successful_candidate_index=successful_candidate_index,
            execution_policy=execution_policy,
            attempted_candidate_indices=attempted_candidate_indices,
            attempted_queries=attempted_queries,
            retry_used=retry_used,
            context_retrieval_seconds=round(context_retrieval_seconds, 4),
            parse_query_seconds=round(parse_query_seconds, 4),
            reasoning_seconds=round(reasoning_seconds, 4),
            source_lookup_seconds=round(source_lookup_seconds, 4),
            answer_generation_seconds=round(answer_generation_seconds, 4),
            senf=senf_diagnostics or parse_result.diagnostics,
        )

    def _query_candidate(self, query: str, result, index: int) -> List[str]:
        common = result.trusted_transient_statements or []
        specific = result.candidate_trusted_transient_statements or []
        raw = common + (specific[index] if 0 <= index < len(specific) else [])
        transient, rejected = self._validate_statements(raw)
        plans = (result.diagnostics or {}).get("candidate_temporal_plans", [])
        plan = (
            plans[index]
            if isinstance(plans, list) and 0 <= index < len(plans)
            else {}
        )
        contextual = isinstance(plan, dict) and (
            plan.get("branch_id", "actual_root") != "actual_root"
            or plan.get("validity_interval_id") is not None
        )
        if rejected:
            for item in rejected[:2]:
                logger.warning(
                    "Ignoring malformed transient context for candidate: %s",
                    item.get("error"),
                )
            # Never execute a partial context. The persistent query is still a
            # valid clean fallback, including for later canonical candidates.
            return [] if contextual else self._reasoner.query(query)
        if contextual:
            isolated_query = getattr(self._reasoner, "query_transient_only", None)
            return isolated_query(query, transient) if callable(isolated_query) else []
        if transient:
            return self._reasoner.query(query, transient_statements=transient)
        return self._reasoner.query(query)

    def _prepare_parser_ingest(
        self, parse_result, added: List[str]
    ) -> tuple[dict | None, bool]:
        prepare = getattr(self._parser, "prepare_ingest", None)
        if not callable(prepare):
            return parse_result.metadata or None, True
        try:
            metadata = prepare(parse_result, added)
            return (metadata if isinstance(metadata, dict) else None), True
        except Exception as exc:
            if getattr(exc, "fail_closed", False):
                raise
            logger.warning("parser ingest preparation failed", exc_info=True)
            parse_result.parser_state = None
            return None, False

    def _commit_parser_ingest(self, parse_result) -> None:
        commit = getattr(self._parser, "commit_ingest", None)
        if not callable(commit):
            return
        try:
            commit(parse_result)
        except Exception as exc:
            if getattr(exc, "fail_closed", False):
                raise
            logger.warning("parser ingest commit failed", exc_info=True)

    def _classify_query_status(
        self, query: str, original_query: str, fallback_used: bool
    ) -> str:
        if not original_query:
            return "no_query"
        if fallback_used:
            return "weakly_aligned"

        normalized = query.strip().lower()
        is_yes_no = normalized.startswith(
            (
                "is ",
                "are ",
                "was ",
                "were ",
                "does ",
                "do ",
                "did ",
                "can ",
                "could ",
                "has ",
                "have ",
                "had ",
            )
        )
        has_variables = self._query_has_goal_variables(original_query)
        if is_yes_no and has_variables:
            return "weakly_aligned"
        return "well_aligned"

    def _query_has_goal_variables(self, query: str) -> bool:
        variables = set(re.findall(r"[$?][A-Za-z_][A-Za-z0-9_]*", query))
        return bool(variables - {"$prf", "$tv", "?prf", "?tv"})

    def _extract_sources(self, proof_traces: List[str], max_atoms: int = 30) -> List[str]:
        """
        Extract atom names from proof traces and reverse-lookup
        their NL source sentences from the vector store.
        """
        if max_atoms <= 0:
            return []

        atoms_to_search = set()
        for trace in proof_traces:
            for match in re.findall(r"\([^()]+?\)", str(trace)):
                if "STV" not in match and len(match) >= 5:
                    atoms_to_search.add(match)

        if max_atoms > 0 and len(atoms_to_search) > max_atoms:
            atoms_to_search = set(list(atoms_to_search)[:max_atoms])
        return self._vector_store.lookup_sources_by_atoms(
            list(atoms_to_search), max_atoms=max_atoms, score_threshold=0.6
        )

    #  Reset

    def reset(self, scope: str):
        with self._operation_lock:
            if scope in ("all", "atomspace"):
                self._reasoner.reset()
                reset_parser = getattr(self._parser, "reset", None)
                if callable(reset_parser):
                    reset_parser()
            if scope in ("all", "vectordb"):
                self._vector_store.reset()
            self._conceptnet.restore_after_reset(self._reasoner, self._vector_store, scope)

    #  Health

    def health(self) -> dict:
        conceptnet = self._conceptnet.status()
        return {
            "atomspace_size": self._reasoner.size,
            "background_atomspace_size": self._reasoner.background_size,
            "vectordb_count": self._vector_store.count,
            "parser": self._parser.__class__.__name__,
            "conceptnet_enabled": conceptnet["enabled"],
            "conceptnet_indexing": conceptnet["indexing"],
            "conceptnet_vectors_indexed": conceptnet["indexed_count"],
            "conceptnet_vectors_expected": conceptnet["expected_count"],
            "conceptnet_last_error": conceptnet["last_error"],
            "status": "degraded" if conceptnet["last_error"] else "ok",
        }

    def ready(self) -> dict:
        conceptnet = self._conceptnet.status()
        qdrant_ready, qdrant_detail = self._vector_store.is_qdrant_available()
        ollama_ready, ollama_detail = self._vector_store.is_ollama_available()
        reasoner_ready = self._reasoner is not None

        conceptnet_status = "disabled"
        if conceptnet["enabled"]:
            conceptnet_status = "degraded" if conceptnet["last_error"] else "ready"

        ready = reasoner_ready and qdrant_ready and ollama_ready
        status = "ready" if ready else "unavailable"
        if ready and conceptnet_status == "degraded":
            status = "degraded"

        return {
            "status": status,
            "parser": self._parser.__class__.__name__,
            "reasoner_ready": reasoner_ready,
            "qdrant_ready": qdrant_ready,
            "ollama_ready": ollama_ready,
            "conceptnet_enabled": conceptnet["enabled"],
            "conceptnet_status": conceptnet_status,
            "conceptnet_last_error": conceptnet["last_error"],
            "details": {
                "qdrant": qdrant_detail,
                "ollama": ollama_detail,
            },
        }

    def close(self):
        self._vector_store.close()
