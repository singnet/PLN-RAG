import asyncio
import re
import ast
import time
from typing import List, Tuple

from config import get_settings
from core.chunker import Chunker
from core.parser import SemanticParser
from core.reasoner import Reasoner
from core.answer_generator import AnswerGenerator
from core.conceptnet import ConceptNetManager
from storage.vector_store import VectorStore
from api.models import IngestItemResult, ReasonResponse


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
        self._conceptnet.ensure_loaded(self._reasoner, self._vector_store)

    #  Ingest

    async def ingest_batch(self, texts: List[str]) -> List[IngestItemResult]:
        """
        Process texts sequentially so each sentence can see
        all previously ingested atoms as context.
        """
        results = []
        for text in texts:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._ingest_single, text
            )
            results.append(result)
        return results

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
                        empty_results += 1
                        print(f"[Service] No statements for batch: '{batch_text[:60]}...'")
                        continue

                    valid, rejected_local = self._validate_statements(parse_result.statements)
                    rejected.extend(rejected_local)
                    added, rejected_reasoner = self._reasoner.add_statements_report(valid)
                    rejected.extend(rejected_reasoner)
                    all_atoms.extend(added)
                    if added:
                        self._vector_store.store(batch_text, added, vector)
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
                        empty_results += 1
                        print(f"[Service] No statements for chunk: '{chunk[:60]}...'")
                        continue

                    # 4. Add to atomspace via reasoner
                    valid, rejected_local = self._validate_statements(parse_result.statements)
                    rejected.extend(rejected_local)
                    added, rejected_reasoner = self._reasoner.add_statements_report(valid)
                    rejected.extend(rejected_reasoner)
                    all_atoms.extend(added)

                    # 5. Store in vector DB for future context retrieval
                    if added:
                        self._vector_store.store(chunk, added, vector)

            if parser_calls > 0 and empty_results == parser_calls and not all_atoms:
                return IngestItemResult(
                    text=text,
                    atoms=[],
                    status="failed",
                    error="Parser produced no statements for any chunk or batch. Check parser logs and model configuration.",
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

        except Exception as e:
            import traceback

            traceback.print_exc()
            return IngestItemResult(
                text=text,
                status="failed",
                error=str(e),
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

        valid: List[str] = []
        rejected: List[dict] = []
        for stmt in statements or []:
            clean = " ".join(str(stmt).split())
            if not clean:
                continue

            ok, err = self._is_valid_statement(clean)
            if ok:
                valid.append(clean)
            else:
                rejected.append({"stmt": clean, "error": err})
        return valid, rejected

    def _is_valid_statement(self, stmt: str) -> tuple[bool, str]:
        if not stmt.startswith("(:"):
            return False, "missing_prefix"
        if "(STV" not in stmt and "(PointMass" not in stmt and "(ParticleFrom" not in stmt:
            return False, "missing_weight"
        if not self._parens_balanced(stmt):
            return False, "unbalanced_parens"

        # Basic top-level check: (: <name> <body> )
        m = re.match(r"^\(:\s+([^\s()]+)\s+(.+)\)$", stmt)
        if not m:
            return False, "bad_toplevel"
        name = m.group(1)
        if name.startswith(("$", "?")):
            return False, "variable_name"

        # Enforce implication schema to avoid reasoner syntax errors.
        if "Implication" in stmt and ("(Premises" not in stmt or "(Conclusions" not in stmt):
            return False, "bad_implication_shape"
        return True, ""

    def _parens_balanced(self, text: str) -> bool:
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

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
        cfg = get_settings()
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

        original_query = parse_result.queries[0] if parse_result.queries else ""
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
            )

        # 3. Add any supporting statements the parser generated for the query
        if parse_result.statements:
            valid, rejected_local = self._validate_statements(parse_result.statements)
            if rejected_local:
                for item in rejected_local[:2]:
                    print(f"[Service] Dropping malformed query-support statement: {item.get('error')}")
            self._reasoner.add_statements(valid)

        # 4. Run reasoning via PeTTaChainer against ordered candidates
        t2 = time.perf_counter()
        proof_traces: List[str] = []
        executed_query = ""
        candidates = (
            parse_result.queries if self._query_fallback_enabled else parse_result.queries[:1]
        )

        candidate_count_total = len(candidates)
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if self._query_fallback_enabled and max_tries > 0:
            candidates = candidates[:max_tries]
        candidate_count_tried = len(candidates)

        executed_candidate_index: int | None = None
        retry_used = False
        for idx, candidate in enumerate(candidates):
            executed_query = candidate
            executed_candidate_index = idx
            proof_traces = self._reasoner.query(candidate)
            if proof_traces:
                break

        # Parser-specific query retry hook (used for hybrid fast-query mode).
        if not proof_traces and hasattr(self._parser, "retry_parse_query"):
            try:
                retry = getattr(self._parser, "retry_parse_query")
                retry_result = retry(query, context, executed_query)
                if retry_result and retry_result.queries:
                    retry_used = True
                    more = (
                        retry_result.queries
                        if self._query_fallback_enabled
                        else retry_result.queries[:1]
                    )
                    candidate_count_total += len(more)
                    if self._query_fallback_enabled and max_tries > 0:
                        remaining = max_tries - candidate_count_tried
                        if remaining <= 0:
                            more = []
                        else:
                            more = more[:remaining]
                    candidate_count_tried += len(more)
                    for idx, candidate in enumerate(more, start=(executed_candidate_index or 0) + 1):
                        executed_query = candidate
                        executed_candidate_index = idx
                        proof_traces = self._reasoner.query(candidate)
                        if proof_traces:
                            break
            except Exception as exc:
                print(f"[Service] retry_parse_query failed: {exc}")

        reasoning_seconds = time.perf_counter() - t2

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
            retry_used=retry_used,
            context_retrieval_seconds=round(context_retrieval_seconds, 4),
            parse_query_seconds=round(parse_query_seconds, 4),
            reasoning_seconds=round(reasoning_seconds, 4),
            source_lookup_seconds=round(source_lookup_seconds, 4),
            answer_generation_seconds=round(answer_generation_seconds, 4),
        )

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

        sources = set()
        if max_atoms > 0 and len(atoms_to_search) > max_atoms:
            atoms_to_search = set(list(atoms_to_search)[:max_atoms])
        for atom_str in atoms_to_search:
            try:
                vector = self._vector_store.embed(atom_str)
                ctx, _ = self._vector_store.retrieve_context(atom_str, top_k=1)
                # retrieve_context returns atoms, not NL — direct search needed
                resp = self._vector_store._client.post(
                    f"{self._vector_store._qdrant}/collections"
                    f"/{self._vector_store._collection}/points/search",
                    json={"vector": vector, "limit": 1, "with_payload": True},
                )
                results = resp.json().get("result", [])
                if results and results[0].get("score", 0) > 0.6:
                    nl = results[0].get("payload", {}).get("nl")
                    if nl:
                        sources.add(nl)
            except Exception:
                pass

        return list(sources)

    #  Reset

    def reset(self, scope: str):
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
