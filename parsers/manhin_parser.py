import logging
import os
import sys
from typing import List
from config import get_settings
from core.parser import SemanticParser, ParseResult


logger = logging.getLogger(__name__)


class ManhinParser(SemanticParser):
    """
    Manhin's LLM-based parser with format self-correction loop,
    FAISS predicate store, and equivalence generation.

    Expects the Manhin parser repo to be on PYTHONPATH or installed.
    Set PARSER=manhin in .env to activate.
    """

    def __init__(self):
        cfg = get_settings()
        self._openai_model = self._normalize_model_name(cfg.openai_model)
        # Lazy import — only loaded if this parser is selected
        self._ensure_module_path()
        from pipelines import nl2pln as manhin_nl2pln
        from vector_index import faiss_store
        self._nl2pln = manhin_nl2pln
        self._faiss_store = faiss_store

    def _ensure_module_path(self):
        candidates = []
        configured = os.getenv("MANHIN_PARSER_PATH", "").strip()
        if configured:
            candidates.append(configured)
        candidates.extend([
            "/deps/manhin-parser",
            "/app/local-deps/manhin-parser",
        ])
        for path in candidates:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _normalize_model_name(self, model: str) -> str:
        if model.startswith("openai/"):
            return model.split("/", 1)[1]
        return model

    def parse(self, text: str, context: List[str]) -> ParseResult:
        try:
            # Build context dict in Manhin's expected format
            context_entries = [{"title": "Existing KB atoms", "content": "\n".join(context)}] if context else []

            result = self._nl2pln(
                text,
                context=context_entries,
                mode="parsing",
                model=self._openai_model,
            )
            if result is None:
                logger.warning("Manhin parser returned no result for preview %r", text[:80])
                return ParseResult()

            _type_defs, stmts, _queries, extra_exprs, _sent_links = result

            # Merge stmts + extra_exprs as statements (type_defs excluded —
            # they are ontological and managed separately if needed)
            all_stmts = list(dict.fromkeys(stmts + extra_exprs))

            return ParseResult(statements=all_stmts, queries=[])

        except Exception:
            logger.exception("Manhin parse failed for preview %r", text[:80])
            return ParseResult()

    def parse_query(self, text: str, context: List[str]) -> ParseResult:
        """For question parsing — uses Manhin's querying mode."""
        try:
            context_entries = [{"title": "Existing KB atoms", "content": "\n".join(context)}] if context else []
            result = self._nl2pln(
                text,
                context=context_entries,
                mode="querying",
                model=self._openai_model,
            )
            if result is None:
                return ParseResult()
            _type_defs, stmts, queries, extra_exprs, _sent_links = result
            return ParseResult(
                statements=list(dict.fromkeys(stmts + extra_exprs)),
                queries=queries
            )
        except Exception:
            logger.exception("Manhin query parse failed for preview %r", text[:80])
            return ParseResult()
