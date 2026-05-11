from __future__ import annotations

import re
from typing import List

from config import get_settings
from core.parser import ParseResult, SemanticParser
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.langextract_pln_parser import LangExtractPLNParser


class CanonicalLangExtractParser(SemanticParser):
    """Conservative hybrid: LangExtract primary, canonical_pln fallback."""

    def __init__(self):
        self._primary = LangExtractPLNParser()
        self._fallback = CanonicalPLNParser()

    def create_chunker(self):
        # Prefer LangExtract's paragraph-first chunker.
        create = getattr(self._primary, "create_chunker", None)
        return create() if callable(create) else None

    def reset(self) -> None:
        reset = getattr(self._primary, "reset", None)
        if callable(reset):
            reset()

    def parse(self, text: str, context: List[str]) -> ParseResult:
        primary = self._primary.parse(text, context)
        if not primary.statements:
            # Conservative fallback: only when primary yields nothing usable.
            return self._fallback.parse(text, context)

        # Quality gate: when LangExtract produces over-literal symbols (common in
        # science/abstract sentences), prefer canonical_pln for that chunk.
        if _looks_overliteral(primary.statements):
            fallback = self._fallback.parse(text, context)
            if fallback.statements:
                return fallback

        return primary

    def parse_query(self, text: str, context: List[str]) -> ParseResult:
        mode = (get_settings().hybrid_query_mode or "").strip().lower()
        if mode not in {"langextract_first", "canonical_first", "canonical_only"}:
            mode = "langextract_first"

        if mode == "canonical_only":
            # Fast path: avoid LangExtract query call.
            return self._fallback.parse_query(text, context)

        primary = self._primary.parse_query(text, context)
        fallback = self._fallback.parse_query(text, context)

        if mode == "canonical_first":
            queries = _dedupe_preserve_order((fallback.queries or []) + (primary.queries or []))
            statements = _dedupe_preserve_order((fallback.statements or []) + (primary.statements or []))
            return ParseResult(statements=statements, queries=queries)

        # Default: try primary candidates first, then canonical fallback.
        queries = _dedupe_preserve_order((primary.queries or []) + (fallback.queries or []))
        statements = _dedupe_preserve_order((primary.statements or []) + (fallback.statements or []))
        return ParseResult(statements=statements, queries=queries)

    def retry_parse_query(self, text: str, context: List[str], attempted_query: str) -> ParseResult | None:
        """Optional second-stage query generation for fast hybrid mode.

        When HYBRID_QUERY_MODE=canonical_only, the service will execute canonical
        candidates first. If none prove, we can retry with LangExtract queries.
        """

        mode = (get_settings().hybrid_query_mode or "").strip().lower()
        if mode != "canonical_only":
            return None

        # Only retry if the initial attempt looked canonical.
        if attempted_query and "Improved" in attempted_query and "surface_wettability" in attempted_query:
            # Over-literal shape suggests LangExtract may also drift; still allow retry.
            pass

        primary = self._primary.parse_query(text, context)
        return ParseResult(
            statements=_dedupe_preserve_order(primary.statements or []),
            queries=_dedupe_preserve_order(primary.queries or []),
        )


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        clean = " ".join(str(item).split())
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _looks_overliteral(statements: List[str]) -> bool:
    """Heuristic: flags very long, phrase-like symbols that hurt reuse/proofs."""
    for stmt in statements:
        # Look at unquoted tokens that are not variables.
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", stmt):
            if token.startswith(("$", "?")):
                continue
            lowered = token.lower()
            if "of_the" in lowered or "in_culture" in lowered:
                return True
            if len(token) >= 32 and "_" in token:
                return True
            if token.count("_") >= 5:
                return True
    return False
