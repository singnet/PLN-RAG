import logging
from typing import List

from core.parser import ParseResult
from parsers.canonical_pln_parser import CanonicalPLNParser


logger = logging.getLogger(__name__)


class CanonicalPLNPrevParser(CanonicalPLNParser):
    """Pre-query-planner CanonicalPLN parser snapshot for comparison."""

    def _parse_with_mode(
        self, text: str, context: List[str], is_query: bool
    ) -> ParseResult:
        try:
            concepts = self._extract_concepts(self._normalize_text(text))
            protected_constants = self._extract_protected_constants(text)
            proper_name_map = self._extract_proper_name_map(text)
            prepared_text, prepared_context = self._build_parser_inputs(
                text, context, is_query=is_query
            )
            result = self._nl2pln(
                sentences=[prepared_text],
                context=prepared_context,
                pln_spec=self._pln_spec,
            )

            statements = self._canonicalize_outputs(
                self._dedupe_preserve_order(result.statements or []),
                concepts,
                protected_constants,
                proper_name_map,
            )
            queries = self._canonicalize_outputs(
                self._dedupe_preserve_order(result.queries or []),
                concepts,
                protected_constants,
                proper_name_map,
            )
            statements = self._filter_statements(statements)

            if is_query and not queries:
                fallback_result = self._nl2pln(
                    sentences=[text.strip()],
                    context=prepared_context,
                    pln_spec=self._pln_spec,
                )
                statements = self._canonicalize_outputs(
                    self._dedupe_preserve_order(
                        fallback_result.statements or statements
                    ),
                    concepts,
                    protected_constants,
                    proper_name_map,
                )
                queries = self._canonicalize_outputs(
                    self._dedupe_preserve_order(fallback_result.queries or queries),
                    concepts,
                    protected_constants,
                    proper_name_map,
                )
                statements = self._filter_statements(statements)

            return ParseResult(statements=statements, queries=queries)
        except Exception:
            logger.exception("CanonicalPLNPrev parse failed for preview %r", text[:80])
            return ParseResult()
