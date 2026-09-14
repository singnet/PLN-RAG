from __future__ import annotations

import logging
from typing import Any

from config import get_settings
from core.langextract_chunker import LangExtractChunker
from core.langextract_examples import load_langextract_prompt_spec
from core.langextract_pln import (
    collect_predicate_heads,
    format_context_hint,
    log_rejections,
    translate_extractions_to_pln,
    translate_query_extractions_to_pln,
)
from core.parser import ParseResult, SemanticParser
from core.pln_postprocessor import PLNPostprocessor


logger = logging.getLogger(__name__)


class LangExtractPLNParser(SemanticParser):
    """Experimental LangExtract-based parser using GPT/OpenAI-compatible models."""

    def __init__(self):
        cfg = get_settings()
        self._model_id = _first_nonempty(cfg.langextract_model_id, "gpt-4o-mini")
        self._model_url = _first_nonempty(cfg.langextract_model_url)
        self._api_key = _first_nonempty(cfg.langextract_api_key, cfg.openai_api_key)
        self._extraction_passes = cfg.langextract_extraction_passes
        self._max_workers = cfg.langextract_max_workers
        self._skip_fuzzy = cfg.langextract_skip_fuzzy
        self._predicate_heads: list[str] = []
        self._postprocessor = PLNPostprocessor()

        if not self._api_key and not self._model_url:
            raise ValueError(
                "No LangExtract model credentials found. Set LANGEXTRACT_API_KEY or OPENAI_API_KEY, or set LANGEXTRACT_MODEL_URL."
            )

        spec = load_langextract_prompt_spec(cfg.langextract_examples_path)
        self._statement_prompt = spec.statement_prompt
        self._query_prompt = spec.query_prompt
        self._statement_examples = spec.statement_examples
        self._query_examples = spec.query_examples

    def create_chunker(self) -> LangExtractChunker:
        return LangExtractChunker()

    def reset(self) -> None:
        self._predicate_heads = []

    def parse(self, text: str, context: list[str]) -> ParseResult:
        try:
            prompt = self._statement_prompt + format_context_hint(context, self._predicate_heads)
            extractions = self._extract(text, prompt, self._statement_examples)
            self._remember_predicates(collect_predicate_heads(extractions))
            translated = translate_extractions_to_pln(
                extractions,
                source_text=text,
                skip_fuzzy=self._skip_fuzzy,
            )
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=[],
                context=context,
                plan_queries=False,
            )
            log_rejections("LangExtractPLNParser", translated.rejected)
            return ParseResult(statements=processed.statements, queries=[])
        except Exception:
            logger.exception("LangExtract statement parse failed for preview %r", text[:80])
            return ParseResult()

    def parse_query(self, text: str, context: list[str]) -> ParseResult:
        try:
            prompt = self._query_prompt + format_context_hint(context, self._predicate_heads)
            extractions = self._extract(text, prompt, self._query_examples)
            translated = translate_query_extractions_to_pln(extractions, source_text=text)
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=translated.queries,
                context=context,
                plan_queries=True,
            )
            log_rejections("LangExtractPLNParser.query", translated.rejected)
            return ParseResult(statements=processed.statements, queries=processed.queries)
        except Exception:
            logger.exception("LangExtract query parse failed for preview %r", text[:80])
            return ParseResult()

    def _extract(self, text: str, prompt: str, examples: list[Any]) -> list[Any]:
        import langextract as lx

        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=self._model_id,
            model_url=self._model_url,
            api_key=self._api_key,
            extraction_passes=self._extraction_passes,
            max_char_buffer=max(len(text) + 1, 1000),
            max_workers=self._max_workers,
            show_progress=False,
        )
        return list(result.extractions) if hasattr(result, "extractions") else []

    def _remember_predicates(self, heads: list[str]) -> None:
        for head in heads:
            if head not in self._predicate_heads:
                self._predicate_heads.append(head)


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None
