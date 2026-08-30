import logging
from typing import List

from core.lm import create_lm
from core.parser import SemanticParser, ParseResult
from config import get_settings


logger = logging.getLogger(__name__)


class NL2PLNParser(SemanticParser):
    """
    DSPy-based NL → PLN parser, optimized via SIMBA/GEPA.
    Loads a compiled module from disk (e.g. simba_all.json).
    """

    def __init__(self):
        cfg = get_settings()

        # Lazy import to avoid loading PeTTa at import time
        from nl2pln import NL2PLNModule, pln_spec

        self._pln_spec = pln_spec
        self._module = NL2PLNModule()
        self._module.load(cfg.nl2pln_module_path)
        self._module.set_lm(create_lm())
        self._nl2pln = self._module.nl2pln

    def parse(self, text: str, context: List[str]) -> ParseResult:
        try:
            result = self._nl2pln(
                sentences=[text],
                context=context,
                pln_spec=self._pln_spec,
            )
            return ParseResult(
                statements=result.statements or [],
                queries=result.queries or []
            )
        except Exception:
            logger.exception("NL2PLN parse failed for text preview %r", text[:80])
            return ParseResult()

    def parse_batch(self, texts: List[str], context: List[str]) -> ParseResult:
        try:
            sentences = [text.strip() for text in texts if text and text.strip()]
            if not sentences:
                return ParseResult()
            result = self._nl2pln(
                sentences=sentences,
                context=context,
                pln_spec=self._pln_spec,
            )
            return ParseResult(
                statements=result.statements or [],
                queries=result.queries or [],
            )
        except Exception:
            preview = texts[0] if texts else ""
            logger.exception("NL2PLN batch parse failed for preview %r", preview[:80])
            return ParseResult()
