import logging

from core.parser import SemanticParser
from config import get_settings


logger = logging.getLogger(__name__)


def _build_parser(name: str, factory):
    try:
        return factory()
    except Exception as exc:
        logger.exception("Failed to initialize parser %r", name)
        raise RuntimeError(f"Failed to initialize parser '{name}': {exc}") from exc


def get_parser() -> SemanticParser:
    """
    Factory: returns the configured parser.
    Add new parsers here — the rest of the system never changes.
    """
    cfg = get_settings()
    name = cfg.parser.lower()

    if name == "nl2pln":
        from parsers.nl2pln_parser import NL2PLNParser

        return _build_parser(name, NL2PLNParser)

    if name == "canonical_pln":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return _build_parser(name, CanonicalPLNParser)

    if name == "manhin":
        from parsers.manhin_parser import ManhinParser

        return _build_parser(name, ManhinParser)

    if name == "langextract":
        from parsers.langextract_pln_parser import LangExtractPLNParser

        return _build_parser(name, LangExtractPLNParser)

    if name == "canonical_langextract":
        from parsers.canonical_langextract_parser import CanonicalLangExtractParser

        return _build_parser(name, CanonicalLangExtractParser)

    raise ValueError(
        f"Unknown parser '{name}'. Set PARSER to one of: nl2pln, canonical_pln, manhin, langextract, canonical_langextract"
    )
