import argparse
import json
from typing import Callable, List

from core.parser import ParseResult


def _load_parser_factories() -> dict[str, Callable[[], object]]:
    from parsers.canonical_pln_parser import CanonicalPLNParser
    from parsers.nl2pln_parser import NL2PLNParser

    factories: dict[str, Callable[[], object]] = {
        "nl2pln": NL2PLNParser,
        "canonical_pln": CanonicalPLNParser,
    }

    try:
        from parsers.langextract_pln_parser import LangExtractPLNParser

        factories["langextract"] = LangExtractPLNParser
    except Exception as exc:

        def _langextract_unavailable() -> object:
            raise RuntimeError(f"LangExtract parser unavailable: {exc}")

        factories["langextract"] = _langextract_unavailable

    try:
        from parsers.canonical_langextract_parser import CanonicalLangExtractParser

        factories["canonical_langextract"] = CanonicalLangExtractParser
    except Exception as exc:

        def _hybrid_unavailable() -> object:
            raise RuntimeError(f"Hybrid parser unavailable: {exc}")

        factories["canonical_langextract"] = _hybrid_unavailable

    try:
        from parsers.manhin_parser import ManhinParser

        factories["manhin"] = ManhinParser
    except Exception as exc:

        def _unavailable() -> object:
            raise RuntimeError(f"Manhin parser unavailable: {exc}")

        factories["manhin"] = _unavailable

    return factories


def _run_parse(
    parser: object, text: str, context: List[str], is_query: bool
) -> ParseResult:
    if is_query and hasattr(parser, "parse_query"):
        return parser.parse_query(text, context)
    return parser.parse(text, context)


def main() -> int:
    cli = argparse.ArgumentParser(
        description="Compare parser outputs for the same input."
    )
    cli.add_argument("--text", required=True, help="Input text or question to parse")
    cli.add_argument(
        "--mode",
        choices=("statement", "query"),
        default="statement",
        help="Whether to run statement parsing or query parsing",
    )
    cli.add_argument(
        "--context",
        action="append",
        default=[],
        help="Repeatable existing atom context line",
    )
    cli.add_argument(
        "--context-file",
        help="Optional JSON or newline-delimited file with context atoms",
    )
    args = cli.parse_args()

    context = list(args.context)
    if args.context_file:
        with open(args.context_file, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw:
            if args.context_file.endswith(".json"):
                loaded = json.loads(raw)
                if not isinstance(loaded, list):
                    raise ValueError("Context JSON must be a list of strings")
                context.extend(str(item) for item in loaded)
            else:
                context.extend(
                    line.strip() for line in raw.splitlines() if line.strip()
                )

    payload = {
        "mode": args.mode,
        "text": args.text,
        "context": context,
        "results": {},
    }
    is_query = args.mode == "query"

    for name, factory in _load_parser_factories().items():
        try:
            parser = factory()
            result = _run_parse(parser, args.text, context, is_query=is_query)
            payload["results"][name] = {
                "status": "ok",
                "statements": result.statements,
                "queries": result.queries,
            }
        except Exception as exc:
            payload["results"][name] = {
                "status": "error",
                "error": str(exc),
            }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
