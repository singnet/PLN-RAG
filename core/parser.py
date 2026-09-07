from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ParseResult:
    statements: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    diagnostics: Optional[dict] = None
    parser_state: Any = None


class SemanticParser(ABC):
    """
    Pluggable interface for NL → PLN conversion.

    All parsers must implement parse(). The rest of the system
    only depends on this interface — never on a concrete parser.

    To add a new parser:
      1. Create parsers/your_parser.py implementing SemanticParser
      2. Add it to parsers/__init__.py
      3. Set PARSER=your_parser in .env
    """

    @abstractmethod
    def parse(self, text: str, context: List[str]) -> ParseResult:
        """
        Convert a natural language sentence to PLN atoms.

        Args:
            text:    A single sentence or short paragraph.
            context: Existing MeTTa atoms from the atomspace,
                     provided so the parser can reuse predicates
                     instead of inventing new ones.

        Returns:
            ParseResult with statements (facts/rules to add to
            atomspace) and queries (to run against the reasoner).
        """
        ...

    def parse_batch(self, texts: List[str], context: List[str]) -> ParseResult:
        """Optional batch interface. Default implementation parses sequentially."""
        statements: List[str] = []
        queries: List[str] = []
        for text in texts:
            result = self.parse(text, context)
            statements.extend(result.statements)
            queries.extend(result.queries)
        return ParseResult(statements=statements, queries=queries)
