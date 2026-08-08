import logging
import re
from typing import Optional

from core.symbol_normalization import canonical_symbol
from core.senf.types import Literal, Mention, MentionType, Role, SENF, SENFFrame

logger = logging.getLogger(__name__)

_POSITIONAL_ROLES = ("Agent", "Patient", "Instrument")

_UNARY_ROLE = "Theme"

_ROLE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "IsA": ("Instance", "Class"),
    "AtLocation": ("Theme", "Location"),
    "HasProperty": ("Theme", "Property"),
    "InGroup": ("Member", "Group"),
}

_PRONOUNS = frozenset(
    {
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those",
    }
)

_NEGATION_HEADS = frozenset({"Not", "NOT", "Negation"})

# Wrappers that carry no predication of their own; their children are the frames.
_STRUCTURAL_HEADS = frozenset(
    {"Implication", "Premises", "Conclusions", "And", "Or", "Conjunction", "Equivalence"}
)

_TRUTH_HEADS = frozenset({"STV", "CTV", "PointMass", "ParticleFrom"})

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_ATOM_RE = re.compile(r"^\(:\s+(\S+)\s+(.*)\)\s*$", re.DOTALL)


def _is_variable(token: str) -> bool:
    return token.startswith(("$", "?"))


def _split_top_level(body: str) -> list[str]:
    #Split a parenthesized body into head and arguments, keeping nesting intact.
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
            if depth == 0:
                tokens.append("".join(current).strip())
                current = []
        elif char.isspace() and depth == 0:
            if current:
                tokens.append("".join(current).strip())
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current).strip())
    return [token for token in tokens if token]


def _strip_outer_parens(expr: str) -> Optional[str]:
    expr = expr.strip()
    if not (expr.startswith("(") and expr.endswith(")")):
        return None
    depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expr) - 1:
                return None  # e.g. "(a) (b)" — not a single expression
    return expr[1:-1].strip()


def _split_statement(statement: str) -> Optional[tuple[str, str]]:
    # Return (atom_name, body_expression) for `(: name body (STV ...))`.
    match = _ATOM_RE.match(statement.strip())
    if not match:
        return None
    name = match.group(1)
    remainder = _split_top_level(match.group(2))
    if not remainder:
        return None
    # Drop the trailing truth value; everything before it is the body.
    body_parts = []
    for part in remainder:
        inner = _strip_outer_parens(part)
        head = _split_top_level(inner)[0] if inner else ""
        if head in _TRUTH_HEADS:
            continue
        body_parts.append(part)
    if not body_parts:
        return None
    return name, body_parts[0]


def _role_name_for(head: str, index: int, arity: int) -> str:
    override = _ROLE_OVERRIDES.get(head)
    if override and index < len(override):
        return override[index]
    if arity == 1:
        return _UNARY_ROLE
    if index < len(_POSITIONAL_ROLES):
        return _POSITIONAL_ROLES[index]
    return f"Arg{index}"


class SENFExtractor:
    # Turns canonicalized atoms plus their source text into a `SENF`.

    def __init__(self, max_mentions_per_sentence: int = 64):
        self._max_mentions = max_mentions_per_sentence

    def extract(self, sentence_id: str, text: str, statements: list[str]) -> SENF:
        senf = SENF(senf_id=f"senf:{sentence_id}", sentence_id=sentence_id)
        mention_index: dict[str, Mention] = {}

        for statement in statements or []:
            try:
                self._consume_statement(statement, sentence_id, text, senf, mention_index)
            except Exception as exc:  # never let one bad atom lose the sentence
                logger.debug("SENF extraction skipped atom %r: %s", statement, exc)

        senf.mentions = list(mention_index.values())[: self._max_mentions]
        return senf

    def _consume_statement(
        self,
        statement: str,
        sentence_id: str,
        text: str,
        senf: SENF,
        mention_index: dict[str, Mention],
    ) -> None:
        split = _split_statement(statement)
        if not split:
            return
        _name, body = split
        self._walk(body, sentence_id, text, senf, mention_index, polarity=True)

    def _walk(
        self,
        expr: str,
        sentence_id: str,
        text: str,
        senf: SENF,
        mention_index: dict[str, Mention],
        polarity: bool,
    ) -> None:
        # Descend through structural wrappers, emitting a frame per predication.
        inner = _strip_outer_parens(expr)
        if inner is None:
            return
        tokens = _split_top_level(inner)
        if not tokens:
            return
        head, args = tokens[0], tokens[1:]

        if head in _TRUTH_HEADS:
            return

        if head in _NEGATION_HEADS:
            for arg in args:
                self._walk(arg, sentence_id, text, senf, mention_index, not polarity)
            return

        if head in _STRUCTURAL_HEADS:
            for arg in args:
                self._walk(arg, sentence_id, text, senf, mention_index, polarity)
            return

        # A predication. Nested expressions among its arguments are frames too.
        nested = [arg for arg in args if arg.startswith("(")]
        flat = [arg for arg in args if not arg.startswith("(")]
        for arg in nested:
            self._walk(arg, sentence_id, text, senf, mention_index, polarity)
        if not flat and nested:
            return

        frame = self._build_frame(
            head, flat, sentence_id, text, senf, mention_index, polarity
        )
        if frame is not None:
            senf.frames.append(frame)

    def _build_frame(
        self,
        head: str,
        args: list[str],
        sentence_id: str,
        text: str,
        senf: SENF,
        mention_index: dict[str, Mention],
        polarity: bool,
    ) -> Optional[SENFFrame]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", head):
            return None

        roles: list[Role] = []
        for index, arg in enumerate(args):
            role_name = _role_name_for(head, index, len(args))
            if _is_variable(arg):
                # Rule variables are not entities; keep the slot so frames from a
                # rule still align positionally with frames from a fact.
                roles.append(Role(role_name, Literal(arg, "variable")))
                continue
            if _NUMERIC_RE.match(arg):
                roles.append(Role(role_name, Literal(arg, "number")))
                continue
            mention = self._mention_for(arg, sentence_id, text, mention_index)
            roles.append(Role(role_name, mention))

        if head == "IsA" and len(roles) == 2:
            instance, klass = roles[0].filler, roles[1].filler
            if isinstance(instance, Mention) and isinstance(klass, Mention):
                senf.kinds.setdefault(instance.canonical_symbol, klass.canonical_symbol)

        return SENFFrame(
            frame_id=f"{sentence_id}:f{len(senf.frames)}",
            predicate_head=head,
            roles=roles,
            polarity=polarity,
            source_sentence_id=sentence_id,
            source_text=text,
        )

    def _mention_for(
        self,
        raw: str,
        sentence_id: str,
        text: str,
        mention_index: dict[str, Mention],
    ) -> Mention:
        symbol = canonical_symbol(raw)
        existing = mention_index.get(symbol)
        if existing is not None:
            return existing
        surface, span = _find_surface(symbol, text)
        mention = Mention(
            surface=surface or raw,
            canonical_symbol=symbol,
            sentence_id=sentence_id,
            char_span=span,
            mention_type=_infer_mention_type(symbol, surface, text, span),
            head_lemma=symbol.rsplit("_", 1)[-1] if symbol else "",
        )
        mention_index[symbol] = mention
        return mention


def _find_surface(symbol: str, text: str) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    # Locate the text span a canonical symbol came from, if it is still there.

    if not symbol or not text:
        return None, None
    parts = [part for part in symbol.split("_") if part]
    if not parts:
        return None, None
    pattern = r"\b" + r"[\s\-_]+".join(re.escape(part) + r"(?:e?s)?" for part in parts) + r"\b"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return None, None
    # A capitalized occurrence away from the start is the only unambiguous
    # evidence of a proper noun, so prefer that span when the text offers one.
    # Chunks are frequently multi-sentence, which makes this worth checking.
    for match in matches:
        if match.start() > 0 and match.group(0)[:1].isupper():
            return match.group(0), (match.start(), match.end())
    first = matches[0]
    return first.group(0), (first.start(), first.end())


def _infer_mention_type(
    symbol: str,
    surface: Optional[str],
    text: str,
    span: Optional[tuple[int, int]],
) -> MentionType:
    if symbol in _PRONOUNS:
        return "pronoun"
    if not surface or span is None:
        return "common"
    if span[0] > 0 and surface[:1].isupper():
        return "proper"
    if "_" in symbol or " " in surface.strip():
        return "nominal"
    return "common"


def extract_senf(
    sentence_id: str,
    text: str,
    statements: list[str],
    max_mentions_per_sentence: int = 64,
) -> SENF:
    return SENFExtractor(max_mentions_per_sentence).extract(sentence_id, text, statements)
