import logging
import math
import re
from dataclasses import replace
from typing import Optional

from core.senf.types import (
    ACTUAL_BRANCH_ID,
    BranchContext,
    ClauseRole,
    Constraint,
    Context,
    Entity,
    EntityPersistence,
    EntityRef,
    FrameRef,
    KindAssertion,
    KindRef,
    Mention,
    MentionType,
    Role,
    SENF,
    SENFFrame,
    SourceSpan,
    SourceUnit,
    ValidityInterval,
    ValueRef,
)

logger = logging.getLogger(__name__)


class _MentionLimitExceeded(RuntimeError):
    pass

_ROLE_REGISTRY: dict[tuple[str, int], tuple[str, ...]] = {
    ("IsA", 2): ("Instance", "Class"),
    ("AtLocation", 2): ("Theme", "Location"),
    ("LocatedAt", 2): ("Theme", "Location"),
    ("LocatedIn", 2): ("Theme", "Location"),
    ("AtTime", 2): ("Theme", "Time"),
    ("HasProperty", 2): ("Theme", "Property"),
    ("InGroup", 2): ("Member", "Group"),
    ("Lent", 3): ("Agent", "Recipient", "Theme"),
    ("Lends", 3): ("Agent", "Recipient", "Theme"),
    ("Loaned", 3): ("Agent", "Recipient", "Theme"),
    ("Returned", 2): ("Agent", "Theme"),
    ("Returns", 2): ("Agent", "Theme"),
    ("Borrowed", 2): ("Agent", "Theme"),
    ("Borrowed", 3): ("Agent", "Theme", "Time"),
    ("Borrows", 2): ("Agent", "Theme"),
    ("Borrows", 3): ("Agent", "Theme", "Time"),
    ("Says", 2): ("Speaker", "Content"),
    ("Claims", 2): ("Speaker", "Content"),
    ("Possible", 1): ("Content",),
    ("Necessary", 1): ("Content",),
    ("May", 1): ("Content",),
    ("Must", 1): ("Content",),
}
_MODALITY_HEADS = {
    "Possible": "possible", "May": "possible",
    "Necessary": "necessary", "Must": "necessary",
}
_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
})
_NEGATION_HEADS = frozenset({"Not", "NOT", "Negation"})
_STRUCTURAL_HEADS = frozenset({
    "Implication", "Premises", "Conclusions", "And", "Or", "Conjunction", "Equivalence",
})
_CONTEXT_HEADS = frozenset({"InContext"})
_DECLARATION_HEADS = frozenset({
    "BranchContext", "ValidityInterval", "EntityPersistence",
})
_TRUTH_HEADS = frozenset({"STV", "CTV", "PointMass", "ParticleFrom"})
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_ATOM_RE = re.compile(r"^\(:\s+(\S+)\s+(.*)\)\s*$", re.DOTALL)


def _is_variable(token: str) -> bool:
    return token.startswith(("$", "?"))


def _split_top_level(body: str) -> list[str]:
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                return []
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
    if depth != 0:
        return []
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
            if depth < 0 or (depth == 0 and index != len(expr) - 1):
                return None
    return expr[1:-1].strip() if depth == 0 else None


def _split_statement(statement: object) -> Optional[tuple[str, str]]:
    if not isinstance(statement, str):
        return None
    match = _ATOM_RE.match(statement.strip())
    if not match:
        return None
    remainder = _split_top_level(match.group(2))
    body_parts = []
    for part in remainder:
        inner = _strip_outer_parens(part)
        tokens = _split_top_level(inner) if inner else []
        if tokens and tokens[0] in _TRUTH_HEADS:
            continue
        body_parts.append(part)
    return (match.group(1), body_parts[0]) if body_parts else None


def _role_name_for(head: str, arity: int, index: int) -> str:
    override = _ROLE_REGISTRY.get((head, arity))
    return override[index] if override and index < len(override) else f"Arg{index}"


class SENFExtractor:
    def __init__(self, max_mentions_per_sentence: Optional[int] = None):
        if max_mentions_per_sentence is not None and max_mentions_per_sentence < 0:
            raise ValueError("max_mentions_per_sentence must be nonnegative")
        self._max_mentions = max_mentions_per_sentence

    def extract(self, sentence_id: str, text: str, statements: list[str]) -> SENF:
        senf = SENF(
            senf_id=f"senf:{sentence_id}", sentence_id=sentence_id,
            source_units=_segment_source_units(sentence_id, text),
            source_atoms=[" ".join(str(atom).split()) for atom in statements or []],
        )
        occurrences: dict[str, list[tuple[Entity, Mention]]] = {}
        occurrence_cursors: dict[str, int] = {}
        persistence_specs: list[tuple[str, str, str, str, Optional[str]]] = []
        declarations: set[int] = set()
        declaration_failed = False
        for index, statement in enumerate(statements or []):
            split = _split_statement(statement)
            if not split:
                continue
            try:
                if self._is_declaration(split[1]):
                    declarations.add(index)
                    self._apply_declaration(split[1], senf, persistence_specs)
            except (TypeError, ValueError) as exc:
                declarations.add(index)
                declaration_failed = True
                logger.debug("SENF extraction skipped declaration %r: %s", statement, exc)
        if declaration_failed:
            senf.frames.clear()
            senf.entities.clear()
            senf.mentions.clear()
            senf.kind_assertions.clear()
            senf.constraints.clear()
            senf.entity_persistence.clear()
            return senf
        for index, statement in enumerate(statements or []):
            lengths = (
                len(senf.entities), len(senf.mentions), len(senf.frames),
                len(senf.kind_assertions), len(senf.constraints),
            )
            saved_occurrences = dict(occurrences)
            saved_cursors = dict(occurrence_cursors)
            try:
                split = _split_statement(statement)
                if split and index not in declarations:
                    atom_id, body = split
                    self._walk(
                        body, sentence_id, text, atom_id, "fact", True,
                        senf, occurrences, occurrence_cursors, None,
                    )
            except Exception as exc:
                del senf.entities[lengths[0]:]
                del senf.mentions[lengths[1]:]
                del senf.frames[lengths[2]:]
                del senf.kind_assertions[lengths[3]:]
                del senf.constraints[lengths[4]:]
                occurrences.clear()
                occurrences.update(saved_occurrences)
                occurrence_cursors.clear()
                occurrence_cursors.update(saved_cursors)
                reason = "capped atom" if isinstance(exc, _MentionLimitExceeded) else "atom"
                logger.debug("SENF extraction skipped %s %r: %s", reason, statement, exc)
        referenced_mentions = {
            role.filler.mention_id
            for frame in senf.frames
            for role in frame.roles
            if isinstance(role.filler, EntityRef)
        }
        senf.mentions[:] = [
            mention for mention in senf.mentions if mention.mention_id in referenced_mentions
        ]
        referenced_entities = {mention.entity_id for mention in senf.mentions}
        senf.entities[:] = [
            entity for entity in senf.entities if entity.entity_id in referenced_entities
        ]
        for symbol, persistence_type, status, branch_id, interval_id in persistence_specs:
            matches = [
                entity for entity in senf.entities if entity.canonical_symbol == symbol
            ]
            for entity in matches:
                senf.entity_persistence.append(EntityPersistence(
                    entity.entity_id, persistence_type, status, branch_id, interval_id,
                ))
        infer_frame_spans(senf)
        return senf

    @staticmethod
    def _declaration_parts(expr: str) -> Optional[tuple[str, list[str]]]:
        inner = _strip_outer_parens(expr)
        tokens = _split_top_level(inner) if inner is not None else []
        if not tokens or tokens[0] not in _DECLARATION_HEADS:
            return None
        if any(token.startswith("(") for token in tokens[1:]):
            raise ValueError("nested Stage 7 declaration")
        return tokens[0], tokens[1:]

    @classmethod
    def _is_declaration(cls, expr: str) -> bool:
        return cls._declaration_parts(expr) is not None

    @classmethod
    def _apply_declaration(
        cls,
        expr: str,
        senf: SENF,
        persistence_specs: list[tuple[str, str, str, str, Optional[str]]],
    ) -> None:
        parsed = cls._declaration_parts(expr)
        if parsed is None:
            return
        head, args = parsed
        if head == "BranchContext":
            if len(args) != 4:
                raise ValueError("BranchContext requires four arguments")
            branch_id, parent_id, branch_type, raw_probability = args
            if branch_type not in ("actual", "counterfactual", "projected"):
                raise ValueError("invalid branch type")
            branch = BranchContext(
                branch_id,
                None if parent_id == "none" else parent_id,
                branch_type,
                float(raw_probability),
            )
            if (
                not math.isfinite(branch.probability)
                or not 0.0 <= branch.probability <= 1.0
            ):
                raise ValueError("invalid branch probability")
            existing = next(
                (item for item in senf.branches if item.branch_id == branch_id), None
            )
            if existing is not None and existing != branch:
                raise ValueError("conflicting branch declaration")
            if existing is None:
                senf.branches.append(branch)
            return
        if head == "ValidityInterval":
            if len(args) not in (3, 5):
                raise ValueError("ValidityInterval requires three or five arguments")
            interval_id, start, end = args[:3]
            inclusive = args[3:] or ["true", "true"]
            if any(value not in ("true", "false") for value in inclusive):
                raise ValueError("invalid interval inclusivity")
            interval = ValidityInterval(
                interval_id,
                None if start == "unbounded" else start,
                None if end == "unbounded" else end,
                inclusive[0] == "true",
                inclusive[1] == "true",
            )
            if any(item.interval_id == interval_id for item in senf.validity_intervals):
                raise ValueError("duplicate interval declaration")
            senf.validity_intervals.append(interval)
            return
        if len(args) != 5:
            raise ValueError("EntityPersistence requires five arguments")
        symbol, persistence_type, status, branch_id, interval_id = args
        if persistence_type not in ("rigid", "flexible", "contingent", "temporal"):
            raise ValueError("invalid persistence type")
        if status not in ("realized", "ghost", "unfulfilled"):
            raise ValueError("invalid entity status")
        persistence_specs.append((
            symbol, persistence_type, status, branch_id,
            None if interval_id == "none" else interval_id,
        ))

    def _walk(
        self,
        expr: str,
        sentence_id: str,
        text: str,
        atom_id: str,
        clause_role: ClauseRole,
        polarity: bool,
        senf: SENF,
        occurrences: dict[str, list[tuple[Entity, Mention]]],
        occurrence_cursors: dict[str, int],
        inherited_context: Optional[Context],
    ) -> Optional[FrameRef]:
        inner = _strip_outer_parens(expr)
        tokens = _split_top_level(inner) if inner is not None else []
        if not tokens:
            return None
        head, args = tokens[0], tokens[1:]
        if head in _TRUTH_HEADS:
            return None
        if head in _NEGATION_HEADS:
            result = None
            for arg in args:
                result = self._walk(
                    arg, sentence_id, text, atom_id, clause_role, not polarity,
                    senf, occurrences, occurrence_cursors, inherited_context,
                ) or result
            return result
        if head in _STRUCTURAL_HEADS:
            result = None
            for arg in args:
                child_inner = _strip_outer_parens(arg)
                child_tokens = _split_top_level(child_inner) if child_inner is not None else []
                child_role = clause_role
                if child_tokens and child_tokens[0] == "Premises":
                    child_role = "premise"
                elif child_tokens and child_tokens[0] == "Conclusions":
                    child_role = "conclusion"
                result = self._walk(
                    arg, sentence_id, text, atom_id, child_role, polarity,
                    senf, occurrences, occurrence_cursors, inherited_context,
                ) or result
            return result
        if head in _CONTEXT_HEADS:
            if len(args) != 3 or args[0].startswith("(") or args[1].startswith("("):
                raise ValueError("InContext requires branch, interval, and content")
            branch_id, interval_id, content = args
            context = replace(
                inherited_context or Context(""),
                branch_id=branch_id,
                validity_interval_id=None if interval_id == "none" else interval_id,
            )
            return self._walk(
                content, sentence_id, text, atom_id, clause_role, polarity,
                senf, occurrences, occurrence_cursors, context,
            )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", head):
            return None

        roles = []
        for index, arg in enumerate(args):
            if arg.startswith("("):
                filler = self._walk(
                    arg, sentence_id, text, atom_id, clause_role, True,
                    senf, occurrences, occurrence_cursors, inherited_context,
                )
                if filler is None:
                    filler = ValueRef(arg, "expression")
            elif _is_variable(arg):
                filler = ValueRef(arg, "variable")
            elif _NUMERIC_RE.fullmatch(arg):
                filler = ValueRef(arg, "number")
            elif head == "IsA" and index == 1:
                filler = KindRef(arg)
            else:
                filler = self._entity_ref(
                    arg, sentence_id, text, senf, occurrences, occurrence_cursors,
                )
            roles.append(Role(_role_name_for(head, len(args), index), filler, index))

        source_unit_id = self._frame_source_unit(roles, senf)

        frame = SENFFrame(
            frame_id=f"{sentence_id}:f{len(senf.frames)}",
            predicate_head=head,
            roles=roles,
            polarity=polarity,
            source_sentence_id=sentence_id,
            source_text=text,
            source_atom_id=atom_id,
            clause_role=clause_role,
            context=replace(inherited_context, source_unit_id=source_unit_id)
            if inherited_context else Context(source_unit_id),
        )
        senf.frames.append(frame)
        self._apply_explicit_context(frame, senf)
        if (
            head == "IsA"
            and len(roles) == 2
            and isinstance(roles[0].filler, EntityRef)
            and isinstance(roles[1].filler, KindRef)
        ):
            senf.kind_assertions.append(KindAssertion(
                roles[0].filler.entity_id, roles[1].filler, polarity, frame.frame_id,
            ))
        return FrameRef(frame.frame_id)

    @staticmethod
    def _frame_source_unit(roles: list[Role], senf: SENF) -> str:
        mentions = {mention.mention_id: mention for mention in senf.mentions}
        frames = {frame.frame_id: frame for frame in senf.frames}
        for role in roles:
            if isinstance(role.filler, EntityRef):
                mention = mentions.get(role.filler.mention_id)
                if mention is not None:
                    return mention.source_unit_id
            if isinstance(role.filler, FrameRef):
                child = frames.get(role.filler.frame_id)
                if child and child.context:
                    return child.context.source_unit_id
        return senf.source_units[0].source_unit_id

    @staticmethod
    def _apply_explicit_context(frame: SENFFrame, senf: SENF) -> None:
        entity_symbols = {entity.entity_id: entity.canonical_symbol for entity in senf.entities}
        frame_by_id = {item.frame_id: item for item in senf.frames}

        def role_value(role: Role) -> Optional[str]:
            if isinstance(role.filler, EntityRef):
                return entity_symbols.get(role.filler.entity_id)
            if isinstance(role.filler, (KindRef, ValueRef)):
                return role.filler.canonical_symbol if isinstance(role.filler, KindRef) else role.filler.value
            return None

        if frame.predicate_head in ("Says", "Claims") and len(frame.roles) == 2:
            speaker = role_value(frame.roles[0])
            content = frame.roles[1].filler
            if speaker and isinstance(content, FrameRef):
                child = frame_by_id.get(content.frame_id)
                if child and child.context:
                    modality = "claim" if frame.predicate_head == "Claims" else "reported_speech"
                    child.context = replace(child.context, speaker=speaker, modality=modality)
                    child.modality = modality
                    senf.constraints.append(Constraint(
                        "modality", modality, child.frame_id, child.context.source_unit_id,
                    ))

        if frame.predicate_head in _MODALITY_HEADS and len(frame.roles) == 1:
            content = frame.roles[0].filler
            if isinstance(content, FrameRef) and content.frame_id in frame_by_id:
                child = frame_by_id[content.frame_id]
                modality = _MODALITY_HEADS[frame.predicate_head]
                if child.context:
                    child.context = replace(child.context, modality=modality)
                child.modality = modality
                senf.constraints.append(Constraint(
                    "modality", modality, child.frame_id, child.context.source_unit_id,
                ))

        for role in frame.roles:
            field = "time_ref" if role.name == "Time" else "location_ref" if role.name == "Location" else None
            value = role_value(role)
            if field and value:
                setattr(frame, field, value)
                if frame.context:
                    frame.context = replace(frame.context, **{field: value})
                senf.constraints.append(Constraint(field, value, frame.frame_id, frame.context.source_unit_id))

        if frame.predicate_head not in ("AtTime", "AtLocation") or len(frame.roles) != 2:
            return
        value = role_value(frame.roles[1])
        if not value:
            return
        field = "time_ref" if frame.predicate_head == "AtTime" else "location_ref"
        target = frame.roles[0].filler
        constrained = frame
        if isinstance(target, FrameRef) and target.frame_id in frame_by_id:
            constrained = frame_by_id[target.frame_id]
            if constrained.context:
                constrained.context = replace(constrained.context, **{field: value})
            setattr(constrained, field, value)
        if constrained is not frame:
            senf.constraints.append(Constraint(
                field, value, constrained.frame_id,
                constrained.context.source_unit_id if constrained.context else frame.context.source_unit_id,
            ))

    def _entity_ref(
        self,
        symbol: str,
        sentence_id: str,
        text: str,
        senf: SENF,
        occurrences: dict[str, list[tuple[Entity, Mention]]],
        occurrence_cursors: dict[str, int],
    ) -> EntityRef:
        # Atom arguments already inhabit the parser's canonical PLN symbol space.
        candidates = occurrences.get(symbol)
        if candidates is None:
            source_occurrences = _find_surfaces(symbol, text) or [(symbol, None)]
            if (
                self._max_mentions is not None
                and len(senf.mentions) + len(source_occurrences) > self._max_mentions
            ):
                raise _MentionLimitExceeded("atom would exceed mention cap")
            candidates = []
            for surface, span in source_occurrences:
                entity = Entity(f"{sentence_id}:e{len(senf.entities)}", symbol)
                mention = Mention(
                    surface=surface,
                    canonical_symbol=symbol,
                    sentence_id=sentence_id,
                    entity_id=entity.entity_id,
                    mention_id=f"{sentence_id}:m{len(senf.mentions)}",
                    char_span=span,
                    mention_type=_infer_mention_type(symbol, surface, span),
                    head_lemma=symbol.rsplit("_", 1)[-1].lower(),
                    source_unit_id=_source_unit_for_span(senf.source_units, span),
                    definiteness=_infer_definiteness(symbol, text, span),
                )
                senf.entities.append(entity)
                senf.mentions.append(mention)
                candidates.append((entity, mention))
            occurrences[symbol] = candidates
        cursor = occurrence_cursors.get(symbol, 0)
        occurrence_cursors[symbol] = cursor + 1
        entity, mention = candidates[cursor % len(candidates)]
        return EntityRef(entity.entity_id, mention.mention_id)


def _find_surfaces(symbol: str, text: str) -> list[tuple[str, SourceSpan]]:
    if not symbol or not text:
        return []
    parts = [part for part in symbol.split("_") if part]
    if not parts:
        return []
    pattern = r"\b" + r"[\s\-_]+".join(
        re.escape(part) + r"(?:e?s)?" for part in parts
    ) + r"\b"
    return [
        (match.group(0), SourceSpan(match.start(), match.end()))
        for match in re.finditer(pattern, text, re.IGNORECASE)
    ]


def _infer_mention_type(
    symbol: str, surface: str, span: Optional[SourceSpan],
) -> MentionType:
    if symbol.lower() in _PRONOUNS:
        return "pronoun"
    if span and span[0] > 0 and surface[:1].isupper():
        return "proper"
    if "_" in symbol or " " in surface.strip():
        return "nominal"
    return "common"


def _infer_definiteness(
    symbol: str, text: str, span: Optional[SourceSpan],
) -> str:
    if symbol.lower() in _PRONOUNS:
        return "pronoun"
    if span is None:
        return "unknown"
    prefix = text[max(0, span[0] - 24):span[0]].lower().split()
    determiner = prefix[-1].strip(" ,:;\"'") if prefix else ""
    if determiner in ("the", "this", "that", "these", "those"):
        return "demonstrative" if determiner != "the" else "definite"
    if determiner in ("a", "an", "another"):
        return "indefinite"
    return "definite" if _infer_mention_type(symbol, text[span[0]:span[1]], span) == "proper" else "unknown"


def _segment_source_units(sentence_id: str, text: str) -> list[SourceUnit]:
    """Split source text at explicit sentence boundaries, preserving exact spans."""
    spans: list[SourceSpan] = []
    start = 0
    for match in re.finditer(r"(?:[.!?]+(?=\s|$)|\n+)", text):
        end = match.end()
        if text[start:end].strip():
            spans.append(SourceSpan(start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if text[start:].strip() or not spans:
        spans.append(SourceSpan(start, len(text)))
    return [
        SourceUnit(f"{sentence_id}:u{index}", sentence_id, text[start:end], SourceSpan(start, end))
        for index, (start, end) in enumerate(spans)
    ]


def _source_unit_for_span(units: list[SourceUnit], span: Optional[SourceSpan]) -> str:
    if span is not None:
        for unit in units:
            if unit.char_span[0] <= span[0] and span[1] <= unit.char_span[1]:
                return unit.source_unit_id
    return units[0].source_unit_id


def infer_frame_spans(senf: SENF) -> None:
    """Populate only spans supported by unambiguous source occurrences."""
    mentions = {mention.mention_id: mention for mention in senf.mentions}
    frames = {frame.frame_id: frame for frame in senf.frames}
    units = {unit.source_unit_id: unit for unit in senf.source_units}

    for frame in senf.frames:
        if frame.context is None:
            continue
        unit = units.get(frame.context.source_unit_id)
        if unit is None:
            continue
        frame.clause_span = unit.char_span
        component_spans: list[SourceSpan] = []
        complete = True
        for role in frame.roles:
            filler = role.filler
            span: Optional[SourceSpan] = None
            if isinstance(filler, EntityRef):
                mention = mentions.get(filler.mention_id)
                span = mention.char_span if mention is not None else None
            elif isinstance(filler, FrameRef):
                child = frames.get(filler.frame_id)
                span = child.frame_span if child is not None else None
            else:
                value = filler.canonical_symbol if isinstance(filler, KindRef) else filler.value
                pattern = re.compile(r"\b" + re.escape(value.replace("_", " ")) + r"\b", re.IGNORECASE)
                matches = list(pattern.finditer(unit.text))
                if len(matches) == 1:
                    span = SourceSpan(
                        unit.char_span.start + matches[0].start(),
                        unit.char_span.start + matches[0].end(),
                    )
            if span is None or not (unit.char_span.start <= span.start <= span.end <= unit.char_span.end):
                complete = False
                break
            component_spans.append(SourceSpan(*span))

        predicate_matches = list(re.finditer(
            r"\b" + re.escape(frame.predicate_head) + r"\b", unit.text, re.IGNORECASE
        ))
        if complete and component_spans and len(predicate_matches) == 1:
            predicate = predicate_matches[0]
            component_spans.append(SourceSpan(
                unit.char_span.start + predicate.start(),
                unit.char_span.start + predicate.end(),
            ))
            frame.frame_span = SourceSpan(
                min(span.start for span in component_spans),
                max(span.end for span in component_spans),
            )


def extract_senf(
    sentence_id: str,
    text: str,
    statements: list[str],
    max_mentions_per_sentence: Optional[int] = None,
) -> SENF:
    return SENFExtractor(max_mentions_per_sentence).extract(sentence_id, text, statements)
