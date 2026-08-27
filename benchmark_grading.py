"""Gold-answer grading for benchmark artifacts.

`proof_found` only reports that some derivation terminated, not that it answered the question.
This module grades a proof's conclusion atom against a gold answer. Pure and offline: it reads
artifacts already on disk, needs no LLM or services, and never mutates its input.

Cumulative runs accumulate knowledge across cases, so a case's proof can cite another case's
atoms. Quote isolated.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from core.symbol_normalization import canonical_symbol

VERDICT_POSITIVE = "positive"
VERDICT_NEGATIVE = "negative"
VERDICTS = (VERDICT_POSITIVE, VERDICT_NEGATIVE)

MATCH_ANY = "any"
MATCH_ALL = "all"


class ArtifactSchemaError(Exception):
    """A case claims a proof but stores it under no key this module recognizes.

    Raised rather than scored zero: a silently unreadable artifact is indistinguishable from a
    genuine regression.
    """


@dataclass
class Conclusion:
    atom: str
    negated: bool
    strength: Optional[float]
    confidence: Optional[float]
    symbols: list[str] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        # An unknown strength (PointMass, ParticleFrom) is not evidence of negation.
        return not self.negated and (self.strength is None or self.strength > 0.0)


def split_top_level(text: str) -> list[str]:
    """Split a MeTTa body into top-level items, where an item is an atom or a paren group."""
    items: list[str] = []
    depth = 0
    buffer: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            depth -= 1
            buffer.append(char)
            if depth == 0:
                items.append("".join(buffer).strip())
                buffer = []
        elif char.isspace() and depth == 0:
            chunk = "".join(buffer).strip()
            if chunk:
                items.append(chunk)
            buffer = []
        else:
            buffer.append(char)
    chunk = "".join(buffer).strip()
    if chunk:
        items.append(chunk)
    return items


def _parse_truth_value(item: str) -> tuple[Optional[float], Optional[float]]:
    parts = split_top_level(item.strip()[1:-1].strip()) if item.startswith("(") else []
    if not parts or parts[0] != "STV":
        return None, None
    try:
        return float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return None, None


def _collect_symbols(atom: str) -> list[str]:
    # Predicate heads count as symbols: an explanatory answer is often carried by the head
    # rather than an argument, as in `(Closer sun other_star)`.
    flattened = atom.replace("(", " ").replace(")", " ")
    symbols: list[str] = []
    for token in flattened.split():
        if token.startswith(("$", "?")):
            continue
        canonical = canonical_symbol(token)
        if canonical and canonical not in symbols:
            symbols.append(canonical)
    return symbols


def parse_trace(trace: str) -> Optional[Conclusion]:
    """Extract the conclusion from `(: <proof-term> <conclusion> (STV s c))`.

    Paren-balanced rather than regex: a regex mangles a `Not`-wrapped conclusion and picks the
    truth value as the conclusion when the proof term is a bare atom.
    """
    text = " ".join(str(trace).split())
    if not text.startswith("(:"):
        return None
    items = split_top_level(text[2:-1].strip())
    if len(items) < 3:
        return None
    atom = items[-2]
    strength, confidence = _parse_truth_value(items[-1])

    negated = False
    inner = atom
    while inner.startswith("("):
        parts = split_top_level(inner[1:-1].strip())
        if len(parts) == 2 and parts[0] == "Not":
            negated = not negated
            inner = parts[1]
            continue
        break

    return Conclusion(
        atom=inner,
        negated=negated,
        strength=strength,
        confidence=confidence,
        symbols=_collect_symbols(inner),
    )


def extract_proof_traces(case_result: dict[str, Any]) -> list[str]:
    """Decode the proof list. Older artifacts store `raw_proof`, current ones `proof`."""
    query = (case_result.get("end_to_end") or {}).get("query") or {}
    raw = query.get("proof")
    if raw is None:
        raw = query.get("raw_proof")
    if raw is None:
        if case_result.get("proof_found"):
            raise ArtifactSchemaError(
                f"case {case_result.get('case', {}).get('case_id')} reports proof_found but "
                "carries neither 'proof' nor 'raw_proof'"
            )
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        # The payload is a JSON string holding a Python repr, so json.loads raises on it.
        decoded = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    return [str(decoded)]


def _canonical_slots(entities: list[list[str]]) -> list[set[str]]:
    return [{canonical_symbol(alias) for alias in slot if alias} for slot in entities]


def grade_case(gold: dict[str, Any], case_result: dict[str, Any]) -> dict[str, Any]:
    """Grade one case against its gold entry.

    Entity matching is exact against the canonicalized alias list, never substring: `angle`
    would otherwise take credit for a gold slot of `contact_angle`.
    """
    slots = _canonical_slots(gold.get("entities") or [])
    match_rule = gold.get("match") or MATCH_ANY
    verdict = gold.get("verdict")
    verdict_gradable = verdict in VERDICTS

    traces = extract_proof_traces(case_result)
    conclusions = [c for c in (parse_trace(t) for t in traces) if c is not None]

    result: dict[str, Any] = {
        "answer_correct": False,
        "answer_score": 0.0,
        "answer_reason": "",
        "verdict_gradable": verdict_gradable,
        "matched_entities": [],
    }

    if not traces:
        result["answer_reason"] = "no proof"
        return result
    if not conclusions:
        result["answer_reason"] = f"unparsable proof ({len(traces)} trace(s))"
        return result

    positive = [c for c in conclusions if c.is_positive]
    negated_present = any(c.negated for c in conclusions)
    strength_zero = not positive and not negated_present

    # Only positive evidence can satisfy a slot, since a strength-0.0 atom asserts its own
    # negation and `(Not (X a))` does not answer "which X". Under a negative verdict the
    # negated conclusion *is* the answer, so it contributes.
    if verdict == VERDICT_NEGATIVE:
        contributing = [c for c in conclusions if c.is_positive or c.negated or strength_zero]
    else:
        contributing = positive

    available: set[str] = set()
    for conclusion in contributing:
        available.update(conclusion.symbols)

    matched = [index for index, slot in enumerate(slots) if slot & available]
    result["matched_entities"] = sorted(
        symbol for index in matched for symbol in slots[index] & available
    )
    result["answer_score"] = round(len(matched) / len(slots), 4) if slots else 0.0

    if not slots:
        entities_ok = True
    elif match_rule == MATCH_ALL:
        entities_ok = len(matched) == len(slots)
    else:
        entities_ok = bool(matched)

    if verdict == VERDICT_POSITIVE:
        verdict_ok = bool(positive)
    elif verdict == VERDICT_NEGATIVE:
        verdict_ok = negated_present or strength_zero
    else:
        verdict_ok = True

    result["answer_correct"] = bool(entities_ok and verdict_ok)

    reasons: list[str] = []
    if strength_zero:
        reasons.append("strength 0.0 (asserts negation)")
    if slots:
        detail = ", ".join(result["matched_entities"][:4]) or "none"
        reasons.append(f"{len(matched)}/{len(slots)} slots [{match_rule}]: {detail}")
    if verdict_gradable:
        reasons.append(f"verdict {verdict}: {'ok' if verdict_ok else 'unmet'}")
    result["answer_reason"] = "; ".join(reasons)
    return result


def grade_results(
    gold_cases: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Grade every case in one parser's result list. Returns rows, does not mutate input."""
    rows: list[dict[str, Any]] = []
    for case_result in results:
        case = case_result.get("case") or {}
        case_id = case.get("case_id") or case.get("id") or case.get("name")
        gold = gold_cases.get(case_id)
        row: dict[str, Any] = {
            "case_id": case_id,
            "proof_found": bool(case_result.get("proof_found")),
        }
        if gold is None:
            row.update(
                answer_correct=None,
                answer_score=None,
                answer_reason="no gold entry",
                verdict_gradable=False,
                matched_entities=[],
            )
        else:
            row.update(grade_case(gold, case_result))
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if row.get("answer_correct") is not None]
    correct = sum(1 for row in graded if row["answer_correct"])
    scores = [row["answer_score"] for row in graded]
    return {
        "cases": len(rows),
        "answer_graded": len(graded),
        "answer_correct": correct,
        "mean_answer_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "verdict_graded": sum(1 for row in graded if row.get("verdict_gradable")),
        "proof_found": sum(1 for row in rows if row.get("proof_found")),
    }
