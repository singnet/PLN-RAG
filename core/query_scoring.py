"""Shared query-candidate scoring.

Extracted from parsers/canonical_pln_parser.py and core/pln_postprocessor.py, which
carried byte-equivalent copies. Divergence between them fails silently: candidates get
ranked differently, the reasoner is asked the wrong question, and the symptom is a missing
proof rather than an error.
"""

from typing import Optional


def same_shape(left: dict, right: dict) -> bool:
    return left["head"] == right["head"] and left["arity"] == right["arity"]


def signature_can_bind(query: dict, signature: dict) -> bool:
    saw_witness = False
    for q_arg, s_arg in zip(query["args"], signature["args"]):
        if q_arg.startswith(("$", "?")):
            if not s_arg.startswith(("$", "?")):
                saw_witness = True
            continue
        if q_arg != s_arg:
            return False
    return saw_witness or not query["variables"]


def has_witness_path(
    query: dict, matching_facts: list[dict], matching_conclusions: list[dict]
) -> bool:
    if not query["variables"]:
        return True
    for signature in matching_facts + matching_conclusions:
        if signature_can_bind(query, signature):
            return True
    return False


def is_fully_grounded_from_signature(query: dict, matching_facts: list[dict]) -> bool:
    for signature in matching_facts:
        if signature["args"] == query["args"]:
            return True
    return False


def score_query_candidate(
    query: dict,
    facts: list[dict],
    conclusions: list[dict],
    is_yes_no: bool,
) -> Optional[int]:
    """Rank a query candidate. Returns None to reject it outright."""
    matching_facts = [sig for sig in facts if same_shape(query, sig)]
    matching_conclusions = [sig for sig in conclusions if same_shape(query, sig)]

    if is_yes_no and query["variables"]:
        if not has_witness_path(query, matching_facts, matching_conclusions):
            return None

    score = 0
    if matching_facts:
        score += 6
    if matching_conclusions:
        score += 4
    if not query["variables"]:
        score += 3 if is_yes_no else 1
    else:
        score += 3 if not is_yes_no else 0
    if is_fully_grounded_from_signature(query, matching_facts):
        score += 2
    return score if score > 0 else None
