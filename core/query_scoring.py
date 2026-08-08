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


def signature_to_query(signature: dict) -> str:
    args = " ".join(signature["args"])
    return f"(: $prf ({signature['head']} {args}) $tv)"


def _ground_with_constants(signature: dict, constants: list[str]) -> Optional[dict]:
    # Signatures are not guaranteed to carry a "variables" key; only parsed
    # queries are. Read variable-ness off the args instead.
    args: list[str] = []
    available = [c for c in constants if c not in signature["args"]]
    for arg in signature["args"]:
        if arg.startswith(("$", "?")):
            if not available:
                return None
            args.append(available.pop(0))
        else:
            args.append(arg)
    return {"head": signature["head"], "args": args, "arity": len(args), "variables": []}


def derive_extra_candidates(
    existing: list[dict],
    facts: list[dict],
    conclusions: list[dict],
    question_constants: list[str],
    limit: int = 4,
) -> list[str]:
    """Extend a candidate list with queries the KB could plausibly answer.

    The planner can only rank what the generator emitted, and the generator emits a
    single candidate for almost every question, so QUERY_FALLBACK_ENABLED and
    QUERY_CANDIDATE_MAX_TRIES never engage. Derive further candidates from KB
    conclusion and fact signatures whose head/arity the generator did not cover,
    grounding free slots with constants named in the question.

    Conclusions come first: those need a rule to fire, which is the case the
    generator is worst at guessing. Capped because each extra candidate costs up to
    one full CHAINING_TIMEOUT in the service's fallback loop.
    """
    covered = {(sig["head"], sig["arity"]) for sig in existing}
    out: list[str] = []
    for signature in conclusions + facts:
        if len(out) >= limit:
            break
        if signature["arity"] == 0:
            continue
        key = (signature["head"], signature["arity"])
        if key in covered:
            continue
        grounded = _ground_with_constants(signature, question_constants)
        if grounded is None:
            continue
        covered.add(key)
        out.append(signature_to_query(grounded))
    return out


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
