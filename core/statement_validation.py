from typing import Iterable, List


_TRUTH_VALUE_ARITY = {
    "STV": 2,
    "PointMass": 1,
    "ParticleFrom": 1,
    "ParticleFromNormal": 2,
}


def has_valid_implication_shape(statement: str) -> bool:
    try:
        expression = _parse_expression(statement)
    except ValueError:
        return False
    if not _is_list(expression) or len(expression) != 4:
        return False
    body = expression[2]
    return not (_is_list(body) and body and body[0] == "Implication") or _valid_implication(body)


def is_valid_statement(statement: str) -> tuple[bool, str]:
    if not statement.startswith("(:"):
        return False, "missing_prefix"
    if not any(weight in statement for weight in ("(STV", "(PointMass", "(ParticleFrom")):
        return False, "missing_weight"
    try:
        expression = _parse_expression(statement)
    except ValueError as exc:
        if str(exc) == "unbalanced_parens":
            return False, "unbalanced_parens"
        return False, "bad_toplevel"

    if not _is_list(expression) or len(expression) != 4 or expression[0] != ":":
        return False, "bad_toplevel"
    name, body, truth_value = expression[1:]
    if not isinstance(name, str):
        return False, "bad_toplevel"
    if name.startswith(("$", "?")):
        return False, "variable_name"
    if not _valid_atom(body):
        return False, "zero_arity_atom"
    if not _valid_truth_value(truth_value):
        return False, "bad_truth_value"
    if _contains_wrapper(body) and not _valid_implication(body):
        return False, "bad_implication_shape"
    return True, ""


def validate_statements(
    statements: Iterable[str] | None,
) -> tuple[List[str], List[dict]]:
    valid: List[str] = []
    rejected: List[dict] = []
    for statement in statements or []:
        clean = " ".join(str(statement).split())
        if not clean:
            continue
        ok, error = is_valid_statement(clean)
        if ok:
            valid.append(clean)
        else:
            rejected.append({"stmt": clean, "error": error})
    return valid, rejected


def _parse_expression(text: str):
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("bad_toplevel")

    def parse_at(index: int):
        if index >= len(tokens) or tokens[index] == ")":
            raise ValueError("unbalanced_parens")
        if tokens[index] != "(":
            return tokens[index], index + 1
        result = []
        index += 1
        while index < len(tokens) and tokens[index] != ")":
            child, index = parse_at(index)
            result.append(child)
        if index >= len(tokens):
            raise ValueError("unbalanced_parens")
        return result, index + 1

    expression, end = parse_at(0)
    if end != len(tokens):
        raise ValueError("bad_toplevel")
    return expression


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isspace() or char in "()":
            if current:
                tokens.append("".join(current))
                current = []
            if char in "()":
                tokens.append(char)
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _is_list(value) -> bool:
    return isinstance(value, list)


def _valid_atom(value) -> bool:
    if not _is_list(value) or len(value) < 2 or not isinstance(value[0], str):
        return False
    return all(isinstance(item, str) or _valid_atom(item) for item in value[1:])


def _valid_truth_value(value) -> bool:
    if not _is_list(value) or not value or not isinstance(value[0], str):
        return False
    head = value[0]
    if head in _TRUTH_VALUE_ARITY:
        return len(value) == _TRUTH_VALUE_ARITY[head] + 1 and all(
            isinstance(item, str) for item in value[1:]
        )
    if head == "ParticleFromPairs" and len(value) == 2:
        pairs = value[1]
        return _is_list(pairs) and bool(pairs) and all(
            _is_list(pair)
            and len(pair) == 2
            and all(isinstance(item, str) for item in pair)
            for pair in pairs
        )
    return False


def _contains_wrapper(value) -> bool:
    if not _is_list(value) or not value:
        return False
    if value[0] in {"Implication", "Premises", "Conclusions"}:
        return True
    return any(_contains_wrapper(item) for item in value[1:] if _is_list(item))


def _valid_implication(body) -> bool:
    if not _is_list(body) or len(body) != 3 or body[0] != "Implication":
        return False
    premises, conclusions = body[1], body[2]
    return (
        _is_list(premises)
        and len(premises) >= 2
        and premises[0] == "Premises"
        and all(_valid_atom(atom) for atom in premises[1:])
        and _is_list(conclusions)
        and len(conclusions) >= 2
        and conclusions[0] == "Conclusions"
        and all(_valid_atom(atom) for atom in conclusions[1:])
    )
