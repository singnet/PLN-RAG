import re


NORMALIZATION_VERSION = 2


# Singular nouns that already end in "s". The suffix rules below would strip that
# "s" and invent a non-word ("diabetes" -> "diabete", "lens" -> "len",
# "species" -> "specy"), which then fails to match any external vocabulary keyed
# on real words -- ConceptNet nodes and the SENF exemplar registry both are.
# Intra-KB matching is unaffected either way, since the mangling was at least
# deterministic; this exists so symbols stay joinable with the outside world.
_SINGULAR_ENDING_IN_S = frozenset(
    {
        "diabetes",
        "lens",
        "series",
        "species",
        "news",
        "herpes",
        "rabies",
        "measles",
        "feces",
        "bias",
        "atlas",
        "canvas",
        "physics",
        "mathematics",
        "statistics",
        "economics",
        "politics",
        "ethics",
        "premises",
    }
)


def singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word in _SINGULAR_ENDING_IN_S:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def pluralize(word: str) -> str:
    if word.endswith("y") and len(word) > 2:
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def canonical_symbol(token: str, lemmatize: bool = True, protect: bool = False) -> str:
    token = token.strip()
    if not token:
        return token
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", token)
    token = token.replace("-", "_")
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    token = token.lower()
    if lemmatize and token and not protect:
        token = "_".join(singularize(part) for part in token.split("_") if part)
    return token


def normalize_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())
