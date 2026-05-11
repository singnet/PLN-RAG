from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from core.symbol_normalization import canonical_symbol, normalize_text, pluralize, singularize


@dataclass
class PLNPostprocessResult:
    statements: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


class PLNPostprocessor:
    """Shared PLN cleanup and query-planning layer."""

    STOPWORDS = {
        "a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","that","the","this","to","was","were","what","when","where","who","why","with","does","do","did","can","could","would","should","has","have","had","if","then","than","into","about","after","before","under","over","not","no","yes",
    }
    STRUCTURAL_HEADS = {
        "Implication",
        "Premises",
        "Conclusions",
        "STV",
        "And",
        "Or",
        "Not",
        "IsA",
        "PointMass",
        "ParticleFromNormal",
        "ParticleFromPairs",
        "GreaterThan",
        "MapDist",
        "Map2Dist",
        "AverageDist",
        "FoldAll",
        "FoldAllValue",
        "Compute",
    }
    PREDICATE_ALIASES = {
        "isa": "IsA",
        "is_a": "IsA",
        "kind_of": "IsA",
        "type_of": "IsA",
        "located_at": "AtLocation",
        "locatedat": "AtLocation",
        "used_for": "UsedFor",
        "usedfor": "UsedFor",
        "capable_of": "CapableOf",
        "capableof": "CapableOf",
        "part_of": "PartOf",
        "partof": "PartOf",
    }
    QUERY_MARKERS = {"who", "what", "when", "where", "why", "how", "which"}

    def process(
        self,
        *,
        text: str,
        statements: List[str],
        queries: List[str],
        context: List[str],
        plan_queries: bool = True,
    ) -> PLNPostprocessResult:
        concepts = self.extract_concepts(normalize_text(text))
        protected_constants = self.extract_protected_constants(text)
        proper_name_map = self.extract_proper_name_map(text)

        processed_statements = self.canonicalize_outputs(
            self.dedupe_preserve_order(statements),
            concepts,
            context,
            protected_constants,
            proper_name_map,
        )
        processed_queries = self.canonicalize_outputs(
            self.dedupe_preserve_order(queries),
            concepts,
            context,
            protected_constants,
            proper_name_map,
        )
        processed_statements = [
            self.prune_generic_sortal_premises(stmt) for stmt in processed_statements
        ]
        processed_statements = self.filter_statements(processed_statements)
        if plan_queries:
            processed_queries = self.plan_queries(
                question=text,
                queries=processed_queries,
                statements=processed_statements,
                context=context,
            )
        return PLNPostprocessResult(
            statements=processed_statements,
            queries=processed_queries,
        )

    def extract_concepts(self, normalized_text: str, max_items: int = 12) -> List[str]:
        concepts: List[str] = []
        for token in normalized_text.split():
            if len(token) < 3 or token in self.STOPWORDS:
                continue
            canonical = singularize(token)
            if canonical not in concepts:
                concepts.append(canonical)
            if len(concepts) >= max_items:
                break
        return concepts

    def extract_context_predicates(self, context: List[str], max_items: int = 12) -> List[str]:
        predicates: List[str] = []
        for atom in context:
            for candidate in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)", atom):
                canonical = self.canonical_head(candidate)
                if canonical and canonical not in predicates:
                    predicates.append(canonical)
                if len(predicates) >= max_items:
                    return predicates
        return predicates

    def extract_context_symbol_map(self, context: List[str]) -> dict[str, str]:
        symbol_map: dict[str, str] = {}
        facts, conclusions = self.collect_available_signatures([], context)
        for signature in facts + conclusions:
            for arg in signature["args"]:
                if arg.startswith(("$", "?")):
                    continue
                symbol_map.setdefault(arg, arg)
                singular = canonical_symbol(arg, lemmatize=True)
                plural = pluralize(singular) if singular else ""
                if singular:
                    symbol_map.setdefault(singular, arg)
                if plural:
                    symbol_map.setdefault(plural, arg)
        return symbol_map

    def canonicalize_outputs(
        self,
        items: List[str],
        concepts: List[str],
        context: List[str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
    ) -> List[str]:
        if not items:
            return items

        concept_map: dict[str, str] = {}
        for concept in concepts:
            concept_map[concept] = concept
            concept_map[pluralize(concept)] = concept
        concept_map.update(self.extract_context_symbol_map(context))

        canonical_items = [
            self.canonicalize_atom(item, concept_map, protected_constants, proper_name_map)
            for item in items
        ]
        canonical_items = [self.normalize_isa_classes(item) for item in canonical_items]
        return self.dedupe_preserve_order(canonical_items)

    def canonicalize_atom(
        self,
        atom: str,
        concept_map: dict[str, str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
    ) -> str:
        result: List[str] = []
        i = 0
        length = len(atom)

        while i < length:
            ch = atom[i]
            if ch in "$?" or ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < length and (atom[i].isalnum() or atom[i] == "_"):
                    i += 1
                token = atom[start:i]
                head = self.is_head_position(atom, start)
                result.append(
                    self.normalize_token(
                        token,
                        head,
                        concept_map,
                        protected_constants,
                        proper_name_map,
                    )
                )
                continue
            result.append(ch)
            i += 1

        return "".join(result)

    def is_head_position(self, text: str, index: int) -> bool:
        j = index - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        return j >= 0 and text[j] == "("

    def normalize_token(
        self,
        token: str,
        head: bool,
        concept_map: dict[str, str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
    ) -> str:
        if token.startswith("$"):
            return "$" + canonical_symbol(token[1:], lemmatize=False)
        if token.startswith("?"):
            return "?" + canonical_symbol(token[1:], lemmatize=False)
        if head:
            canonical = self.canonical_head(token)
            return canonical if canonical else token
        lowered = token.lower()
        if lowered in proper_name_map:
            return proper_name_map[lowered]
        if lowered in concept_map:
            return concept_map[lowered]
        return canonical_symbol(token, protect=lowered in protected_constants)

    def canonical_head(self, token: str) -> str:
        if token in self.STRUCTURAL_HEADS:
            return token
        return self.PREDICATE_ALIASES.get(canonical_symbol(token, lemmatize=False), token)

    def extract_protected_constants(self, text: str) -> set[str]:
        protected: set[str] = set()
        # Protect proper names and acronyms, but avoid protecting sentence-initial
        # capitalized common nouns (e.g. "Dogs are animals") which should be lemmatized.
        first_word = (text.strip().split()[:1] or [""])[0]
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            if token == first_word and token[1:].islower():
                continue
            canonical = canonical_symbol(token, lemmatize=False)
            if canonical:
                protected.add(canonical)
        return protected

    def extract_proper_name_map(self, text: str) -> dict[str, str]:
        proper_names: dict[str, str] = {}
        first_word = (text.strip().split()[:1] or [""])[0]
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            if token == first_word and token[1:].islower():
                continue
            canonical = canonical_symbol(token, lemmatize=False)
            if not canonical:
                continue
            proper_names[canonical] = canonical
            singular = canonical_symbol(token, lemmatize=True)
            if singular and singular != canonical:
                proper_names[singular] = canonical
        return proper_names

    def filter_statements(self, statements: List[str]) -> List[str]:
        filtered: List[str] = []
        for statement in statements:
            if "Implication" in statement and not self.has_valid_implication_shape(statement):
                continue
            if "Implication" not in statement and re.search(r"\$[A-Za-z_][A-Za-z0-9_]*", statement):
                continue
            filtered.append(statement)
        return filtered

    def prune_generic_sortal_premises(self, statement: str) -> str:
        if "Implication" not in statement:
            return statement
        match = re.search(r"\(Premises\s+((?:\([^()]+\)\s*)+)\)", statement)
        if not match:
            return statement
        premises = [atom.group(0) for atom in re.finditer(r"\([^()]+\)", match.group(1))]
        if len(premises) <= 1:
            return statement
        kept: List[str] = []
        for premise in premises:
            parsed = self.parse_simple_atom(premise)
            if not parsed:
                kept.append(premise)
                continue
            if parsed["head"] == "IsA" and len(parsed["args"]) == 2 and parsed["args"][0].startswith(("$", "?")):
                continue
            kept.append(premise)
        if len(kept) == len(premises) or not kept:
            return statement
        replacement = "(Premises " + " ".join(kept) + ")"
        return statement[: match.start()] + replacement + statement[match.end() :]

    def has_valid_implication_shape(self, statement: str) -> bool:
        if "Implication" not in statement:
            return True
        return "(Premises" in statement and "(Conclusions" in statement

    def normalize_isa_classes(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            subject = match.group(1)
            klass = match.group(2)
            normalized = canonical_symbol(klass, lemmatize=True)
            return f"(IsA {subject} {normalized})"

        return re.sub(r"\(IsA\s+([^()\s]+)\s+([^()\s]+)\)", repl, text)

    def plan_queries(self, question: str, queries: List[str], statements: List[str], context: List[str]) -> List[str]:
        if not queries:
            return queries
        facts, conclusions = self.collect_available_signatures(statements, context)
        is_yes_no = self.is_yes_no_question(question)
        planned: List[tuple[int, str]] = []
        for query in queries:
            parsed = self.parse_query_signature(query)
            if not parsed:
                continue
            score = self.score_query_candidate(parsed, facts, conclusions, is_yes_no)
            if score is None:
                continue
            planned.append((score, query))
        if not planned:
            return queries[:1]
        planned.sort(key=lambda item: item[0], reverse=True)
        ordered = [query for _, query in planned]
        return self.dedupe_preserve_order(ordered)

    def collect_available_signatures(self, statements: List[str], context: List[str]) -> tuple[list[dict], list[dict]]:
        facts: list[dict] = []
        conclusions: list[dict] = []
        for atom in statements + context:
            facts.extend(self.extract_fact_signatures(atom))
            conclusions.extend(self.extract_conclusion_signatures(atom))
        return facts, conclusions

    def extract_fact_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for match in re.finditer(r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\((?:STV|PointMass|ParticleFromNormal|ParticleFromPairs)", text):
            parsed = self.parse_simple_atom(match.group(1))
            if parsed and parsed["head"] != "Implication":
                signatures.append(parsed)
        return signatures

    def extract_conclusion_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for block in re.finditer(r"\(Conclusions\s+((?:\([^()]+\)\s*)+)\)", text):
            for atom in re.finditer(r"\([^()]+\)", block.group(1)):
                parsed = self.parse_simple_atom(atom.group(0))
                if parsed:
                    signatures.append(parsed)
        return signatures

    def parse_query_signature(self, query: str) -> dict | None:
        match = re.search(r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\$?[A-Za-z_][A-Za-z0-9_]*\)", query)
        if not match:
            return None
        return self.parse_simple_atom(match.group(1))

    def parse_simple_atom(self, atom: str) -> dict | None:
        match = re.fullmatch(r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)*)\)", atom.strip())
        if not match:
            return None
        head = match.group(1)
        args = [part for part in match.group(2).split() if part]
        return {
            "head": head,
            "args": args,
            "arity": len(args),
            "variables": [arg for arg in args if arg.startswith(("$", "?"))],
        }

    def score_query_candidate(self, query: dict, facts: list[dict], conclusions: list[dict], is_yes_no: bool) -> int | None:
        matching_facts = [sig for sig in facts if self.same_shape(query, sig)]
        matching_conclusions = [sig for sig in conclusions if self.same_shape(query, sig)]
        if is_yes_no and query["variables"] and not self.has_witness_path(query, matching_facts, matching_conclusions):
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
        if self.is_fully_grounded_from_signature(query, matching_facts):
            score += 2
        return score if score > 0 else None

    def same_shape(self, left: dict, right: dict) -> bool:
        return left["head"] == right["head"] and left["arity"] == right["arity"]

    def has_witness_path(self, query: dict, matching_facts: list[dict], matching_conclusions: list[dict]) -> bool:
        if not query["variables"]:
            return True
        for signature in matching_facts + matching_conclusions:
            if self.signature_can_bind(query, signature):
                return True
        return False

    def signature_can_bind(self, query: dict, signature: dict) -> bool:
        saw_witness = False
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg.startswith(("$", "?")):
                if not s_arg.startswith(("$", "?")):
                    saw_witness = True
                continue
            if q_arg != s_arg:
                return False
        return saw_witness or not query["variables"]

    def is_fully_grounded_from_signature(self, query: dict, matching_facts: list[dict]) -> bool:
        for signature in matching_facts:
            if signature["args"] == query["args"]:
                return True
        return False

    def is_yes_no_question(self, question: str) -> bool:
        tokens = normalize_text(question).split()
        return bool(tokens) and tokens[0] in {"is", "are", "was", "were", "does", "do", "did", "can", "could", "has", "have", "had"}

    def build_query_hints(self, original: str, normalized: str, predicates: List[str]) -> List[str]:
        hints: List[str] = []
        tokens = normalized.split()
        if not tokens:
            return hints
        if tokens[0] in {"is", "are", "was", "were", "does", "do", "did", "can", "could", "has", "have", "had"}:
            hints.append("; query intent: yes/no question - prefer a direct provable query")
        elif any(marker in tokens for marker in self.QUERY_MARKERS):
            hints.append("; query intent: open question - prefer a variable-bearing query")
        if predicates:
            hints.append(f"; prioritize these predicate heads first: {', '.join(predicates[:5])}")
        if original.endswith("?"):
            hints.append("; preserve the question semantics while keeping the final query executable")
        return hints

    def dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            clean = " ".join(item.split())
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped
