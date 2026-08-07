import logging
import re
from typing import List

from config import get_settings
from core.lm import create_lm
from core.parser import ParseResult, SemanticParser
from core.symbol_normalization import canonical_symbol, normalize_text, pluralize, singularize


logger = logging.getLogger(__name__)


class CanonicalPLNParser(SemanticParser):
    """NL2PLN wrapper with safer, spec-driven normalization."""

    _STOPWORDS = {
        "a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","that","the","this","to","was","were","what","when","where","who","why","with","does","do","did","can","could","would","should","has","have","had","if","then","than","into","about","after","before","under","over","not","no","yes",
    }
    _STRUCTURAL_HEADS = {
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
    _PREDICATE_ALIASES = {
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
    _QUERY_MARKERS = {"who", "what", "when", "where", "why", "how", "which"}

    def __init__(self):
        cfg = get_settings()

        from nl2pln import NL2PLNModule, pln_spec

        self._pln_spec = pln_spec
        self._module = NL2PLNModule()
        self._module.load(cfg.canonical_pln_nl2pln_module_path)
        self._module.set_lm(create_lm())
        self._nl2pln = self._module.nl2pln

    def parse(self, text: str, context: List[str]) -> ParseResult:
        return self._parse_with_mode(text, context, is_query=False)

    def parse_batch(self, texts: List[str], context: List[str]) -> ParseResult:
        return self._parse_many_with_mode(texts, context, is_query=False)

    def parse_query(self, text: str, context: List[str]) -> ParseResult:
        return self._parse_with_mode(text, context, is_query=True)

    def _parse_with_mode(
        self, text: str, context: List[str], is_query: bool
    ) -> ParseResult:
        return self._parse_many_with_mode([text], context, is_query=is_query)

    def _parse_many_with_mode(
        self, texts: List[str], context: List[str], is_query: bool
    ) -> ParseResult:
        try:
            texts = [text.strip() for text in texts if text and text.strip()]
            if not texts:
                return ParseResult()

            concepts: List[str] = []
            protected_constants: set[str] = set()
            proper_name_map: dict[str, str] = {}
            for text in texts:
                for concept in self._extract_concepts(self._normalize_text(text)):
                    if concept not in concepts:
                        concepts.append(concept)
                protected_constants.update(self._extract_protected_constants(text))
                proper_name_map.update(self._extract_proper_name_map(text))

            prepared_texts, prepared_context = self._build_parser_inputs_batch(
                texts, context, is_query=is_query, concepts=concepts
            )
            result = self._nl2pln(
                sentences=prepared_texts,
                context=prepared_context,
                pln_spec=self._pln_spec,
            )

            statements = self._canonicalize_outputs(
                self._dedupe_preserve_order(result.statements or []),
                concepts,
                context,
                protected_constants,
                proper_name_map,
            )
            queries = self._canonicalize_outputs(
                self._dedupe_preserve_order(result.queries or []),
                concepts,
                context,
                protected_constants,
                proper_name_map,
            )
            statements = [self._prune_generic_sortal_premises(stmt) for stmt in statements]
            statements = self._filter_statements(statements)

            if not is_query:
                # If NL2PLN emits only rules, materialize obvious grounded premises
                # as facts when their variable names match words in the sentence.
                # This keeps reasoning from failing due to missing witnesses.
                statements = self._dedupe_preserve_order(
                    statements + self._materialize_grounded_premise_facts(texts, statements)
                )

            question_text = " ".join(texts)
            queries = self._plan_queries(question=question_text, queries=queries, statements=statements, context=context)

            if is_query and len(texts) == 1 and not queries:
                fallback_result = self._nl2pln(
                    sentences=[texts[0]],
                    context=prepared_context,
                    pln_spec=self._pln_spec,
                )
                statements = self._canonicalize_outputs(
                    self._dedupe_preserve_order(
                        fallback_result.statements or statements
                    ),
                    concepts,
                    context,
                    protected_constants,
                    proper_name_map,
                )
                queries = self._canonicalize_outputs(
                    self._dedupe_preserve_order(fallback_result.queries or queries),
                    concepts,
                    context,
                    protected_constants,
                    proper_name_map,
                )
                statements = [self._prune_generic_sortal_premises(stmt) for stmt in statements]
                statements = self._filter_statements(statements)
                queries = self._plan_queries(question=texts[0], queries=queries, statements=statements, context=context)

            return ParseResult(statements=statements, queries=queries)
        except Exception:
            preview = texts[0] if texts else ""
            logger.exception("CanonicalPLN parse failed for preview %r", preview[:80])
            return ParseResult()

    def _build_parser_inputs_batch(
        self, texts: List[str], context: List[str], is_query: bool, concepts: List[str]
    ) -> tuple[List[str], List[str]]:
        predicates = self._extract_context_predicates(context)

        hint_lines: List[str] = [
            "; keep PLN structural constructors exactly as Implication, Premises, Conclusions, STV, And, Not",
            "; keep IsA as the canonical class-membership predicate",
            "; normalize entity and class symbols to lowercase snake_case",
            "; lemmatize common nouns and verbs so plural and singular forms reuse one symbol",
            "; reuse existing predicate heads from context when possible",
            "; keep symbols consistent across all sentences in this batch",
        ]
        if concepts:
            hint_lines.append(f"; canonical common concepts: {', '.join(concepts[:12])}")
        if predicates:
            hint_lines.append(
                f"; preferred predicate heads: {', '.join(predicates[:8])}"
            )
        if is_query and texts:
            original = " ".join(" ".join(text.strip().split()) for text in texts)
            normalized = self._normalize_text(original)
            hint_lines.append(
                "; query mode: ask only for forms that your own facts or rules can directly derive"
            )
            hint_lines.extend(self._build_query_hints(original, normalized, predicates))
        elif not is_query:
            hint_lines.append(
                "; statement mode: prefer rules whose conclusions match the eventual query predicate shape"
            )

        prepared_texts = [" ".join(text.strip().split()) for text in texts]
        enriched_context = self._dedupe_preserve_order(context + hint_lines)
        return prepared_texts, enriched_context

    def _build_parser_inputs(
        self, text: str, context: List[str], is_query: bool
    ) -> tuple[str, List[str]]:
        normalized = self._normalize_text(text)
        concepts = self._extract_concepts(normalized)
        prepared_texts, prepared_context = self._build_parser_inputs_batch(
            [text], context, is_query=is_query, concepts=concepts
        )
        return prepared_texts[0], prepared_context

    def _normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def _extract_concepts(self, normalized_text: str, max_items: int = 12) -> List[str]:
        concepts: List[str] = []
        for token in normalized_text.split():
            if len(token) < 3 or token in self._STOPWORDS:
                continue
            canonical = self._singularize(token)
            if canonical not in concepts:
                concepts.append(canonical)
            if len(concepts) >= max_items:
                break
        return concepts

    def _singularize(self, word: str) -> str:
        return singularize(word)

    def _pluralize(self, word: str) -> str:
        return pluralize(word)

    def _extract_context_predicates(
        self, context: List[str], max_items: int = 12
    ) -> List[str]:
        predicates: List[str] = []
        for atom in context:
            for candidate in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)", atom):
                canonical = self._canonical_head(candidate)
                if canonical and canonical not in predicates:
                    predicates.append(canonical)
                if len(predicates) >= max_items:
                    return predicates
        return predicates

    def _canonicalize_outputs(
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
            concept_map[self._pluralize(concept)] = concept
        concept_map.update(self._extract_context_symbol_map(context))

        canonical_items = [
            self._canonicalize_atom(item, concept_map, protected_constants, proper_name_map)
            for item in items
        ]
        canonical_items = [self._normalize_isa_classes(item) for item in canonical_items]
        return self._dedupe_preserve_order(canonical_items)

    def _materialize_grounded_premise_facts(
        self, texts: List[str], statements: List[str]
    ) -> List[str]:
        normalized = self._normalize_text(" ".join(texts))
        # Avoid asserting conditional premises for definitional/indicator rules.
        # Example: "X indicates Y" should stay a rule, not assert X.
        if any(phrase in normalized for phrase in (" indicate ", " indicates ", " implies ", " suggest ", " suggests ", " if ", " when ")):
            return []
        tokens = set(normalized.split())
        facts: List[str] = []

        # Pull premise atoms out of implication statements.
        for stmt in statements:
            m = re.search(r"\(Implication\s+\(Premises\s+(.+?)\)\s+\(Conclusions\s+.+?\)\)", stmt)
            if not m:
                continue
            premises_blob = m.group(1)
            for atom in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)\s+([^()]+?)\)", premises_blob):
                head, arg_blob = atom
                args = [a.strip() for a in arg_blob.split() if a.strip()]
                if not args:
                    continue

                bound: List[str] = []
                ok = True
                for a in args:
                    if a.startswith(("$", "?")):
                        name = self._canonical_symbol(a[1:])
                        if name in tokens:
                            bound.append(name)
                        else:
                            ok = False
                            break
                    else:
                        # Already grounded.
                        bound.append(self._canonical_symbol(a))

                if not ok:
                    continue

                fact_atom = f"({head} {' '.join(bound)})"
                fact_name = f"materialized_{self._canonical_symbol(head)}_fact"
                facts.append(f"(: {fact_name} {fact_atom} (STV 1.0 1.0))")

        return self._dedupe_preserve_order(facts)

    def _extract_context_symbol_map(self, context: List[str]) -> dict[str, str]:
        symbol_map: dict[str, str] = {}
        facts, conclusions = self._collect_available_signatures([], context)
        for signature in facts + conclusions:
            for arg in signature["args"]:
                if arg.startswith(("$", "?")):
                    continue
                symbol_map.setdefault(arg, arg)
                singular = self._canonical_symbol(arg, lemmatize=True)
                plural = self._pluralize(singular) if singular else ""
                if singular:
                    symbol_map.setdefault(singular, arg)
                if plural:
                    symbol_map.setdefault(plural, arg)
        return symbol_map

    def _canonicalize_atom(
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
                head = self._is_head_position(atom, start)
                result.append(
                    self._normalize_token(
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

    def _is_head_position(self, text: str, index: int) -> bool:
        j = index - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        return j >= 0 and text[j] == "("

    def _normalize_token(
        self,
        token: str,
        head: bool,
        concept_map: dict[str, str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
    ) -> str:
        if token.startswith("$"):
            return "$" + self._canonical_symbol(token[1:], lemmatize=False)
        if token.startswith("?"):
            return "?" + self._canonical_symbol(token[1:], lemmatize=False)

        if head:
            canonical = self._canonical_head(token)
            return canonical if canonical else token

        lowered = token.lower()
        if lowered in proper_name_map:
            return proper_name_map[lowered]
        if lowered in concept_map:
            return concept_map[lowered]
        return self._canonical_symbol(token, protect=lowered in protected_constants)

    def _canonical_head(self, token: str) -> str:
        if token in self._STRUCTURAL_HEADS:
            return token
        return self._PREDICATE_ALIASES.get(
            self._canonical_symbol(token, lemmatize=False), token
        )

    def _canonical_symbol(
        self, token: str, lemmatize: bool = True, protect: bool = False
    ) -> str:
        return canonical_symbol(token, lemmatize=lemmatize, protect=protect)

    def _extract_protected_constants(self, text: str) -> set[str]:
        protected: set[str] = set()
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            canonical = self._canonical_symbol(token, lemmatize=False)
            if canonical:
                protected.add(canonical)
        return protected

    def _extract_proper_name_map(self, text: str) -> dict[str, str]:
        proper_names: dict[str, str] = {}
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            canonical = self._canonical_symbol(token, lemmatize=False)
            if not canonical:
                continue
            proper_names[canonical] = canonical
            singular = self._canonical_symbol(token, lemmatize=True)
            if singular and singular != canonical:
                proper_names[singular] = canonical
        return proper_names

    def _filter_statements(self, statements: List[str]) -> List[str]:
        filtered: List[str] = []
        for statement in statements:
            if "Implication" in statement and not self._has_valid_implication_shape(
                statement
            ):
                continue
            if "Implication" not in statement and re.search(
                r"\$[A-Za-z_][A-Za-z0-9_]*", statement
            ):
                continue
            filtered.append(statement)
        return filtered

    def _prune_generic_sortal_premises(self, statement: str) -> str:
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
            parsed = self._parse_simple_atom(premise)
            if not parsed:
                kept.append(premise)
                continue
            if (
                parsed["head"] == "IsA"
                and len(parsed["args"]) == 2
                and parsed["args"][0].startswith(("$", "?"))
            ):
                continue
            kept.append(premise)

        if len(kept) == len(premises) or not kept:
            return statement

        replacement = "(Premises " + " ".join(kept) + ")"
        return statement[: match.start()] + replacement + statement[match.end() :]

    def _has_valid_implication_shape(self, statement: str) -> bool:
        if "Implication" not in statement:
            return True
        return "(Premises" in statement and "(Conclusions" in statement

    def _normalize_isa_classes(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            subject = match.group(1)
            klass = match.group(2)
            normalized = self._canonical_symbol(klass, lemmatize=True)
            return f"(IsA {subject} {normalized})"

        return re.sub(r"\(IsA\s+([^()\s]+)\s+([^()\s]+)\)", repl, text)

    def _plan_queries(
        self,
        question: str,
        queries: List[str],
        statements: List[str],
        context: List[str],
    ) -> List[str]:
        if not queries:
            return queries

        facts, conclusions = self._collect_available_signatures(statements, context)
        is_yes_no = self._is_yes_no_question(question)
        planned: List[tuple[int, str]] = []

        for query in queries:
            parsed = self._parse_query_signature(query)
            if not parsed:
                continue
            score = self._score_query_candidate(parsed, facts, conclusions, is_yes_no)
            if score is None:
                continue
            planned.append((score, query))

        if not planned:
            if is_yes_no:
                fallback = self._build_grounded_yes_no_fallbacks(
                    question, queries, facts, conclusions
                )
                heuristic = self._build_heuristic_question_queries(question)
                return self._dedupe_preserve_order(queries + fallback + heuristic)
            return queries[:1]

        planned.sort(key=lambda item: item[0], reverse=True)
        ordered = [query for _, query in planned]
        ordered.extend(
            self._build_question_aligned_fallbacks(
                question, ordered or queries, facts, conclusions
            )
        )
        if is_yes_no:
            fallback = self._build_grounded_yes_no_fallbacks(
                question, queries, facts, conclusions
            )
            ordered.extend(fallback)
            ordered.extend(self._build_heuristic_question_queries(question))
        return self._dedupe_preserve_order(ordered)

    def _build_heuristic_question_queries(self, question: str) -> List[str]:
        normalized = self._normalize_text(question).strip("?. ")
        patterns = [
            (r"^(?:does|do|did|has|have|had)\s+(.+?)\s+have\s+(.+)$", "HasA"),
            (r"^(?:is|are|was|were)\s+(.+?)\s+used for\s+(.+)$", "UsedFor"),
            (r"^(?:is|are|was|were|can|could)\s+(.+?)\s+capable of\s+(.+)$", "CapableOf"),
            (r"^(?:is|are|was|were)\s+(.+?)\s+part of\s+(.+)$", "PartOf"),
            (r"^(?:is|are|was|were)\s+(.+?)\s+located (?:at|in|on)\s+(.+)$", "AtLocation"),
        ]

        queries: List[str] = []
        for pattern, head in patterns:
            match = re.match(pattern, normalized)
            if not match:
                continue
            subject = self._canonical_phrase(match.group(1))
            target = self._canonical_phrase(match.group(2))
            if subject and target:
                queries.append(f"(: $prf ({head} {subject} {target}) $tv)")
        return queries

    def _canonical_phrase(self, phrase: str) -> str:
        tokens = [
            token
            for token in self._normalize_text(phrase).split()
            if token not in {"a", "an", "the"}
        ]
        normalized = [self._canonical_symbol(token) for token in tokens if token]
        return "_".join(token for token in normalized if token)

    def _build_grounded_yes_no_fallbacks(
        self,
        question: str,
        queries: List[str],
        facts: list[dict],
        conclusions: list[dict],
    ) -> List[str]:
        parsed_queries = [
            parsed
            for query in queries
            if (parsed := self._parse_query_signature(query)) is not None
        ]
        if not parsed_queries:
            return []

        question_symbols = set(self._extract_question_constants(question))
        grounded: List[tuple[int, str]] = []

        for query in parsed_queries:
            if not query["variables"]:
                continue
            for signature in facts + conclusions:
                if signature["variables"]:
                    continue
                if not self._same_shape(query, signature):
                    continue
                if not self._signature_can_bind(query, signature):
                    continue
                if not self._preserves_grounded_args(query, signature, question_symbols):
                    continue

                score = 0
                if question_symbols.intersection(signature["args"]):
                    score += 5
                if signature in facts:
                    score += 3
                if signature in conclusions:
                    score += 2
                grounded.append((score, self._signature_to_query(signature)))

        grounded.sort(key=lambda item: item[0], reverse=True)
        return self._dedupe_preserve_order([query for _, query in grounded])

    def _build_question_aligned_fallbacks(
        self,
        question: str,
        queries: List[str],
        facts: list[dict],
        conclusions: list[dict],
    ) -> List[str]:
        normalized = self._normalize_text(question)
        preferred_heads = self._preferred_heads_from_question(normalized)
        if not preferred_heads:
            return []

        aligned: List[tuple[int, str]] = []
        candidates = facts + conclusions
        for query_text in queries:
            parsed = self._parse_query_signature(query_text)
            if not parsed or parsed["variables"]:
                continue
            for signature in candidates:
                if signature["arity"] != parsed["arity"]:
                    continue
                if signature["head"] not in preferred_heads:
                    continue
                score = self._score_signature_alignment(parsed, signature, normalized)
                if score <= 0:
                    continue
                aligned.append((score, self._signature_to_query(signature)))

        aligned.sort(key=lambda item: item[0], reverse=True)
        return self._dedupe_preserve_order([query for _, query in aligned])

    def _preferred_heads_from_question(self, normalized_question: str) -> List[str]:
        heads: List[str] = []
        if any(
            phrase in normalized_question
            for phrase in {"located at", "located in", " in the ", " in a ", " in an "}
        ):
            heads.append("AtLocation")
        if "used for" in normalized_question:
            heads.append("UsedFor")
        if "capable of" in normalized_question:
            heads.append("CapableOf")
        if "part of" in normalized_question:
            heads.append("PartOf")
        return heads

    def _score_signature_alignment(
        self, query: dict, signature: dict, normalized_question: str
    ) -> int:
        score = 0
        question_tokens = set(normalized_question.split())
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg == s_arg:
                score += 4
                continue
            if self._canonical_symbol(q_arg) == self._canonical_symbol(s_arg):
                score += 3
                continue
            q_parts = set(q_arg.split("_"))
            s_parts = set(s_arg.split("_"))
            if q_parts and q_parts.issubset(s_parts):
                score += 2
                continue
            if s_parts.intersection(question_tokens):
                score += 1
                continue
            return -1
        if signature["head"] != query["head"]:
            score += 3
        return score

    def _collect_available_signatures(
        self, statements: List[str], context: List[str]
    ) -> tuple[list[dict], list[dict]]:
        facts: list[dict] = []
        conclusions: list[dict] = []
        for atom in statements + context:
            facts.extend(self._extract_fact_signatures(atom))
            conclusions.extend(self._extract_conclusion_signatures(atom))
        return facts, conclusions

    def _extract_fact_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for match in re.finditer(r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\((?:STV|PointMass|ParticleFromNormal|ParticleFromPairs)", text):
            parsed = self._parse_simple_atom(match.group(1))
            if parsed and parsed["head"] != "Implication":
                signatures.append(parsed)
        return signatures

    def _extract_conclusion_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for block in re.finditer(r"\(Conclusions\s+((?:\([^()]+\)\s*)+)\)", text):
            for atom in re.finditer(r"\([^()]+\)", block.group(1)):
                parsed = self._parse_simple_atom(atom.group(0))
                if parsed:
                    signatures.append(parsed)
        return signatures

    def _parse_query_signature(self, query: str) -> dict | None:
        match = re.search(r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\$?[A-Za-z_][A-Za-z0-9_]*\)", query)
        if not match:
            return None
        return self._parse_simple_atom(match.group(1))

    def _parse_simple_atom(self, atom: str) -> dict | None:
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

    def _signature_to_query(self, signature: dict) -> str:
        args = " ".join(signature["args"])
        return f"(: $prf ({signature['head']} {args}) $tv)"

    def _extract_question_constants(self, question: str) -> List[str]:
        constants: List[str] = []
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", question):
            canonical = self._canonical_symbol(token, lemmatize=False)
            if canonical and canonical not in constants:
                constants.append(canonical)
        return constants

    def _score_query_candidate(
        self,
        query: dict,
        facts: list[dict],
        conclusions: list[dict],
        is_yes_no: bool,
    ) -> int | None:
        matching_facts = [sig for sig in facts if self._same_shape(query, sig)]
        matching_conclusions = [sig for sig in conclusions if self._same_shape(query, sig)]

        if is_yes_no and query["variables"]:
            if not self._has_witness_path(query, matching_facts, matching_conclusions):
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
        if self._is_fully_grounded_from_signature(query, matching_facts):
            score += 2
        return score if score > 0 else None

    def _same_shape(self, left: dict, right: dict) -> bool:
        return left["head"] == right["head"] and left["arity"] == right["arity"]

    def _has_witness_path(
        self,
        query: dict,
        matching_facts: list[dict],
        matching_conclusions: list[dict],
    ) -> bool:
        if not query["variables"]:
            return True
        for signature in matching_facts + matching_conclusions:
            if self._signature_can_bind(query, signature):
                return True
        return False

    def _signature_can_bind(self, query: dict, signature: dict) -> bool:
        saw_witness = False
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg.startswith(("$", "?")):
                if not s_arg.startswith(("$", "?")):
                    saw_witness = True
                continue
            if q_arg != s_arg:
                return False
        return saw_witness or not query["variables"]

    def _preserves_grounded_args(
        self, query: dict, signature: dict, question_symbols: set[str]
    ) -> bool:
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg.startswith(("$", "?")):
                continue
            if q_arg != s_arg:
                return False
        grounded_question_symbols = {
            arg
            for arg in query["args"]
            if not arg.startswith(("$", "?")) and arg in question_symbols
        }
        return grounded_question_symbols.issubset(set(signature["args"]))

    def _is_fully_grounded_from_signature(
        self, query: dict, matching_facts: list[dict]
    ) -> bool:
        for signature in matching_facts:
            if signature["args"] == query["args"]:
                return True
        return False

    def _is_yes_no_question(self, question: str) -> bool:
        tokens = self._normalize_text(question).split()
        return bool(tokens) and tokens[0] in {
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "can",
            "could",
            "has",
            "have",
            "had",
        }

    def _build_query_hints(
        self, original: str, normalized: str, predicates: List[str]
    ) -> List[str]:
        hints: List[str] = []
        tokens = normalized.split()
        if not tokens:
            return hints

        if tokens[0] in {
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "can",
            "could",
            "has",
            "have",
            "had",
        }:
            hints.append(
                "; query intent: yes/no question - prefer a direct provable query"
            )
        elif any(marker in tokens for marker in self._QUERY_MARKERS):
            hints.append(
                "; query intent: open question - prefer a variable-bearing query"
            )

        if any(
            token in {"any", "anything", "someone", "somebody", "something"}
            for token in tokens
        ):
            hints.append(
                "; existential wording may justify a helper predicate when a direct query shape is not derivable"
            )

        if "not" in tokens or "never" in tokens:
            hints.append(
                "; use negation only if it is directly supported by explicit facts or rules"
            )

        if predicates:
            hints.append(
                f"; prioritize these predicate heads first: {', '.join(predicates[:5])}"
            )

        if original.endswith("?"):
            hints.append(
                "; preserve the question semantics while keeping the final query executable"
            )

        return hints

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            clean = " ".join(item.split())
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped
