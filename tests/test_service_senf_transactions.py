from types import SimpleNamespace

from core.parser import ParseResult, SemanticParser
from core.senf.types import SENF_PAYLOAD_KEY, senf_from_payload
from core.service import PLNRAGService
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_langextract_parser import CanonicalLangExtractParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


CAMERA = "(: camera_fact (HasProperty camera wide_lens) (STV 1.0 1.0))"
PRONOUN = "(: pronoun_fact (HasProperty it expensive) (STV 1.0 1.0))"


class OneChunk:
    def chunk(self, text):
        return [text]

    def batch_chunks(self, text, max_sentences, max_chars):
        return [[text]]


class RecordingReasoner:
    def __init__(self, accepted=None):
        self.accepted = accepted
        self.calls = []
        self.query_calls = []

    def add_statements_report(self, statements):
        self.calls.append(list(statements))
        if self.accepted is None:
            return list(statements), []
        return list(self.accepted), []

    def add_statements(self, statements):
        self.calls.append(list(statements))
        return list(statements)

    def query(self, query, transient_statements=None):
        self.query_calls.append((query, transient_statements))
        return ["proof"]

    def query_transient_only(self, query, transient_statements):
        return self.query(query, transient_statements)


class FixedParser(SemanticParser):
    def __init__(self, statements):
        self.statements = statements
        self.prepared = []
        self.committed = []

    def parse(self, text, context):
        return ParseResult(statements=list(self.statements), parser_state=text)

    def prepare_ingest(self, result, accepted):
        self.prepared.append(list(accepted))
        result.metadata = {"accepted": list(accepted)} if accepted else {}
        return result.metadata

    def commit_ingest(self, result):
        self.committed.append(result.metadata)


class FailingStore:
    points = []

    def retrieve_context(self, text, top_k):
        return [], [0.0]

    def store(self, sentence, atoms, vector, metadata=None):
        raise RuntimeError("vector storage unavailable")


def service_for(parser, vector_store, reasoner=None):
    service = PLNRAGService.__new__(PLNRAGService)
    service._parser = parser
    service._chunker = OneChunk()
    service._reasoner = reasoner or RecordingReasoner()
    service._vector_store = vector_store
    service._context_top_k = 5
    service._enrich_context = lambda context: context
    return service


def senf_parser(monkeypatch):
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    parser._use_vector_context = False
    return parser


def accepted_hook(parser, text, statements):
    filtered, _ = parser._post_filter_hook([text], list(statements), [], [], False)
    result = ParseResult(statements=filtered, parser_state=parser._pending_ingest)
    metadata = parser.prepare_ingest(result, filtered)
    parser.commit_ingest(result)
    parser._pending_ingest = None
    return filtered, metadata


def test_all_rejected_ingest_fails_and_discards_parser_state(fake_vector_store):
    parser = FixedParser(["(: invalid (Ghost entity))"])
    service = service_for(parser, fake_vector_store)

    result = service._ingest_single("Ghost entity.")

    assert result.status == "failed"
    assert result.atoms == []
    assert result.rejected_count == 1
    assert result.error == "All generated statements were rejected before storage."
    assert parser.prepared == [[]]
    assert parser.committed == []
    assert fake_vector_store.points == []


def test_only_reasoner_accepted_atoms_reach_metadata_and_commit(fake_vector_store):
    rejected = "(: rejected (Other ghost) (STV 1.0 1.0))"
    parser = FixedParser([CAMERA, rejected])
    service = service_for(
        parser, fake_vector_store, reasoner=RecordingReasoner(accepted=[CAMERA])
    )

    result = service._ingest_single("The camera has a wide lens.")

    assert result.status == "success"
    assert result.atoms == [CAMERA]
    assert parser.prepared == [[CAMERA]]
    assert parser.committed == [{"accepted": [CAMERA]}]
    assert fake_vector_store.points[0]["payload"]["accepted"] == [CAMERA]


def test_vector_failure_does_not_commit_parser_session():
    parser = FixedParser([CAMERA])
    reasoner = RecordingReasoner()
    service = service_for(parser, FailingStore(), reasoner=reasoner)

    result = service._ingest_single("The camera has a wide lens.")

    assert result.status == "failed"
    assert reasoner.calls == [[CAMERA]]
    assert parser.committed == []


def test_prepare_failure_skips_commit_but_preserves_canonical_storage(fake_vector_store):
    class BrokenSENFParser(FixedParser):
        def prepare_ingest(self, result, accepted):
            raise RuntimeError("senf extraction failed")

    parser = BrokenSENFParser([CAMERA])
    service = service_for(parser, fake_vector_store)

    result = service._ingest_single("The camera has a wide lens.")

    assert result.status == "success"
    assert result.atoms == [CAMERA]
    assert parser.committed == []
    assert fake_vector_store.points[0]["payload"]["pln"] == [CAMERA]


def test_senf_session_and_payload_exclude_rejected_atoms(monkeypatch):
    parser = senf_parser(monkeypatch)
    ghost = "(: ghost_fact (Haunts ghost camera) (STV 1.0 1.0))"
    filtered, _ = parser._post_filter_hook(
        ["The camera has a wide lens; a ghost haunts it."],
        [CAMERA, ghost],
        [],
        [],
        False,
    )
    result = ParseResult(statements=filtered, parser_state=parser._pending_ingest)

    metadata = parser.prepare_ingest(result, [filtered[0]])
    parser.commit_ingest(result)

    stored = senf_from_payload(metadata[SENF_PAYLOAD_KEY])
    assert "ghost" not in stored.symbols()
    assert "ghost" not in parser._session[-1].symbols()


def test_stored_pln_and_senf_preserve_pronoun_while_query_transport_targets_camera(
    monkeypatch, fake_vector_store
):
    parser = senf_parser(monkeypatch)
    accepted_hook(parser, "The camera has a wide lens.", [CAMERA])

    query_statements, queries = parser._post_filter_hook(
        ["Is it expensive?"],
        [PRONOUN],
        ["(: $prf (HasProperty it expensive) $tv)"],
        [],
        True,
    )
    assert queries == ["(: $prf (HasProperty it expensive) $tv)"]
    planned = parser._plan_queries(
        "Is it expensive?", queries, query_statements, []
    )
    assert "(: $prf (HasProperty camera expensive) $tv)" in planned

    filtered, _ = parser._post_filter_hook(
        ["It is expensive."], [PRONOUN], [], [], False
    )
    pending = parser._pending_ingest
    parser.parse_batch = lambda _batch, _context: ParseResult(
        statements=filtered, parser_state=pending
    )
    service = service_for(parser, fake_vector_store)

    result = service._ingest_single("It is expensive.")

    payload = fake_vector_store.points[-1]["payload"]
    stored = senf_from_payload(payload[SENF_PAYLOAD_KEY])
    pronoun = next(mention for mention in stored.mentions if mention.canonical_symbol == "it")
    assert result.atoms == [PRONOUN]
    assert payload["pln"] == [PRONOUN]
    assert stored.symbols() == {"it", "expensive"}
    assert (pronoun.surface, pronoun.char_span) == ("It", (0, 2))


def test_consecutive_queries_reuse_the_same_ingested_session(monkeypatch):
    parser = senf_parser(monkeypatch)
    accepted_hook(parser, "The camera has a wide lens.", [CAMERA])
    query = "(: $prf (HasProperty camera wide_lens) $tv)"

    statements, queries = parser._post_filter_hook(
        ["Does the camera have a wide lens?"], [], [query], [], True
    )
    parser._plan_queries(
        "Does the camera have a wide lens?", queries, statements, []
    )
    first_sources = {pair.source_frame_id for pair in parser._weave.pairs}
    statements, queries = parser._post_filter_hook(
        ["Does the camera still have a wide lens?"], [], [query], [], True
    )
    parser._plan_queries(
        "Does the camera still have a wide lens?", queries, statements, []
    )

    assert parser._weave is not None
    assert {pair.source_frame_id for pair in parser._weave.pairs} == first_sources


def query_service(parser, vector_store, reasoner):
    service = PLNRAGService.__new__(PLNRAGService)
    service._parser = parser
    service._reasoner = reasoner
    service._vector_store = vector_store
    service._context_top_k = 5
    service._query_fallback_enabled = True
    service._enrich_context = lambda context: context
    service._extract_sources = lambda traces, max_atoms: []
    service._answer_gen = SimpleNamespace(generate=lambda query, traces: "")
    return service


def query_settings():
    return SimpleNamespace(
        query_candidate_max_tries=5,
        source_lookup_max_atoms=0,
        answer_generation_enabled=False,
    )


def test_ordinary_query_statements_are_non_authoritative_and_never_persisted(
    monkeypatch, fake_vector_store
):
    transient = "(: helper (Helper camera) (STV 1.0 1.0))"
    query = "(: $prf (Answer camera) $tv)"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=[query], transient_statements=[transient])

    reasoner = RecordingReasoner()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is the answer?")

    assert response.proof == "['proof']"
    assert reasoner.calls == []
    assert reasoner.query_calls == [(query, None)]


def test_legacy_query_support_statements_are_non_authoritative(
    monkeypatch, fake_vector_store
):
    support = "(: helper (Helper camera) (STV 1.0 1.0))"
    query = "(: $prf (Answer camera) $tv)"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(statements=[support], queries=[query])

    reasoner = RecordingReasoner()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    service._reason("What is the answer?")

    assert reasoner.calls == []
    assert reasoner.query_calls == [(query, None)]


def test_explicitly_trusted_query_adapters_are_executed_but_not_persisted(
    monkeypatch, fake_vector_store
):
    adapter = "(: adapter (Equivalent camera device) (STV 1.0 1.0))"
    query = "(: $prf (Answer camera) $tv)"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=[query], trusted_transient_statements=[adapter]
            )

    reasoner = RecordingReasoner()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    service._reason("What is the answer?")

    assert reasoner.calls == []
    assert reasoner.query_calls == [(query, [adapter])]


def test_selected_counterfactual_candidate_reports_its_temporal_plan(
    monkeypatch, fake_vector_store
):
    support = "(: stage7_rain_wet (Wet ground) (STV 0.4 0.4))"
    query = "(: $prf (Wet ground) $tv)"
    plan = {
        "query": query,
        "branch_id": "rain",
        "validity_interval_id": None,
        "decisions": [{"allowed": True, "reason": "allowed"}],
    }

    class CounterfactualParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=[query],
                candidate_trusted_transient_statements=[[support]],
                diagnostics={"candidate_temporal_plans": [plan]},
            )

    reasoner = RecordingReasoner()
    service = query_service(CounterfactualParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("In the rain branch, is the ground wet?")

    assert reasoner.query_calls == [(query, [support])]
    assert response.senf["executed_temporal_plan"] == plan


def test_retry_reports_the_retry_temporal_plan(monkeypatch, fake_vector_store):
    first = "(: $prf (First answer) $tv)"
    retry_query = "(: $prf (Retry answer) $tv)"
    first_plan = {"query": first, "branch_id": "first"}
    retry_plan = {"query": retry_query, "branch_id": "retry"}

    class RetryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=[first], diagnostics={"candidate_temporal_plans": [first_plan]}
            )

        def retry_parse_query(self, text, context, attempted_query):
            return ParseResult(
                queries=[retry_query],
                diagnostics={"candidate_temporal_plans": [retry_plan]},
            )

    class RetryReasoner(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == retry_query else []

    service = query_service(RetryParser([]), fake_vector_store, RetryReasoner())
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("Retry the answer.")

    assert response.retry_used is True
    assert response.senf["executed_temporal_plan"] == retry_plan


def test_malformed_transient_context_fails_closed_then_retries_ordinary_query(
    monkeypatch, fake_vector_store
):
    first_query = "(: $prf (Answer camera) $tv)"
    retry_query = "(: $prf (Known camera) $tv)"

    class RetryingParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=[first_query],
                trusted_transient_statements=["(Malformed camera)"],
            )

        def retry_parse_query(self, text, context, attempted_query):
            return ParseResult(queries=[retry_query])

    class RetrySucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == retry_query else []

    reasoner = RetrySucceeds()
    service = query_service(RetryingParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.retry_used is True
    assert reasoner.calls == []
    assert reasoner.query_calls == [(first_query, None), (retry_query, None)]


def test_hybrid_primary_and_fallback_query_support_remains_transient(monkeypatch):
    primary_support = "(: p (Primary camera) (STV 1.0 1.0))"
    fallback_support = "(: f (Fallback camera) (STV 1.0 1.0))"
    parser = CanonicalLangExtractParser.__new__(CanonicalLangExtractParser)
    parser._primary = SimpleNamespace(parse_query=lambda text, context: ParseResult(
        statements=[primary_support], queries=["primary"],
    ))
    parser._fallback = SimpleNamespace(parse_query=lambda text, context: ParseResult(
        transient_statements=[fallback_support], queries=["fallback"],
    ))
    monkeypatch.setattr(
        "parsers.canonical_langextract_parser.get_settings",
        lambda: SimpleNamespace(hybrid_query_mode="langextract_first"),
    )

    result = parser.parse_query("question", [])

    assert result.statements == []
    assert result.queries == ["primary", "fallback"]
    assert result.transient_statements == []
    assert result.candidate_transient_statements == [
        [primary_support], [fallback_support]
    ]


def test_hybrid_retry_propagates_legacy_and_current_transient_support(monkeypatch):
    legacy = "(: p (Primary camera) (STV 1.0 1.0))"
    current = "(: q (Question camera) (STV 1.0 1.0))"
    parser = CanonicalLangExtractParser.__new__(CanonicalLangExtractParser)
    parser._primary = SimpleNamespace(parse_query=lambda text, context: ParseResult(
        statements=[legacy], transient_statements=[current], queries=["primary"],
    ))
    monkeypatch.setattr(
        "parsers.canonical_langextract_parser.get_settings",
        lambda: SimpleNamespace(hybrid_query_mode="canonical_only"),
    )

    result = parser.retry_parse_query("question", [], "canonical")

    assert result.statements == []
    assert result.queries == ["primary"]
    assert result.transient_statements == []
    assert result.candidate_transient_statements == [[current, legacy]]


def test_hybrid_query_support_cannot_self_prove(monkeypatch, fake_vector_store):
    query = "(: $prf (Answer camera) $tv)"
    self_proof = "(: invented (Answer camera) (STV 1.0 1.0))"
    parser = CanonicalLangExtractParser.__new__(CanonicalLangExtractParser)
    parser._primary = SimpleNamespace(
        parse_query=lambda text, context: ParseResult(
            statements=[self_proof], queries=[query]
        )
    )
    parser._fallback = SimpleNamespace(
        parse_query=lambda text, context: ParseResult(queries=[])
    )
    monkeypatch.setattr(
        "parsers.canonical_langextract_parser.get_settings",
        lambda: SimpleNamespace(hybrid_query_mode="langextract_first"),
    )

    class WouldSelfProve(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["self-proof"] if transient_statements else []

    reasoner = WouldSelfProve()
    service = query_service(parser, fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("Is camera an answer?")

    assert response.proof == "[]"
    assert reasoner.calls == []
    assert reasoner.query_calls == [(query, None)]


def test_candidate_transients_do_not_cross_contaminate(monkeypatch, fake_vector_store):
    common = "(: common (Common camera) (STV 1.0 1.0))"
    first = "(: first (First camera) (STV 1.0 1.0))"
    second = "(: second (Second camera) (STV 1.0 1.0))"
    queries = ["(: $prf (Answer first) $tv)", "(: $prf (Answer second) $tv)"]

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=queries,
                trusted_transient_statements=[common],
                candidate_trusted_transient_statements=[[first], [second]],
            )

    class SecondSucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == queries[1] else []

    reasoner = SecondSucceeds()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    service._reason("What is the answer?")

    assert reasoner.query_calls == [
        (queries[0], [common, first]),
        (queries[1], [common, second]),
    ]


def test_malformed_candidate_transient_does_not_suppress_clean_fallback(
    monkeypatch, fake_vector_store
):
    queries = ["(: $prf (Wrong camera) $tv)", "(: $prf (Known camera) $tv)"]

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=queries,
                candidate_trusted_transient_statements=[["(Malformed camera)"], []],
            )

    class CleanFallbackSucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == queries[1] else []

    reasoner = CleanFallbackSucceeds()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.executed_query == queries[1]
    assert reasoner.query_calls == [(queries[0], None), (queries[1], None)]


def test_query_generated_fact_cannot_prove_its_own_candidate(
    monkeypatch, fake_vector_store
):
    query = "(: $prf (Answer camera) $tv)"
    self_proof = "(: invented (Answer camera) (STV 1.0 1.0))"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                statements=[self_proof],
                transient_statements=[self_proof],
                queries=[query],
                candidate_transient_statements=[[self_proof]],
            )

    class WouldSelfProve(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["self-proof"] if transient_statements else []

    reasoner = WouldSelfProve()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("Is camera an answer?")

    assert response.proof == "[]"
    assert reasoner.calls == []
    assert reasoner.query_calls == [(query, None)]


def test_original_query_provenance_survives_rewritten_candidate_execution(
    monkeypatch, fake_vector_store
):
    canonical = "(: $prf (HasProperty it expensive) $tv)"
    rewritten = "(: $prf (HasProperty camera expensive) $tv)"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=[rewritten, canonical], original_query=canonical)

    reasoner = RecordingReasoner()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("Is it expensive?")

    assert response.original_query == canonical
    assert response.executed_query == rewritten
    assert response.fallback_used is True


def test_ranked_first_only_does_not_traverse_later_candidates(
    monkeypatch, fake_vector_store
):
    queries = ["(: $prf (Wrong camera) $tv)", "(: $prf (Known camera) $tv)"]

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=queries, original_query=queries[1])

    class SecondSucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == queries[1] else []

    reasoner = SecondSucceeds()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    service._query_execution_policy = "ranked_first_only"
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.proof == "[]"
    assert response.candidate_count == 2
    assert response.candidate_count_tried == 1
    assert response.attempted_candidate_indices == [0]
    assert response.successful_candidate_index is None
    assert reasoner.query_calls == [(queries[0], None)]


def test_ranked_first_proof_records_actual_attempts(monkeypatch, fake_vector_store):
    queries = ["(: $prf (Wrong camera) $tv)", "(: $prf (Known camera) $tv)"]

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=queries, original_query=queries[1])

    class SecondSucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == queries[1] else []

    service = query_service(QueryParser([]), fake_vector_store, SecondSucceeds())
    service._query_execution_policy = "ranked_first_proof"
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.candidate_count_tried == 2
    assert response.attempted_candidate_indices == [0, 1]
    assert response.successful_candidate_index == 1


def test_original_only_uses_matching_candidate_transients(monkeypatch, fake_vector_store):
    rewritten = "(: $prf (Wrong camera) $tv)"
    original = "(: $prf (Known camera) $tv)"
    original_adapter = "(: known (Known camera) (STV 1.0 1.0))"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(
                queries=[rewritten, original],
                original_query=original,
                candidate_trusted_transient_statements=[[], [original_adapter]],
            )

    class OriginalSucceeds(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == original else []

    reasoner = OriginalSucceeds()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    service._query_execution_policy = "original_only"
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.executed_query == original
    assert response.executed_candidate_index == 1
    assert response.successful_candidate_index == 1
    assert response.attempted_queries == [original]
    assert reasoner.query_calls == [(original, [original_adapter])]


def test_original_only_does_not_execute_an_excluded_original(
    monkeypatch, fake_vector_store
):
    ranked = "(: $prf (Known camera) $tv)"
    excluded = "(: $prf (Excluded camera) $tv)"

    class QueryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=[ranked], original_query=excluded)

    reasoner = RecordingReasoner()
    service = query_service(QueryParser([]), fake_vector_store, reasoner)
    service._query_execution_policy = "original_only"
    monkeypatch.setattr("core.service.get_settings", query_settings)

    response = service._reason("What is known?")

    assert response.candidate_count_tried == 0
    assert response.attempted_queries == []
    assert response.proof == "[]"
    assert reasoner.query_calls == []


def test_legacy_disabled_fallback_still_allows_one_retry(
    monkeypatch, fake_vector_store
):
    first = "(: $prf (First answer) $tv)"
    retry_query = "(: $prf (Retry answer) $tv)"
    later_retry = "(: $prf (Later retry answer) $tv)"

    class RetryParser(FixedParser):
        def parse_query(self, text, context):
            return ParseResult(queries=[first])

        def retry_parse_query(self, text, context, attempted_query):
            return ParseResult(queries=[retry_query, later_retry])

    class RetryReasoner(RecordingReasoner):
        def query(self, query, transient_statements=None):
            self.query_calls.append((query, transient_statements))
            return ["proof"] if query == retry_query else []

    service = query_service(RetryParser([]), fake_vector_store, RetryReasoner())
    service._query_execution_policy = "ranked_first_only"
    service._legacy_retry_first_candidate = True
    monkeypatch.setattr(
        "core.service.get_settings",
        lambda: SimpleNamespace(
            query_candidate_max_tries=1,
            source_lookup_max_atoms=0,
            answer_generation_enabled=False,
        ),
    )

    response = service._reason("Retry the answer.")

    assert response.retry_used is True
    assert response.proof != "[]"
    assert response.attempted_queries == [first, retry_query]


def test_canonical_planner_records_candidate_before_subclass_rewrite():
    canonical = "(: $prf (HasProperty it expensive) $tv)"
    rewritten = "(: $prf (HasProperty camera expensive) $tv)"

    class RewritingParser(CanonicalPLNParser):
        def _plan_queries(self, question, queries, statements, context):
            return [rewritten]

    parser = RewritingParser.__new__(RewritingParser)

    planned, original = parser._plan_with_provenance(
        "Is it expensive?", [canonical], [], []
    )

    assert planned == [rewritten]
    assert original == canonical
