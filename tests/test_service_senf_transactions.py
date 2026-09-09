from core.parser import ParseResult, SemanticParser
from core.senf.types import SENF_PAYLOAD_KEY, senf_from_payload
from core.service import PLNRAGService
from parsers.canonical_pln_parser import CanonicalPLNParser
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

    def add_statements_report(self, statements):
        self.calls.append(list(statements))
        if self.accepted is None:
            return list(statements), []
        return list(self.accepted), []


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


def test_vector_failure_does_not_split_reasoner_and_parser_state():
    parser = FixedParser([CAMERA])
    reasoner = RecordingReasoner()
    service = service_for(parser, FailingStore(), reasoner=reasoner)

    result = service._ingest_single("The camera has a wide lens.")

    assert result.status == "failed"
    assert reasoner.calls == [[CAMERA]]
    assert parser.committed == [{"accepted": [CAMERA]}]


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

    _, queries = parser._post_filter_hook(
        ["Is it expensive?"],
        [PRONOUN],
        ["(: $prf (HasProperty it expensive) $tv)"],
        [],
        True,
    )
    assert queries == ["(: $prf (HasProperty camera expensive) $tv)"]

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

    parser._post_filter_hook(
        ["Does the camera have a wide lens?"], [], [query], [], True
    )
    first_sources = {pair.source_frame_id for pair in parser._weave.pairs}
    parser._post_filter_hook(
        ["Does the camera still have a wide lens?"], [], [query], [], True
    )

    assert parser._weave is not None
    assert {pair.source_frame_id for pair in parser._weave.pairs} == first_sources
