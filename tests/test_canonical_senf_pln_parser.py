import pytest

from core.senf.extractor import extract_senf
from core.senf.types import senf_to_payload
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


CAMERA = "(: a (HasProperty camera wide_lens) (STV 1.0 1.0))"
PRONOUN = "(: b (HasProperty it expensive) (STV 1.0 1.0))"


class RecordingStore:
    def __init__(self, blobs=None, raises=None):
        self._blobs = blobs or []
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    def retrieve_senf_context(self, text: str, top_k: int) -> list[dict]:
        self.calls.append((text, top_k))
        if self._raises:
            raise self._raises
        return self._blobs


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    made = CanonicalSENFPLNParser()
    made._use_vector_context = False
    made._vector_store = None
    return made


def hook(parser, text, statements, queries=None, is_query=False):
    return parser._post_filter_hook(
        [text], list(statements), list(queries or []), [], is_query
    )


class TestSeamIsIdentityByDefault:
    def test_base_parser_hook_returns_its_inputs_unchanged(self):
        base = CanonicalPLNParser.__new__(CanonicalPLNParser)
        statements, queries = base._post_filter_hook(
            ["text"], [CAMERA], ["(: $prf (Smart kebede) $tv)"], [], False
        )
        assert statements == [CAMERA]
        assert queries == ["(: $prf (Smart kebede) $tv)"]

    def test_base_parser_hook_does_not_copy_or_reorder(self):
        base = CanonicalPLNParser.__new__(CanonicalPLNParser)
        given = [CAMERA, PRONOUN]
        statements, _ = base._post_filter_hook(["t"], given, [], [], False)
        assert statements is given


class TestSymbolRewriting:
    def test_pronoun_is_rewritten_to_its_antecedent(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, _ = hook(parser, "It is expensive.", [PRONOUN])
        assert statements == ["(: b (HasProperty camera expensive) (STV 1.0 1.0))"]

    def test_queries_are_rewritten_with_the_same_map(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser,
            "Is it expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )
        assert statements == ["(: b (HasProperty camera expensive) (STV 1.0 1.0))"]
        assert queries == ["(: $prf (HasProperty camera expensive) $tv)"]

    def test_variables_are_never_rewritten(self, parser):
        _, queries = hook(
            parser,
            "The camera has a wide lens.",
            [CAMERA],
            ["(: $prf (HasProperty $camera wide_lens) $tv)"],
            is_query=True,
        )
        assert "$prf" in queries[0]
        assert "$camera" in queries[0]

    def test_predicate_heads_and_truth_values_survive(self, parser):
        statements, _ = hook(parser, "The camera has a wide lens.", [CAMERA])
        assert "HasProperty" in statements[0]
        assert "(STV 1.0 1.0)" in statements[0]

    def test_unrelated_sentences_are_left_alone(self, parser):
        given = ["(: a (Eats kebede fish) (STV 1.0 1.0))"]
        statements, _ = hook(parser, "Kebede eats fish.", given)
        assert statements == given

    def test_empty_input_short_circuits(self, parser):
        assert hook(parser, "Nothing here.", [], []) == ([], [])


class TestSessionState:
    def test_reset_clears_the_antecedent(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        parser.reset()
        statements, _ = hook(parser, "It is expensive.", [PRONOUN])
        assert "camera" not in statements[0]

    def test_reset_restarts_sentence_numbering(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        parser.reset()
        assert parser._sentence_counter == 0

    def test_a_query_does_not_join_the_session(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA], is_query=True)
        assert parser._session == []

    def test_session_is_capped_by_frame_count(self, parser):
        parser._max_frames = 2
        for index in range(6):
            hook(parser, f"Sentence {index}.", [f"(: a{index} (P e{index}) (STV 1.0 1.0))"])
        assert sum(len(senf.frames) for senf in parser._session) <= 2

    def test_a_sentence_larger_than_the_cap_is_still_kept(self, parser):
        parser._max_frames = 1
        hook(
            parser,
            "Kebede eats fish and is smart.",
            [
                "(: a (Eats kebede fish) (STV 1.0 1.0))",
                "(: b (Smart kebede) (STV 1.0 1.0))",
            ],
        )
        assert len(parser._session) == 1

    def test_the_most_recent_sentence_is_the_one_kept(self, parser):
        parser._max_frames = 1
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(parser, "Kebede eats fish.", ["(: b (Eats kebede fish) (STV 1.0 1.0))"])
        assert parser._session[-1].sentence_id == "s2"


class TestVectorContext:
    def test_stored_senf_supplies_an_antecedent_across_sessions(self, monkeypatch):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        prior = extract_senf("s1", "The camera has a wide lens.", [CAMERA])
        made = CanonicalSENFPLNParser()
        made._vector_store = RecordingStore(blobs=[senf_to_payload(prior)])
        statements, _ = hook(made, "It is expensive.", [PRONOUN])
        assert "camera" in statements[0]

    def test_retrieval_failure_is_fail_open(self, monkeypatch):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        made = CanonicalSENFPLNParser()
        made._vector_store = RecordingStore(raises=RuntimeError("qdrant down"))
        given = [CAMERA]
        statements, _ = hook(made, "The camera has a wide lens.", given)
        assert statements == given

    def test_disabling_the_setting_skips_the_store(self, monkeypatch):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        made = CanonicalSENFPLNParser()
        made._use_vector_context = False
        store = RecordingStore()
        made._vector_store = store
        hook(made, "The camera has a wide lens.", [CAMERA])
        assert store.calls == []


class TestSettings:
    def test_config_default_tracks_the_resolver_default(self):
        """config.py restates 0.75 rather than importing it; drift would be silent."""
        from config import Settings
        from core.senf.identity import DEFAULT_IDENTITY_THRESHOLD

        field = Settings.model_fields["senf_identity_threshold"]
        assert field.default == DEFAULT_IDENTITY_THRESHOLD

    def test_the_parser_reads_its_knobs_from_settings(self, parser):
        from config import get_settings

        cfg = get_settings()
        assert parser._threshold == cfg.senf_identity_threshold
        assert parser._context_top_k == cfg.senf_context_top_k
        assert parser._max_frames == cfg.senf_session_max_frames


class TestFactoryRegistration:
    def test_benchmark_factory_resolves_the_name(self):
        import benchmark_parsers as bp

        assert bp._get_parser_factory("canonical_senf_pln") is CanonicalSENFPLNParser

    def test_unknown_parser_message_lists_the_new_name(self, monkeypatch):
        import parsers

        monkeypatch.setattr(parsers, "get_settings", lambda: type("S", (), {"parser": "nope"})())
        with pytest.raises(ValueError, match="canonical_senf_pln"):
            parsers.get_parser()
