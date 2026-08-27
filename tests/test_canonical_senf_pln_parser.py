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


class TestBaseParserCompatibility:
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
        assert statements == [
            "(: b (HasProperty camera expensive) (STV 0.951229 0.952381))"
        ]

    def test_queries_are_rewritten_with_the_same_map(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser,
            "Is it expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )
        assert statements == [
            "(: b (HasProperty camera expensive) (STV 0.951229 0.952381))"
        ]
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

    def test_ungrounded_base_output_is_not_admitted_to_senf(self, parser):
        hook(
            parser,
            "The camera has a wide lens.",
            [CAMERA, "(: noise (Invented ghost_entity) (STV 1.0 1.0))"],
        )

        report = parser.senf_telemetry()
        assert report["frame_count"] == 1
        assert report["mention_count"] == 2


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
        assert parser._session[-1].sentence_id.endswith(":s2")


class TestVectorContext:
    def test_stored_senf_supplies_an_antecedent_across_sessions(self, monkeypatch):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        prior = extract_senf("s1", "The camera has a wide lens.", [CAMERA])
        made = CanonicalSENFPLNParser()
        made._vector_store = RecordingStore(blobs=[senf_to_payload(prior)])
        statements, _ = hook(made, "It is expensive.", [PRONOUN])
        assert "camera" in statements[0]

    def test_parser_metadata_survives_recreation(self, monkeypatch, fake_vector_store):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        first = CanonicalSENFPLNParser()
        first._use_vector_context = False
        hook(first, "The camera has a wide lens.", [CAMERA])
        fake_vector_store.store(
            "The camera has a wide lens.",
            [CAMERA],
            fake_vector_store.embed("The camera has a wide lens."),
            metadata=first.storage_metadata(),
        )

        recreated = CanonicalSENFPLNParser()
        recreated._vector_store = fake_vector_store
        statements, _ = hook(recreated, "It is expensive.", [PRONOUN])

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


class TestWeaveScoring:
    def test_no_weave_means_no_senf_signals(self, parser):
        """Ingest never builds a weave, so scoring must stay pre-SENF there."""
        assert parser._senf_signals() is None

    def test_a_question_builds_a_weave_against_prior_sentences(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(
            parser,
            "Does the camera have a wide lens?",
            [CAMERA],
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
            is_query=True,
        )

        assert parser._weave is not None
        assert "camera" in parser._weave.grounded_symbols

    def test_the_weave_does_not_leak_into_the_next_question(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(parser, "Is it expensive?", [PRONOUN], is_query=True)
        first = parser._weave

        hook(parser, "Nothing at all.", [], [], is_query=True)

        assert parser._weave is not first

    def test_signals_carry_the_configured_weights(self, parser):
        from config import get_settings

        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(parser, "Does the camera have a wide lens?", [CAMERA], is_query=True)
        signals = parser._senf_signals()
        cfg = get_settings()

        assert signals.source_grounding_weight == cfg.senf_source_grounding_weight
        assert signals.role_compat_weight == cfg.senf_role_compat_weight

    def test_scoring_without_a_weave_matches_the_base_parser(self, parser):
        from core import query_scoring

        query = {"head": "Smart", "arity": 1, "args": ["kebede"], "variables": []}
        facts = [{"head": "Smart", "arity": 1, "args": ["kebede"]}]

        assert parser._score_query_candidate(
            query, facts, [], True
        ) == query_scoring.score_query_candidate(query, facts, [], True)

    def test_query_planning_uses_the_supported_predicate_family(self, parser):
        from core.senf.weave import build_weaves

        source = extract_senf(
            "s1",
            "The camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )
        query = extract_senf(
            "q1",
            "Is the camera located in the lab?",
            ["(: $prf (LocatedIn camera lab) $tv)"],
        )
        parser._weaves = build_weaves(query, [source])
        parser._weave = parser._weaves[0]

        planned = parser._plan_queries(
            "Is the camera located in the lab?",
            ["(: $prf (LocatedIn camera lab) $tv)"],
            [],
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )

        assert planned == ["(: $prf (AtLocation camera lab) $tv)"]

    def test_query_planning_rejects_predicates_from_an_older_case(self, parser):
        source = extract_senf(
            "s2", "The camera has a wide lens.", [CAMERA]
        )
        query = extract_senf(
            "q2",
            "Does the camera have a wide lens?",
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
        )
        from core.senf.weave import build_weaves

        parser._weaves = build_weaves(query, [source])
        parser._weave = parser._weaves[0]
        parser._query_source_heads = frozenset({"HasProperty"})

        planned = parser._plan_queries(
            "Does the camera have a wide lens?",
            [
                "(: $prf (ImprovesOutcome tirzepatide) $tv)",
                "(: $prf (HasProperty camera wide_lens) $tv)",
            ],
            [],
            [
                "(: old (ImprovesOutcome tirzepatide) (STV 1.0 1.0))",
                CAMERA,
            ],
        )

        assert planned == ["(: $prf (HasProperty camera wide_lens) $tv)"]


class TestTelemetry:
    """Telemetry reports existing hook state without changing parser output."""

    def test_no_telemetry_before_anything_is_parsed(self, parser):
        assert parser.senf_telemetry() is None

    def test_a_question_reports_frames_and_weave(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(
            parser,
            "Does the camera have a wide lens?",
            [CAMERA],
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
            is_query=True,
        )
        report = parser.senf_telemetry()

        assert report["frame_count"] > 0
        assert report["weave_distortion"] is not None
        assert report["weave_pair_count"] >= 1

    def test_a_merge_is_counted_and_attributed_to_atoms(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(parser, "It is expensive.", [PRONOUN])
        report = parser.senf_telemetry()

        # The two are reported separately because identity merging without any atom
        # changing is a distinct failure from identity finding nothing.
        assert report["merge_count"] >= 1
        assert report["rewritten_atom_count"] >= 1

    def test_ingest_does_not_report_the_previous_questions_weave(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        hook(parser, "Is it expensive?", [PRONOUN], is_query=True)
        hook(parser, "The camera has a wide lens.", [CAMERA])

        assert parser.senf_telemetry()["weave_distortion"] is None

    def test_reset_clears_telemetry(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        assert parser.senf_telemetry() is not None

        parser.reset()

        assert parser.senf_telemetry() is None

    def test_telemetry_is_a_copy_a_caller_cannot_corrupt(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        parser.senf_telemetry()["frame_count"] = 999

        assert parser.senf_telemetry()["frame_count"] != 999

    def test_the_base_parser_reports_nothing(self):
        assert not hasattr(CanonicalPLNParser, "senf_telemetry")


class TestFactoryRegistration:
    def test_benchmark_factory_resolves_the_name(self):
        import benchmark_parsers as bp

        assert bp._get_parser_factory("canonical_senf_pln") is CanonicalSENFPLNParser

    def test_unknown_parser_message_lists_the_new_name(self, monkeypatch):
        import parsers

        monkeypatch.setattr(parsers, "get_settings", lambda: type("S", (), {"parser": "nope"})())
        with pytest.raises(ValueError, match="canonical_senf_pln"):
            parsers.get_parser()
