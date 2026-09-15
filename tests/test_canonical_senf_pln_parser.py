import pytest

from core.parser import ParseResult
from core.senf.exemplars import score_exemplars
from core.senf.extractor import extract_senf
from core.senf.identity import IdentityGraph
from core.senf.types import SENF_PAYLOAD_KEY, senf_to_payload
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


CAMERA = "(: a (HasProperty camera wide_lens) (STV 1.0 1.0))"
PRONOUN = "(: b (HasProperty it expensive) (STV 1.0 1.0))"


class RecordingStore:
    def __init__(self, records=None, raises=None):
        self._records = records or []
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    def retrieve_senf_context(self, text: str, top_k: int) -> list[dict]:
        self.calls.append((text, top_k))
        if self._raises:
            raise self._raises
        return self._records


def retrieval_record(senf, text, atoms):
    return {SENF_PAYLOAD_KEY: senf_to_payload(senf), "nl": text, "pln": atoms}


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    made = CanonicalSENFPLNParser()
    made._use_vector_context = False
    made._vector_store = None
    return made


def hook(parser, text, statements, queries=None, is_query=False):
    filtered = parser._post_filter_hook(
        [text], list(statements), list(queries or []), [], is_query
    )
    if not is_query:
        result = ParseResult(statements=filtered[0], parser_state=parser._pending_ingest)
        parser.prepare_ingest(result, filtered[0])
        parser.commit_ingest(result)
        parser._pending_ingest = None
    return filtered


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


class TestIdentityTransport:
    def test_ingest_preserves_the_accepted_pronoun_atom(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, _ = hook(parser, "It is expensive.", [PRONOUN])
        assert statements == [PRONOUN]

    def test_query_identity_transport_is_transient(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser,
            "Is it expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )
        assert statements == [PRONOUN]
        assert queries == ["(: $prf (HasProperty it expensive) $tv)"]

        planned = parser._plan_queries(
            "Is it expensive?", queries, statements, [CAMERA]
        )
        assert "(: $prf (HasProperty it expensive) $tv)" in planned
        assert "(: $prf (HasProperty camera expensive) $tv)" in planned

    def test_ingest_never_emits_identity_bridges(self, parser):
        parser._emit_bridge_atoms = True
        hook(parser, "The camera has a wide lens.", [CAMERA])

        statements, _ = hook(parser, "It is expensive.", [PRONOUN])

        assert statements == [PRONOUN]
        assert parser.senf_telemetry()["bridge_atom_count"] == 0

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

    def test_query_rendering_does_not_rewrite_token_collisions(self, parser):
        senf = extract_senf(
            "q1", "Does it follow an itinerary?",
            ["(: proof_7 (Follows it itinerary) $tv)"],
        )
        it_ref = senf.frames[0].roles[0].filler
        graph = IdentityGraph(
            representatives={it_ref.entity_id: "prior:e0"},
            mention_entities={it_ref.mention_id: it_ref.entity_id},
            entity_symbols={it_ref.entity_id: "it", "prior:e0": "camera"},
        )

        rendered = parser._render_query(
            "(: proof_7 (Follows it itinerary) $tv)", senf, graph
        )

        assert rendered == "(: proof_7 (Follows camera itinerary) $tv)"

    def test_query_rendering_distinguishes_equal_symbols_by_entity_ref(self, parser):
        senf = extract_senf(
            "q1", "one camera beside another camera",
            ["(: proof_8 (Beside camera camera) $tv)"],
        )
        left, right = [role.filler for role in senf.frames[0].roles]
        graph = IdentityGraph(
            representatives={left.entity_id: "prior:e0"},
            mention_entities={
                left.mention_id: left.entity_id,
                right.mention_id: right.entity_id,
            },
            entity_symbols={
                left.entity_id: "camera",
                right.entity_id: "camera",
                "prior:e0": "security_camera",
            },
        )

        rendered = parser._render_query(
            "(: proof_8 (Beside camera camera) $tv)", senf, graph
        )

        assert rendered == "(: proof_8 (Beside security_camera camera) $tv)"

    def test_query_rendering_preserves_kind_value_head_and_atom_id(self, parser):
        senf = extract_senf(
            "q1", "Is the camera a device?",
            ["(: proof_9 (IsA camera device) $tv)"],
        )
        ref = senf.frames[0].roles[0].filler
        graph = IdentityGraph(
            mention_entities={ref.mention_id: ref.entity_id},
            entity_symbols={ref.entity_id: "camera"},
        )

        assert parser._render_query(
            "(: proof_9 (IsA camera device) $tv)", senf, graph
        ) == "(: proof_9 (IsA camera device) $tv)"

    def test_multiple_distinct_query_atom_ids_render_their_own_frames(self, parser):
        queries = ["(: p1 (Sees it camera) $tv)", "(: p2 (Uses it lens) $tv)"]
        senf = extract_senf("q1", "Does it see a camera and use a lens?", queries)
        pronoun = next(mention for mention in senf.mentions if mention.canonical_symbol == "it")
        graph = IdentityGraph(
            representatives={pronoun.entity_id: "prior:e0"},
            mention_entities={pronoun.mention_id: pronoun.entity_id},
            entity_symbols={
                **{entity.entity_id: entity.canonical_symbol for entity in senf.entities},
                "prior:e0": "robot",
            },
        )

        assert parser._render_queries(queries, senf, graph) == [
            "(: p1 (Sees robot camera) $tv)",
            "(: p2 (Uses robot lens) $tv)",
        ]

    def test_repeated_query_atom_id_renders_each_occurrence(self, parser):
        queries = [
            "(: $prf (Sees it camera) $tv)",
            "(: $prf (Uses it lens) $tv)",
        ]
        senf = extract_senf("q1", "Does it see a camera and use a lens?", queries)
        graph = IdentityGraph(
            representatives={entity.entity_id: "prior:e0" for entity in senf.entities if entity.canonical_symbol == "it"},
            mention_entities={
                mention.mention_id: mention.entity_id for mention in senf.mentions
            },
            entity_symbols={
                **{entity.entity_id: entity.canonical_symbol for entity in senf.entities},
                "prior:e0": "robot",
            },
        )

        assert parser._render_queries(queries, senf, graph) == [
            "(: $prf (Sees robot camera) $tv)",
            "(: $prf (Uses robot lens) $tv)",
        ]

    def test_query_rendering_preserves_nested_negation(self, parser):
        query = "(: proof_10 (Believes alex (Not (Trusted sam))) $tv)"
        senf = extract_senf("q1", "Alex does not trust Sam.", [query])
        graph = IdentityGraph(
            mention_entities={
                mention.mention_id: mention.entity_id for mention in senf.mentions
            },
            entity_symbols={
                entity.entity_id: entity.canonical_symbol for entity in senf.entities
            },
        )

        assert parser._render_query(query, senf, graph) == query

    def test_query_rendering_preserves_outer_negation_scope(self, parser):
        query = "(: proof_10 (Not (Believes alex (Trusted sam))) $tv)"
        senf = extract_senf("q1", "Alex does not believe Sam is trusted.", [query])
        graph = IdentityGraph(
            mention_entities={
                mention.mention_id: mention.entity_id for mention in senf.mentions
            },
            entity_symbols={
                entity.entity_id: entity.canonical_symbol for entity in senf.entities
            },
        )

        assert parser._render_query(query, senf, graph) == query

    def test_query_rendering_fails_closed_for_logical_wrappers(self, parser):
        query = (
            "(: proof_11 (Implication (Premises (Sees alex sam)) "
            "(Conclusions (Knows alex sam))) $tv)"
        )
        senf = extract_senf("q1", "If Alex sees Sam, Alex knows Sam.", [query])

        assert parser._render_query(query, senf, IdentityGraph()) == query

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
        made._vector_store = RecordingStore(records=[retrieval_record(
            prior, "The camera has a wide lens.", [CAMERA]
        )])
        statements, queries = hook(
            made,
            "Is it, the camera, expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )
        planned = made._plan_queries(
            "Is it, the camera, expensive?", queries, statements, [CAMERA]
        )
        assert any("camera" in query for query in planned)
        assert "(: $prf (HasProperty it expensive) $tv)" in planned

    def test_parser_metadata_survives_recreation(self, monkeypatch, fake_vector_store):
        monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
        first = CanonicalSENFPLNParser()
        first._use_vector_context = False
        hook(first, "The camera has a wide lens.", [CAMERA])
        fake_vector_store.store(
            "The camera has a wide lens.",
            [CAMERA],
            fake_vector_store.embed("The camera has a wide lens."),
            metadata={SENF_PAYLOAD_KEY: senf_to_payload(first._session[-1])},
        )

        recreated = CanonicalSENFPLNParser()
        recreated._vector_store = fake_vector_store
        statements, queries = hook(
            recreated,
            "Is it, the camera, expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )

        planned = recreated._plan_queries(
            "Is it, the camera, expensive?", queries, statements, [CAMERA]
        )
        assert any("camera" in query for query in planned)

    @pytest.mark.parametrize("forgery", ["source_text", "source_atom_id"])
    def test_recalled_senf_must_match_stored_nl_and_accepted_atom_ids(
        self, parser, forgery
    ):
        text = "The camera has a wide lens."
        prior = extract_senf("s1", text, [CAMERA])
        payload = senf_to_payload(prior)
        if forgery == "source_text":
            payload["frames"][0]["source_text"] = "Forged source."
            record = {SENF_PAYLOAD_KEY: payload, "nl": text, "pln": [CAMERA]}
        else:
            record = {SENF_PAYLOAD_KEY: payload, "nl": text, "pln": [
                "(: other (HasProperty camera wide_lens) (STV 1 1))"
            ]}

        assert parser._validated_recalled_senf(record) is None

    def test_recalled_senf_rejects_frame_body_forged_under_an_accepted_id(self, parser):
        prior = extract_senf("s1", "The camera is at the lab.", [
            "(: src (AtLocation camera lab) (STV 1 1))"
        ])
        forged = extract_senf("s1", "The camera is at the lab.", [
            "(: src (IsA camera device) (STV 1 1))"
        ])
        record = {
            "nl": "The camera is at the lab.",
            "pln": ["(: src (AtLocation camera lab) (STV 1 1))"],
            SENF_PAYLOAD_KEY: senf_to_payload(forged),
        }

        assert prior.frames[0].source_atom_id == forged.frames[0].source_atom_id
        assert parser._validated_recalled_senf(record) is None

    def test_recalled_exemplar_annotations_are_recomputed(self, parser):
        text = "The Nikon camera arrived."
        atom = "(: arrived (Arrived camera) (STV 1 1))"
        prior = score_exemplars(extract_senf("s1", text, [atom]))
        payload = senf_to_payload(prior)
        mention_id = next(iter(payload["exemplar_scores"]))
        payload["exemplar_scores"][mention_id][0]["distance"] = 0.99
        record = {SENF_PAYLOAD_KEY: payload, "nl": text, "pln": [atom]}

        recalled = parser._validated_recalled_senf(record)

        assert recalled is not None
        assert recalled.exemplar_scores[mention_id][0].distance != 0.99

    def test_session_senfs_do_not_require_persisted_sibling_validation(self, parser):
        trusted = extract_senf("s1", "The camera arrived.", [
            "(: arrived (Arrived camera) (STV 1 1))"
        ])
        parser._session = [trusted]

        assert parser._prior_senfs("Did the camera arrive?", is_query=True) == [trusted]

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
        statements, queries = hook(
            parser,
            "Does the camera have a wide lens?",
            [CAMERA],
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
            is_query=True,
        )
        parser._plan_queries(
            "Does the camera have a wide lens?", queries, statements, [CAMERA]
        )

        assert parser._weave is not None
        assert "camera" in parser._weave.grounded_symbols

    def test_the_weave_does_not_leak_into_the_next_question(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser, "Is it expensive?", [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"], is_query=True,
        )
        parser._plan_queries("Is it expensive?", queries, statements, [CAMERA])
        first = parser._weave

        hook(parser, "Nothing at all.", [], [], is_query=True)

        assert parser._weave is not first

    def test_signals_carry_the_configured_weights(self, parser):
        from config import get_settings

        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser, "Does the camera have a wide lens?", [CAMERA],
            ["(: $prf (HasProperty camera wide_lens) $tv)"], is_query=True,
        )
        parser._plan_queries(
            "Does the camera have a wide lens?", queries, statements, [CAMERA]
        )
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

    def test_no_senf_evidence_preserves_base_candidate_order_exactly(self, parser):
        candidates = [
            "(: $prf (Smart kebede) $tv)",
            "(: $prf (Tall kebede) $tv)",
        ]
        context = [
            "(: smart (Smart kebede) (STV 1.0 1.0))",
            "(: tall (Tall kebede) (STV 1.0 1.0))",
        ]
        expected = CanonicalPLNParser._plan_queries(
            parser, "Is Kebede smart?", candidates, [], context
        )

        assert parser._plan_queries(
            "Is Kebede smart?", candidates, [], context
        ) == expected

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
        parser._query_prior = (source,)

        planned = parser._plan_queries(
            "Is the camera located in the lab?",
            ["(: $prf (LocatedIn camera lab) $tv)"],
            [],
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )

        assert planned[:2] == [
            "(: $prf (AtLocation camera lab) $tv)",
            "(: $prf (LocatedIn camera lab) $tv)",
        ]

    def test_query_planning_ranks_current_predicates_before_canonical_fallbacks(self, parser):
        source = extract_senf(
            "s2", "The camera has a wide lens.", [CAMERA]
        )
        query = extract_senf(
            "q2",
            "Does the camera have a wide lens?",
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
        )
        from core.senf.weave import build_weaves

        parser._query_prior = (source,)
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

        assert planned[0] == "(: $prf (HasProperty camera wide_lens) $tv)"
        assert "(: $prf (ImprovesOutcome tirzepatide) $tv)" in planned[1:]

    def test_candidate_plans_are_bounded_deduped_and_report_scores(self, parser):
        parser._candidate_limit = 2
        hook(parser, "The camera is at the lab.", [
            "(: a (AtLocation camera lab) (STV 1.0 1.0))"
        ])
        statements, queries = hook(
            parser,
            "Is the camera located in the lab?",
            [],
            ["(: $prf (LocatedIn camera lab) $tv)"],
            is_query=True,
        )

        planned = parser._plan_queries(
            "Is the camera located in the lab?", queries, statements,
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )

        assert len(planned) <= 2
        assert len(planned) == len(set(planned))
        assert "(: $prf (LocatedIn camera lab) $tv)" in planned
        assert "(: $prf (AtLocation camera lab) $tv)" in planned
        report = parser.senf_telemetry()
        assert len(report["candidate_score_breakdown"]) == len(planned)

    def test_senf_variants_do_not_displace_second_canonical_fallback(self, parser):
        parser._candidate_limit = 3
        hook(parser, "The camera is at the lab.", [
            "(: a (AtLocation camera lab) (STV 1.0 1.0))"
        ])
        canonical = [
            "(: $prf (LocatedIn it lab) $tv)",
            "(: $prf (AtLocation camera lab) $tv)",
        ]
        statements, queries = hook(
            parser, "Is it located in the lab?", [], canonical, is_query=True
        )

        planned = parser._plan_queries(
            "Is it located in the lab?", queries, statements,
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )

        assert canonical[0] in planned
        assert canonical[1] in planned
        assert planned.index(canonical[1]) < parser._candidate_limit

    def test_bridge_flag_disabled_produces_no_candidate_adapters(self, parser):
        parser._emit_bridge_atoms = False
        hook(parser, "The camera is at the lab.", [
            "(: a (AtLocation camera lab) (STV 1.0 1.0))"
        ])
        statements, queries = hook(
            parser,
            "Is it located in the lab?",
            [],
            ["(: $prf (LocatedIn it lab) $tv)"],
            is_query=True,
        )

        parser._plan_queries(
            "Is it located in the lab?", queries, statements, []
        )

        assert statements == []
        assert all(not plan.adapters for plan in parser._candidate_plans)
        assert parser.senf_telemetry()["bridge_atom_count"] == 0

        parser._emit_bridge_atoms = True
        parser._plan_queries(
            "Is it located in the lab?", queries, statements, []
        )
        assert any(plan.adapters for plan in parser._candidate_plans)
        assert parser.senf_telemetry()["bridge_atom_count"] > 0

    def test_global_prior_frame_and_mention_limits(self, parser):
        parser._max_priors = 2
        parser._max_source_frames = 2
        parser._max_mentions = 3
        prior = [
            extract_senf(
                f"s{index}",
                f"Camera {index} is in lab {index}.",
                [f"(: a{index} (AtLocation camera_{index} lab_{index}) (STV 1.0 1.0))"],
            )
            for index in range(4)
        ]

        bounded = parser._bound_prior_work(prior)

        assert len(bounded) <= 2
        assert sum(len(item.frames) for item in bounded) <= 2
        assert sum(len(item.mentions) for item in bounded) <= 3
        assert [item.senf_id for item in bounded] == ["senf:s3"]
        assert bounded[0] is prior[3]

    def test_prior_budget_preserves_recent_session_before_recalled_records(
        self, parser, monkeypatch
    ):
        parser._max_source_frames = 1
        parser._max_mentions = 2
        session = extract_senf(
            "session", "The camera arrived.",
            ["(: local (Arrived camera) (STV 1 1))"],
        )
        recalled = extract_senf(
            "recalled", "A lens cracked.",
            ["(: old (Cracked lens) (STV 1 1))"],
        )
        parser._session = [session]
        monkeypatch.setattr(parser, "_retrieve_senf_records", lambda _text: [{}])
        monkeypatch.setattr(parser, "_validated_recalled_senf", lambda _record: recalled)
        monkeypatch.setattr(parser, "_senf_overlaps_question", lambda *_args: True)

        assert parser._prior_senfs("Did it arrive?", is_query=True) == [session]

    @pytest.mark.parametrize("cue", ["another", "different"])
    def test_contrastive_ambiguity_keeps_the_canonical_fallback(self, parser, cue):
        hook(parser, "A camera arrived.", [
            "(: a (Arrived camera) (STV 1.0 1.0))"
        ])
        statements, queries = hook(
            parser,
            f"Did {cue} camera arrive?",
            [],
            [f"(: $prf (Arrived {cue}_camera) $tv)"],
            is_query=True,
        )

        planned = parser._plan_queries(
            f"Did {cue} camera arrive?", queries, statements,
            ["(: a (Arrived camera) (STV 1.0 1.0))"],
        )

        assert planned[-1] == f"(: $prf (Arrived {cue}_camera) $tv)"
        assert "(: $prf (Arrived camera) $tv)" not in planned[:-1]


class TestTelemetry:
    """Telemetry reports existing hook state without changing parser output."""

    def test_no_telemetry_before_anything_is_parsed(self, parser):
        assert parser.senf_telemetry() is None

    def test_a_question_reports_frames_and_weave(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser,
            "Does the camera have a wide lens?",
            [CAMERA],
            ["(: $prf (HasProperty camera wide_lens) $tv)"],
            is_query=True,
        )
        parser._plan_queries(
            "Does the camera have a wide lens?", queries, statements, [CAMERA]
        )
        report = parser.senf_telemetry()

        assert report["frame_count"] > 0
        assert report["weave_distortion"] is not None
        assert report["weave_pair_count"] >= 1

    def test_identity_merges_and_query_rewrites_are_reported_separately(self, parser):
        hook(parser, "The camera has a wide lens.", [CAMERA])
        statements, queries = hook(
            parser,
            "Is it expensive?",
            [PRONOUN],
            ["(: $prf (HasProperty it expensive) $tv)"],
            is_query=True,
        )
        parser._plan_queries("Is it expensive?", queries, statements, [CAMERA])
        report = parser.senf_telemetry()

        assert report["merge_count"] >= 1
        assert report["query_rewritten_atom_count"] >= 1
        assert report["query_rewritten_atom_count"] < report["candidate_count"]

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
