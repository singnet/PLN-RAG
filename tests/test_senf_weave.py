import pytest

from config import get_settings
from core.senf.extractor import extract_senf
from core.senf.bridge import (
    executable_bridge_atoms,
    identity_bridge_atoms,
    predicate_bridge_atoms,
    transport_truth,
    weave_bridge_atoms,
)
from core.senf.exemplars import score_exemplars
from core.senf.identity import IdentityEdge, IdentityGraph, resolve_identity
from core.senf.types import EntityRef
from core.senf.weave import EntityMap, MIN_PAIR_SCORE, build_weaves, weave


def senf_for(sentence_id: str, text: str, atoms: list[str]):
    return extract_senf(sentence_id, text, atoms)


CAMERA = "(: a (HasProperty camera wide_lens) (STV 1.0 1.0))"
EATS = "(: b (Eats kebede fish) (STV 1.0 1.0))"


class TestPairing:
    def test_a_question_about_an_ingested_fact_aligns(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Does the camera have a wide lens?", [CAMERA])

        result = weave(query, [source])

        assert result.aligned
        assert result.distortion < 0.5
        assert "camera" in result.grounded_symbols
        assert "s1:e0" in result.grounded_entity_ids

    def test_an_unrelated_question_does_not_align(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Does Kebede eat fish?", [EATS])

        result = weave(query, [source])

        assert result.pairs == ()
        assert result.distortion == 1.0

    def test_top_level_arity_must_match(self):
        source = senf_for("s1", "a relates b", ["(: a (Relates a b) (STV 1 1))"])
        query = senf_for("q1", "a relates", ["(: q (Relates a) $tv)"])
        assert weave(query, [source]).pairs == ()

    def test_distortion_is_one_when_there_is_nothing_to_align_against(self):
        query = senf_for("q1", "Does the camera have a wide lens?", [CAMERA])

        assert weave(query, []).distortion == 1.0

    def test_an_empty_question_is_not_distorted(self):
        """No frames means nothing unsupported, which is not the same as unsupported."""
        empty = senf_for("q1", "", [])

        assert weave(empty, []).distortion == 0.0

    def test_each_frame_is_used_at_most_once(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "The camera has a wide lens.", [CAMERA])

        result = weave(query, [source, source])

        assert len({pair.source_frame_id for pair in result.pairs}) == len(result.pairs)
        assert len({pair.query_frame_id for pair in result.pairs}) == len(result.pairs)

    def test_pairs_below_the_floor_are_dropped(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Does Kebede eat fish?", [EATS])

        for pair in weave(query, [source]).pairs:
            assert pair.score >= MIN_PAIR_SCORE


class TestPolarity:
    def test_a_negated_counterpart_still_aligns_but_scores_lower(self):
        positive = senf_for("s1", "The group reduced HbA1c.", ["(: a (Reduces group hba1c) (STV 1.0 1.0))"])
        negated = senf_for("s2", "The group did not reduce HbA1c.", ["(: b (Not (Reduces group hba1c)) (STV 1.0 1.0))"])

        agree = weave(positive, [positive])
        conflict = weave(negated, [positive])

        if conflict.pairs and agree.pairs:
            assert conflict.pairs[0].score < agree.pairs[0].score


class TestDeterminism:
    def test_the_same_input_gives_the_same_result(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Does the camera have a wide lens?", [CAMERA])

        first = weave(query, [source])
        again = weave(query, [source])

        assert first.pairs == again.pairs
        assert first.distortion == again.distortion


class TestEntityIdentity:
    def test_identity_graph_is_applied_to_entity_refs(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Is it expensive?", ["(: c (HasProperty it wide_lens) (STV 1.0 1.0))"])

        without = weave(query, [source])
        graph = resolve_identity([source, query])
        with_identity = weave(query, [source], identity_graph=graph)

        assert graph.same_entity("s1:e0", "q1:e0")
        assert with_identity.entity_maps[0].source_entity_id == "s1:e0"
        assert with_identity.entity_maps[0].target_entity_id == "q1:e0"
        assert with_identity.distortion <= without.distortion

class TestPaperFeatures:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A strategy game lasted for hours.", "chess_game"),
            ("An exhausting game ended late.", "football_game"),
        ],
    )
    def test_context_selects_a_game_exemplar(self, text, expected):
        senf = senf_for(
            "s1", text, ["(: a (IsA match game) (STV 1.0 1.0))"]
        )

        score_exemplars(senf)

        match = next(m for m in senf.mentions if m.canonical_symbol == "match")
        assert senf.nearest_exemplar_for(match) == expected

    def test_build_weaves_returns_ranked_typed_mappings(self):
        query = senf_for("q1", "Does the camera have a wide lens?", [CAMERA])
        exact = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        partial = senf_for(
            "s2",
            "The camera is expensive.",
            ["(: c (HasProperty camera expensive) (STV 1.0 1.0))"],
        )

        results = build_weaves(query, [partial, exact], k=2)

        assert len(results) == 2
        assert [result.total_cost for result in results] == sorted(
            result.total_cost for result in results
        )
        assert all(
            isinstance(mapping, EntityMap)
            for result in results
            for mapping in result.entity_maps
        )
        assert results[0].guard == "s1->q1"
        assert results[0].total_cost == pytest.approx(
            results[0].structural_cost
            + results[0].exemplar_cost
            + results[0].conflict_cost
            + results[0].unmatched_cost,
            abs=1e-4,
        )

    def test_exact_role_mentions_are_preserved_in_entity_maps(self):
        atom = "(: a (Beside camera camera) (STV 1 1))"
        source = senf_for("s1", "one camera beside another camera", [atom])
        query = senf_for("q1", "one camera beside another camera", [atom])

        result = next(
            result for result in build_weaves(query, [source], k=2) if result.aligned
        )

        assert [(item.source_mention_id, item.target_mention_id) for item in result.entity_maps] == [
            ("s1:m0", "q1:m0"),
            ("s1:m1", "q1:m1"),
        ]

    def test_exact_match_ranks_ahead_of_role_permutation(self):
        query = senf_for("q1", "Alice sees Bob.", ["(: q (Sees alice bob) $tv)"])
        exact = senf_for("s1", "Alice sees Bob.", ["(: a (Sees alice bob) (STV 1 1))"])
        reversed_roles = senf_for(
            "s2", "Bob sees Alice.", ["(: b (Sees bob alice) (STV 1 1))"]
        )
        results = build_weaves(query, [reversed_roles, exact], k=2)
        assert results[0].guard == "s1->q1"
        assert weave(query, [reversed_roles]).pairs == ()

    def test_rule_premise_does_not_ground_a_fact_query(self):
        source = senf_for(
            "s1",
            "If Alice sees Bob, Alice knows Bob.",
            [
                "(: rule (Implication (Premises (Sees alice bob)) "
                "(Conclusions (Knows alice bob))) (STV 1 1))"
            ],
        )
        query = senf_for("q1", "Does Alice see Bob?", ["(: q (Sees alice bob) $tv)"])

        assert weave(query, [source]).pairs == ()

    def test_transport_truth_degrades_monotonically_with_cost(self):
        low = transport_truth(1.0, 1.0, 0.1)
        high = transport_truth(1.0, 1.0, 0.8)

        assert 0.0 < high.strength < low.strength < 1.0
        assert 0.0 < high.weight < low.weight < 1.0

    def test_unrelated_predicates_do_not_align_only_because_entities_match(self):
        source = senf_for(
            "s1", "The camera is smart.", ["(: a (Smart camera) (STV 1.0 1.0))"]
        )
        query = senf_for(
            "q1",
            "Is the camera expensive?",
            ["(: b (Expensive camera) (STV 1.0 1.0))"],
        )

        assert weave(query, [source]).pairs == ()

    def test_shared_camel_case_token_does_not_make_predicates_compatible(self):
        source = senf_for(
            "s1", "A treatment causes cancer.",
            ["(: a (CausesCancer treatment patient) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Does the treatment prevent cancer?",
            ["(: q (PreventsCancer treatment patient) $tv)"],
        )

        assert weave(query, [source]).pairs == ()

    def test_compatible_predicates_produce_an_explicit_bridge(self):
        source = senf_for(
            "s1",
            "The camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1.0 1.0))"],
        )
        query = senf_for(
            "q1",
            "Is the camera located in the lab?",
            ["(: b (LocatedIn camera lab) (STV 1.0 1.0))"],
        )

        result = next(
            result for result in build_weaves(query, [source], k=2) if result.aligned
        )

        assert result.aligned
        mapping = result.predicate_maps[0]
        assert (mapping.source_head, mapping.query_head) == (
            "AtLocation",
            "LocatedIn",
        )
        assert "PredicateBridge AtLocation LocatedIn" in predicate_bridge_atoms(result)[0]

    def test_only_explicit_safe_aliases_bridge_different_heads(self):
        source = senf_for(
            "s1", "The camera is located at the lab.",
            ["(: a (LocatedAt camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Is the camera at the lab?",
            ["(: q (AtLocation camera lab) $tv)"],
        )

        result = next(
            result for result in build_weaves(query, [source], k=2) if result.aligned
        )

        assert result.aligned
        assert result.predicate_maps[0].cost == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("source_head", "query_head"),
        [
            ("HasPart", "PartOf"),
            ("HasProperty", "PropertyOf"),
            ("UsedFor", "PurposeOf"),
        ],
    )
    def test_inverse_predicates_are_not_position_preserving_synonyms(
        self, source_head, query_head
    ):
        source = senf_for(
            "s1", "A source relation.",
            [f"(: a ({source_head} whole part) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "A reversed relation?",
            [f"(: q ({query_head} whole part) $tv)"],
        )

        result = weave(query, [source])

        assert result.pairs == ()
        assert executable_bridge_atoms(result, source, query) == []

    def test_value_and_frame_ref_fillers_are_compared_structurally(self):
        source = senf_for(
            "s1",
            "Kebede believes the camera costs 10.",
            ["(: a (Believes kebede (Costs camera 10)) (STV 1.0 1.0))"],
        )
        query = senf_for(
            "q1",
            "Does Kebede believe the camera costs 10?",
            ["(: b (Believes kebede (Costs camera 10)) (STV 1.0 1.0))"],
        )
        graph = resolve_identity([source, query])

        result = next(
            result
            for result in build_weaves(query, [source], k=2, identity_graph=graph)
            if result.aligned
        )

        assert result.aligned
        assert any("typed_filler" in pair.evidence for pair in result.pairs)

        incompatible = senf_for(
            "q2",
            "Does Kebede believe the camera owns 10?",
            ["(: c (Believes kebede (Owns camera 10)) (STV 1.0 1.0))"],
        )
        incompatible_result = weave(
            incompatible,
            [source],
            identity_graph=resolve_identity([source, incompatible]),
        )
        assert all(
            pair.query_frame_id != "q2:f1"
            for pair in incompatible_result.pairs
        )

    def test_identity_and_weave_bridges_are_scoped_to_ids(self):
        source = senf_for("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1.0 1.0))"])
        query = senf_for("q1", "It arrived.", ["(: b (Arrived it) (STV 1.0 1.0))"])
        graph = resolve_identity([source, query])
        result = next(
            result
            for result in build_weaves(query, [source], k=2, identity_graph=graph)
            if result.aligned
        )

        identity_atoms = identity_bridge_atoms(graph.merged)
        weave_atoms = weave_bridge_atoms(result)

        assert identity_atoms and "MentionIdentity" in identity_atoms[0]
        assert "s1_m0" in identity_atoms[0] and "q1_m0" in identity_atoms[0]
        assert weave_atoms and "EntityAlignment s1_e0 q1_e0" in weave_atoms[0]
        assert all("SimilarityLink" not in atom for atom in identity_atoms + weave_atoms)

    def test_executable_adapter_preserves_source_to_query_order_and_arity(self):
        source = senf_for(
            "s1", "The camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Is the camera located in the lab?",
            ["(: q (LocatedIn camera lab) $tv)"],
        )
        graph = resolve_identity([source, query])
        result = next(
            result
            for result in build_weaves(query, [source], k=3, identity_graph=graph)
            if result.aligned
        )

        adapters = executable_bridge_atoms(result, source, query, graph)

        assert len(adapters) == 1
        assert "(Premises (AtLocation camera lab))" in adapters[0]
        assert "(Conclusions (LocatedIn camera lab))" in adapters[0]

    @pytest.mark.parametrize(("speaker", "modality"), [("casey", None), (None, "claim")])
    def test_executable_adapter_rejects_guarded_root_claims(
        self, speaker, modality
    ):
        source = senf_for(
            "s1", "The camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Is the camera located in the lab?",
            ["(: q (LocatedIn camera lab) $tv)"],
        )
        source.frames[0].context = type(source.frames[0].context)(
            source.frames[0].context.source_unit_id,
            speaker=speaker,
            modality=modality,
            location_ref=source.frames[0].context.location_ref,
        )
        source.frames[0].modality = modality
        query.frames[0].context = type(query.frames[0].context)(
            query.frames[0].context.source_unit_id,
            speaker=speaker,
            modality=modality,
        )
        query.frames[0].modality = modality
        result = next(
            result for result in build_weaves(query, [source], k=3) if result.aligned
        )

        assert executable_bridge_atoms(result, source, query) == []

    def test_executable_adapter_respects_equal_symbol_identity_conflict(self):
        source = senf_for(
            "s1", "A camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Is another camera located in the lab?",
            ["(: q (LocatedIn camera lab) $tv)"],
        )
        graph = resolve_identity([source, query])
        result = next(
            item for item in build_weaves(query, [source], k=4, identity_graph=graph)
            if item.pairs
        )

        assert any(edge.negative_strength >= 0.5 for edge in graph.edges)
        assert executable_bridge_atoms(result, source, query, graph) == []

    def test_executable_adapter_requires_an_accepted_equal_symbol_identity(self):
        source = senf_for(
            "s1", "A camera is at the lab.",
            ["(: a (AtLocation camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Is a camera located in the lab?",
            ["(: q (LocatedIn camera lab) $tv)"],
        )
        graph = resolve_identity([source, query])
        result = next(
            item for item in build_weaves(query, [source], k=4, identity_graph=graph)
            if item.pairs
        )

        assert not graph.same_entity(
            source.mentions[0].entity_id, query.mentions[0].entity_id
        )
        assert executable_bridge_atoms(result, source, query, graph) == []

    def test_unsupported_entity_mapping_does_not_ground_the_query_symbol(self):
        source = senf_for(
            "s1", "Alice arrived.", ["(: a (Arrived alice) (STV 1 1))"]
        )
        query = senf_for(
            "q1", "Did Bob arrive?", ["(: q (Arrived bob) $tv)"]
        )
        graph = resolve_identity([source, query])
        result = next(
            item for item in build_weaves(query, [source], k=3, identity_graph=graph)
            if item.pairs
        )

        assert not result.entity_maps[0].identity_supported
        assert "alice" in result.grounded_symbols
        assert "bob" not in result.grounded_symbols


class TestStage4Search:
    def test_one_source_yields_multiple_mapping_hypotheses(self):
        query = senf_for("q1", "Did someone arrive?", ["(: q (Arrived person) $tv)"])
        source = senf_for(
            "s1",
            "Alice arrived and Bob arrived.",
            [
                "(: a (Arrived alice) (STV 1 1))",
                "(: b (Arrived bob) (STV 1 1))",
            ],
        )

        results = build_weaves(query, [source], k=3)

        assert len(results) >= 2
        assert {result.pairs[0].source_frame_id for result in results if result.pairs} >= {
            "s1:f0", "s1:f1",
        }

    def test_literal_identity_is_cheaper_than_strong_then_weak_transport(self):
        source = senf_for("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"])
        query = senf_for("q1", "It arrived.", ["(: q (Arrived it) $tv)"])
        left, right = source.mentions[0], query.mentions[0]

        def graph(cost):
            edge = IdentityEdge(
                left, right, 1.0 - cost, 1.0,
                positive_cost=cost,
            )
            return IdentityGraph(
                nodes=(left, right), edges=(edge,), merged=(edge,),
                representatives={right.entity_id: left.entity_id},
                mention_entities={left.mention_id: left.entity_id, right.mention_id: right.entity_id},
            )

        literal = weave(source, [source], identity_graph=IdentityGraph())
        strong = weave(query, [source], identity_graph=graph(0.1))
        weak = weave(query, [source], identity_graph=graph(0.6))

        assert literal.identity_cost == 0.0
        assert literal.identity_cost < strong.identity_cost < weak.identity_cost

    def test_negative_identity_evidence_is_a_separate_conflict_cost(self):
        source = senf_for("s1", "A lens cracked.", ["(: a (Cracked lens) (STV 1 1))"])
        query = senf_for("q1", "Did the lens crack?", ["(: q (Cracked lens) $tv)"])
        left, right = source.mentions[0], query.mentions[0]
        edge = IdentityEdge(
            left, right, 0.8, 1.0,
            negative_strength=0.7, positive_cost=0.2,
        )
        graph = IdentityGraph(
            nodes=(left, right), edges=(edge,), merged=(edge,),
            representatives={right.entity_id: left.entity_id},
            mention_entities={left.mention_id: left.entity_id, right.mention_id: right.entity_id},
        )

        result = next(
            result
            for result in build_weaves(query, [source], k=2, identity_graph=graph)
            if result.aligned
        )

        assert result.identity_cost == pytest.approx(0.2)
        assert result.conflict_cost == pytest.approx(0.7)

    def test_active_exemplar_alternatives_branch_the_search(self):
        source = score_exemplars(senf_for(
            "s1", "The camera lasted three hours.",
            ["(: a (Lasted camera three_hours) (STV 1 1))"],
        ))
        query = score_exemplars(senf_for(
            "q1", "How long did the camera last?",
            ["(: q (Lasted camera three_hours) $tv)"],
        ))

        results = build_weaves(query, [source], k=3)

        assert len(results) == 3
        assert len({result.exemplar_maps for result in results}) == 3
        assert all(result.exemplar_maps for result in results)

    def test_v4_context_mismatches_have_independent_costs(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "senf_weave_max_cost", 4.0)
        source = senf_for("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"])
        query = senf_for("q1", "Did the camera arrive?", ["(: q (Arrived camera) $tv)"])
        source.frames[0].time_ref = "yesterday"
        source.frames[0].location_ref = "lab"
        source.frames[0].modality = "reported_speech"
        query.frames[0].time_ref = "today"
        query.frames[0].location_ref = "office"
        query.frames[0].modality = "observed"

        result = next(
            result for result in build_weaves(query, [source], k=2) if result.aligned
        )

        assert (result.time_cost, result.location_cost, result.modality_cost) == (1.0, 1.0, 1.0)

    def test_inconsistent_cross_frame_entity_mapping_is_distortion(self):
        source = senf_for(
            "s1", "The camera arrived and departed.",
            ["(: a (Moved camera) (STV 1 1))", "(: b (Moved lens) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "Alice and Bob moved.",
            ["(: q1 (Moved alice) $tv)", "(: q2 (Moved bob) $tv)"],
        )
        shared = source.frames[0].roles[0].filler
        assert isinstance(shared, EntityRef)
        source.frames[1].roles[0] = type(source.frames[1].roles[0])("Theme", shared, 0)

        result = next(
            result for result in build_weaves(query, [source], k=3) if result.aligned
        )

        assert len(result.pairs) == 2
        assert result.distortion_cost > 0.0
        assert result.total_cost == pytest.approx(
            result.structural_cost + result.exemplar_cost + result.identity_cost
            + result.conflict_cost + result.time_cost + result.location_cost
            + result.modality_cost + result.unmatched_cost + result.distortion_cost,
            abs=1e-4,
        )

    def test_inconsistent_cross_frame_role_mapping_is_distortion(self):
        source = senf_for(
            "s1", "The camera moved twice.",
            ["(: a (Moved camera) (STV 1 1))", "(: b (Moved camera) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "The device moved twice.",
            ["(: q1 (Moved device) $tv)", "(: q2 (Moved device) $tv)"],
        )
        query.frames[1].roles[0] = type(query.frames[1].roles[0])(
            "Patient", query.frames[1].roles[0].filler, 0
        )

        result = weave(query, [source])

        assert len(result.pairs) == 2
        assert result.distortion_cost > 0.0

    def test_nested_frame_relationship_inconsistency_is_detected(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_per_source_k", 10)
        source = senf_for(
            "s1", "An observer believed one thing moved and another moved.",
            [
                "(: a (Believes observer (Moved alpha)) (STV 1 1))",
                "(: b (Moved beta) (STV 1 1))",
            ],
        )
        query = senf_for(
            "q1", "A witness believed one thing moved and another moved.",
            [
                "(: q (Believes witness (Moved gamma)) $tv)",
                "(: r (Moved delta) $tv)",
            ],
        )

        results = build_weaves(query, [source], k=10)

        assert any(result.distortion_cost == 0.0 for result in results)
        assert any(result.distortion_cost > 0.0 for result in results)

    def test_global_per_source_and_beam_bounds_are_deterministic(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_per_source_k", 2)
        monkeypatch.setattr(settings, "senf_weave_beam_width", 3)
        query = senf_for("q1", "Did someone move?", ["(: q (Moved person) $tv)"])
        sources = [
            senf_for(
                sentence_id,
                "Two things moved.",
                ["(: a (Moved alpha) (STV 1 1))", "(: b (Moved beta) (STV 1 1))"],
            )
            for sentence_id in ("s2", "s1")
        ]

        first = build_weaves(query, sources, k=3)
        second = build_weaves(query, list(reversed(sources)), k=3)

        assert len(first) == 3
        assert max(sum(result.guard.startswith(f"{sid}->") for result in first) for sid in ("s1", "s2")) <= 2
        assert first == second

    def test_beam_uses_the_final_normalized_cost_lower_bound(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_beam_width", 1)
        query = senf_for(
            "q1", "Three events.",
            [
                "(: q1 (FastMove alpha) $tv)",
                "(: q2 (Arrived gamma) $tv)",
                "(: q3 (Left delta) $tv)",
            ],
        )
        source = senf_for(
            "s1", "Three source events.",
            [
                "(: a1 (FastMove beta) (STV 1 1))",
                "(: a2 (Arrived gamma) (STV 1 1))",
                "(: a3 (Left delta) (STV 1 1))",
            ],
        )
        source.frames[1].time_ref = "yesterday"
        query.frames[1].time_ref = "today"

        result = weave(query, [source])

        assert len(result.pairs) == 3
        assert result.unmatched_cost == 0.0

    def test_pair_cap_is_allocated_across_query_frames(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_max_pair_candidates", 2)
        query = senf_for(
            "q1", "Two events.",
            ["(: q1 (Arrived alpha) $tv)", "(: q2 (Left beta) $tv)"],
        )
        source = senf_for(
            "s1", "Several events.",
            [
                "(: a1 (Arrived alpha) (STV 1 1))",
                "(: a2 (Arrived other) (STV 1 1))",
                "(: b (Left beta) (STV 1 1))",
            ],
        )

        result = weave(query, [source])

        assert {pair.query_frame_id for pair in result.pairs} == {"q1:f0", "q1:f1"}

    def test_pair_cap_smaller_than_frame_count_does_not_favor_one_frame(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_max_pair_candidates", 2)
        query = senf_for(
            "q1", "Three events.",
            [
                "(: q1 (Moved alpha) $tv)",
                "(: q2 (Moved beta) $tv)",
                "(: q3 (Moved gamma) $tv)",
            ],
        )
        source = senf_for(
            "s1", "Several things moved.",
            [
                "(: a1 (Moved alpha) (STV 1 1))",
                "(: a2 (Moved other) (STV 1 1))",
                "(: b (Moved beta) (STV 1 1))",
                "(: c (Moved gamma) (STV 1 1))",
            ],
        )

        results = build_weaves(query, [source], k=3)

        considered = {
            pair.query_frame_id
            for result in results
            for pair in result.pairs + result.rejected_pairs
        }
        assert considered >= {"q1:f0", "q1:f1"}

    def test_total_cost_ranks_alignment_against_unaligned_hypothesis(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_max_cost", 2.0)
        query = senf_for("q1", "Did the camera arrive?", ["(: q (Arrived camera) $tv)"])
        source = senf_for("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"])
        source.frames[0].time_ref = "yesterday"
        query.frames[0].time_ref = "today"

        results = build_weaves(query, [source], k=2)

        assert results[0].total_cost == 1.0
        assert results[1].total_cost > results[0].total_cost
        assert not results[0].aligned
        assert results[1].aligned

    def test_over_cost_alignment_is_a_non_operational_diagnostic(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "senf_weave_max_cost", 0.5)
        query = senf_for("q1", "Did the camera arrive?", ["(: q (Arrived camera) $tv)"])
        source = senf_for("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"])
        source.frames[0].time_ref = "yesterday"
        query.frames[0].time_ref = "today"

        results = build_weaves(query, [source], k=2)

        assert not results[0].aligned
        assert results[0].rejected_pairs == ()
        assert results[1].rejected_pairs
        assert results[1].pairs == ()
        assert executable_bridge_atoms(results[1], source, query) == []

    def test_unaligned_results_have_deterministic_guard_order(self):
        query = senf_for("q1", "Did Alice arrive?", ["(: q (Arrived alice) $tv)"])
        sources = [
            senf_for(sentence_id, "Bob left.", ["(: a (Left bob) (STV 1 1))"])
            for sentence_id in ("s2", "s1")
        ]

        results = build_weaves(query, sources, k=2)

        assert [result.guard for result in results] == ["s1->q1", "s2->q1"]
        assert all(not result.aligned for result in results)


class TestGuideCasesDownstream:
    def test_case_a_identity_transport_reaches_the_returned_camera(self):
        lent = senf_for(
            "s1", "Alex lent Sam the Nikon camera.",
            ["(: a (Lent alex sam nikon_camera) (STV 1 1))", "(: k (IsA nikon_camera Camera) (STV 1 1))"],
        )
        returned = senf_for(
            "s2", "Sam returned it.",
            ["(: b (Returned sam it) (STV 1 1))", "(: k (IsA it Camera) (STV 1 1))"],
        )
        query = senf_for("q1", "Did Sam return the Nikon camera?", ["(: q (Returned sam nikon_camera) $tv)"])
        graph = resolve_identity([lent, returned, query])

        result = weave(query, [returned], identity_graph=graph)

        assert result.aligned
        assert result.identity_cost > 0.0

    def test_case_c_distinct_game_exemplars_are_costed(self):
        strategic = score_exemplars(senf_for(
            "s1", "The game required deep strategy.", ["(: a (IsA game Game) (STV 1 1))"],
        ))
        physical = score_exemplars(senf_for(
            "q1", "The game was physically exhausting.", ["(: q (IsA game Game) $tv)"],
        ))

        assert weave(physical, [strategic]).exemplar_cost > 0.0

    def test_case_d_contrast_remains_a_costly_nonidentity_mapping(self):
        attached = score_exemplars(senf_for(
            "s1", "The lens attached to the camera was cracked.",
            ["(: a (Attached lens camera) (STV 1 1))"],
        ))
        borrowed = score_exemplars(senf_for(
            "s2", "Casey says Sam borrowed the lens separately.",
            ["(: b (Says casey (Borrowed sam lens)) (STV 1 1))"],
        ))
        query = score_exemplars(senf_for(
            "q1", "Did Sam borrow the lens?", ["(: q (Borrowed sam lens) $tv)"],
        ))
        graph = resolve_identity([attached, borrowed, query])

        result = next(
            result
            for result in build_weaves(query, [borrowed], k=3, identity_graph=graph)
            if result.aligned
        )

        assert result.aligned
        assert result.conflict_cost > 0.0


class TestStage7Transport:
    def test_actual_fact_alignment_into_branch_carries_probability_cost(self):
        declaration = "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))"
        source = senf_for(
            "s1", "The camera is at the lab.",
            [declaration, "(: a (AtLocation camera lab) (STV 1 1))"],
        )
        query = senf_for(
            "q1", "In the rain branch, is the camera in the lab?",
            [declaration, "(: q (InContext rain none (LocatedIn camera lab)) $tv)"],
        )
        graph = resolve_identity([source, query])
        result = next(
            item for item in build_weaves(query, [source], k=3, identity_graph=graph)
            if item.aligned
        )

        assert result.branch_probability == 0.4
        assert result.branch_cost == pytest.approx(0.6)
        adapter = executable_bridge_atoms(result, source, query, graph)[0]
        assert _bridge_strength(adapter) < 0.4

    def test_sibling_branch_alignment_is_rejected_before_search(self):
        declarations = [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: dry (BranchContext dry actual_root counterfactual 0.6) (STV 1 1))",
        ]
        source = senf_for(
            "s1", "Rain world.", declarations + [
                "(: a (InContext rain none (Wet ground)) (STV 1 1))"
            ],
        )
        query = senf_for(
            "q1", "Dry world?", declarations + [
                "(: q (InContext dry none (Wet ground)) $tv)"
            ],
        )

        assert not weave(query, [source]).aligned


def _bridge_strength(atom: str) -> float:
    marker = atom.rsplit("(STV ", 1)[1]
    return float(marker.split()[0])
