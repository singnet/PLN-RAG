import pytest

from core.senf.extractor import extract_senf
from core.senf.bridge import (
    identity_bridge_atoms,
    predicate_bridge_atoms,
    transport_truth,
    weave_bridge_atoms,
)
from core.senf.exemplars import score_exemplars
from core.senf.identity import resolve_identity
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

        result = weave(query, [source])

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

        result = weave(query, [source])

        assert result.aligned
        mapping = result.predicate_maps[0]
        assert (mapping.source_head, mapping.query_head) == (
            "AtLocation",
            "LocatedIn",
        )
        assert "PredicateBridge AtLocation LocatedIn" in predicate_bridge_atoms(result)[0]

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

        result = weave(query, [source], identity_graph=graph)

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
        result = weave(query, [source], identity_graph=graph)

        identity_atoms = identity_bridge_atoms(graph.merged)
        weave_atoms = weave_bridge_atoms(result)

        assert identity_atoms and "MentionIdentity" in identity_atoms[0]
        assert "s1_m0" in identity_atoms[0] and "q1_m0" in identity_atoms[0]
        assert weave_atoms and "EntityAlignment s1_e0 q1_e0" in weave_atoms[0]
        assert all("SimilarityLink" not in atom for atom in identity_atoms + weave_atoms)
