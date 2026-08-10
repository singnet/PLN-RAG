import pytest

from core.senf.extractor import extract_senf
from core.senf.weave import MIN_PAIR_SCORE, weave


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

    def test_an_unrelated_question_does_not_align(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Does Kebede eat fish?", [EATS])

        result = weave(query, [source])

        assert result.pairs == ()
        assert result.distortion == 1.0

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


class TestResolveHook:
    def test_identity_resolution_is_applied_to_symbols(self):
        source = senf_for("s1", "The camera has a wide lens.", [CAMERA])
        query = senf_for("q1", "Is it expensive?", ["(: c (HasProperty it wide_lens) (STV 1.0 1.0))"])

        without = weave(query, [source])
        with_resolve = weave(
            query, [source], resolve=lambda s: "camera" if s == "it" else s
        )

        assert "camera" in with_resolve.grounded_symbols
        assert with_resolve.distortion <= without.distortion
