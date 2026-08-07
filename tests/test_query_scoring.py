"""Characterization tests for query candidate scoring.

These lock the CURRENT scoring behavior so the C1 extraction (and the C7/C8 scoring
extensions) are provably behavior-preserving. Weights here are the observed constants,
not aspirational values - if a change is intended, change the test deliberately.
"""

import pytest

from core import query_scoring


def sig(head: str, *args: str) -> dict:
    return {"head": head, "arity": len(args), "args": list(args)}


def q(head: str, *args: str) -> dict:
    variables = [a for a in args if a.startswith(("$", "?"))]
    return {"head": head, "arity": len(args), "args": list(args), "variables": variables}


class TestSameShape:
    def test_matches_on_head_and_arity(self):
        assert query_scoring.same_shape(q("Smart", "kebede"), sig("Smart", "abebe"))

    def test_rejects_different_head(self):
        assert not query_scoring.same_shape(q("Smart", "kebede"), sig("Tall", "kebede"))

    def test_rejects_different_arity(self):
        assert not query_scoring.same_shape(q("Eats", "kebede", "fish"), sig("Eats", "kebede"))


class TestSignatureCanBind:
    def test_grounded_query_matching_signature_binds(self):
        assert query_scoring.signature_can_bind(q("Smart", "kebede"), sig("Smart", "kebede"))

    def test_grounded_mismatch_does_not_bind(self):
        assert not query_scoring.signature_can_bind(q("Smart", "kebede"), sig("Smart", "abebe"))

    def test_variable_binds_to_constant_witness(self):
        assert query_scoring.signature_can_bind(q("Smart", "$x"), sig("Smart", "kebede"))

    def test_variable_against_variable_is_no_witness(self):
        assert not query_scoring.signature_can_bind(q("Smart", "$x"), sig("Smart", "$y"))

    def test_mixed_grounded_mismatch_short_circuits(self):
        assert not query_scoring.signature_can_bind(
            q("Eats", "$x", "fish"), sig("Eats", "kebede", "meat")
        )

    def test_question_mark_treated_as_variable(self):
        assert query_scoring.signature_can_bind(q("Smart", "?x"), sig("Smart", "kebede"))


class TestHasWitnessPath:
    def test_grounded_query_always_has_path(self):
        assert query_scoring.has_witness_path(q("Smart", "kebede"), [], [])

    def test_variable_query_with_no_signatures_has_no_path(self):
        assert not query_scoring.has_witness_path(q("Smart", "$x"), [], [])

    def test_conclusion_alone_provides_path(self):
        assert query_scoring.has_witness_path(q("Smart", "$x"), [], [sig("Smart", "kebede")])


class TestScoreQueryCandidate:
    def test_grounded_yes_no_with_exact_fact(self):
        # fact(6) + grounded-yes/no(3) + fully-grounded(2) = 11
        assert query_scoring.score_query_candidate(
            q("Smart", "kebede"), [sig("Smart", "kebede")], [], True
        ) == 11

    def test_grounded_yes_no_shape_match_only(self):
        # fact(6) + grounded-yes/no(3), args differ so no fully-grounded bonus
        assert query_scoring.score_query_candidate(
            q("Smart", "kebede"), [sig("Smart", "abebe")], [], True
        ) == 9

    def test_conclusion_match_scores_four(self):
        # conclusion(4) + grounded-yes/no(3)
        assert query_scoring.score_query_candidate(
            q("Smart", "kebede"), [], [sig("Smart", "$person")], True
        ) == 7

    def test_grounded_wh_question_scores_one_not_three(self):
        # fact(6) + grounded-wh(1) + fully-grounded(2)
        assert query_scoring.score_query_candidate(
            q("Smart", "kebede"), [sig("Smart", "kebede")], [], False
        ) == 9

    def test_variable_wh_question_gets_three(self):
        # fact(6) + variable-wh(3)
        assert query_scoring.score_query_candidate(
            q("Smart", "$x"), [sig("Smart", "kebede")], [], False
        ) == 9

    def test_variable_yes_no_gets_zero_for_variables(self):
        # fact(6) + variable-yes/no(0)
        assert query_scoring.score_query_candidate(
            q("Smart", "$x"), [sig("Smart", "kebede")], [], True
        ) == 6

    def test_yes_no_with_variable_and_no_witness_is_rejected(self):
        assert query_scoring.score_query_candidate(q("Smart", "$x"), [], [], True) is None

    def test_variable_wh_question_scores_even_with_no_matches(self):
        # variable-wh(3) alone. Note this is NOT rejected: an unmatched wh-candidate
        # still outranks nothing, which is why fallback has something to try.
        assert query_scoring.score_query_candidate(q("Smart", "$x"), [], [], False) == 3

    def test_grounded_wh_question_scores_one_with_no_matches(self):
        assert query_scoring.score_query_candidate(q("Smart", "kebede"), [], [], False) == 1

    def test_rejection_comes_from_witness_guard_not_zero_score(self):
        # The only reachable None is the yes/no + variables + no-witness path.
        # The `score > 0` guard is defensive: every other branch adds at least 1.
        assert query_scoring.score_query_candidate(q("Smart", "$x"), [], [], True) is None
        assert query_scoring.score_query_candidate(q("Smart", "$x"), [], [], False) == 3

    def test_facts_and_conclusions_stack(self):
        # fact(6) + conclusion(4) + grounded-yes/no(3) + fully-grounded(2) = 15
        assert query_scoring.score_query_candidate(
            q("Smart", "kebede"),
            [sig("Smart", "kebede")],
            [sig("Smart", "kebede")],
            True,
        ) == 15


class TestDelegationParity:
    """Both historical call sites must agree with the shared implementation.

    They carried duplicate copies before C1; this is the regression guard against
    them drifting apart again.
    """

    @pytest.mark.parametrize(
        "query,facts,conclusions,is_yes_no",
        [
            (q("Smart", "kebede"), [sig("Smart", "kebede")], [], True),
            (q("Smart", "$x"), [sig("Smart", "kebede")], [], False),
            (q("Smart", "$x"), [], [], True),
            (q("Eats", "kebede", "fish"), [sig("Eats", "kebede", "fish")], [], True),
        ],
    )
    def test_postprocessor_matches_shared(self, query, facts, conclusions, is_yes_no):
        from core.pln_postprocessor import PLNPostprocessor

        expected = query_scoring.score_query_candidate(query, facts, conclusions, is_yes_no)
        assert PLNPostprocessor().score_query_candidate(query, facts, conclusions, is_yes_no) == expected
