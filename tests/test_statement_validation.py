"""Characterization tests for the shared statement validator.

The service delegates its persistence and transient gates here. Failures are
dropped and counted rather than raised,
never raised. Any new producer of atoms (SENF bridge atoms in particular) must
emit shapes this accepts, so the accept/reject table is pinned here verbatim.
"""

import pytest

from core.parser import ParseResult
from core.service import PLNRAGService
from core.statement_validation import is_valid_statement, validate_statements

VALID_FACT = "(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))"
VALID_RULE = (
    "(: fish_rule (Implication (Premises (Eats $p fish)) (Conclusions (Smart $p)))"
    " (STV 1.0 1.0))"
)


def test_parse_result_addition_preserves_existing_positional_fields():
    result = ParseResult([VALID_FACT], ["query"], {"source": "legacy"})

    assert result.metadata == {"source": "legacy"}
    assert result.transient_statements == []
    assert result.trusted_transient_statements == []
    assert result.original_query == ""


@pytest.fixture
def validate():
    """Direct validator access without constructing the service graph."""
    return is_valid_statement


class TestAccepted:
    @pytest.mark.parametrize(
        "stmt",
        [
            VALID_FACT,
            VALID_RULE,
            "(: p (P a) (PointMass 1.0))",
            "(: p (P a) (ParticleFrom x))",
            "(: p (P a) (ParticleFromNormal 1.0 0.2))",
            "(: p (P a) (ParticleFromPairs ((1 0.4) (2 0.6))))",
            "(: p (Smart $x) (STV 1.0 1.0))",  # free variable in a body is allowed
        ],
    )
    def test_accepted(self, validate, stmt):
        ok, err = validate(stmt)
        assert ok, f"expected accept, got reject({err})"
        assert err == ""


class TestRejected:
    @pytest.mark.parametrize(
        "stmt,reason",
        [
            ("(Eats kebede fish)", "missing_prefix"),
            ("(: p (P a))", "missing_weight"),
            ("(: p (P a) (STV 1.0 1.0)", "unbalanced_parens"),
            ("(: (P a) (STV 1.0 1.0))", "bad_toplevel"),  # no name
            ("(:x (P a) (STV 1.0 1.0))", "bad_toplevel"),  # no space after ":"
            ("(: $p (P a) (STV 1.0 1.0))", "variable_name"),
            ("(: r (Implication (Eats $p fish) (Smart $p)) (STV 1.0 1.0))",
             "bad_implication_shape"),
            ("(: r (Implication (Premises (Eats $p fish))) (STV 1.0 1.0))",
             "bad_implication_shape"),
            ("(: p (P a) (STV 1.0 1.0) (STV 0.5 0.5))", "bad_toplevel"),
            ("(: p (P a) (STV 1.0))", "bad_truth_value"),
            ("(: p (P a) (STV 1.0 1.0 extra))", "bad_truth_value"),
            ("(: p (P a) (ParticleFromNormal 1.0))", "bad_truth_value"),
            ("(: p (P a) (ParticleFromBogus x))", "bad_truth_value"),
            ("(: p (P a) (ParticleFromPairs ((1 0.4 9))))", "bad_truth_value"),
            ("(: p (P a) (UnknownTV 1.0 1.0) (STV 1 1))", "bad_toplevel"),
            ("(: p (P a) (Nested b (C d)) (STV 1.0 1.0)", "unbalanced_parens"),
            ("(: p (Implication (Conclusions (P a)) (Premises (Q a))) (STV 1 1))",
             "bad_implication_shape"),
            ("(: p (Premises (P a)) (STV 1 1))", "bad_implication_shape"),
            ("(: p (Implication (Premises (P a)) (Conclusions)) (STV 1 1))",
             "zero_arity_atom"),
        ],
    )
    def test_rejected_with_reason(self, validate, stmt, reason):
        ok, err = validate(stmt)
        assert not ok
        assert err == reason


class TestZeroArityAtoms:
    """Observed in stress25 case S02.

    The parser emitted `(Conclusions (AssociatedWithSevereDisease))` while the query
    asked `(: $prf (AssociatedWithSevereDisease $marker) $tv)`. A 0-ary conclusion can
    never unify with a 1-ary goal, so the rule was unreachable and the only symptom
    was a missing proof.
    """

    def test_zero_arity_conclusion_is_rejected(self, validate):
        ok, err = validate(
            "(: r (Implication (Premises (P a)) (Conclusions (AssociatedWithSevereDisease)))"
            " (STV 1.0 1.0))"
        )
        assert not ok
        assert err == "zero_arity_atom"

    def test_zero_arity_fact_is_rejected(self, validate):
        ok, err = validate("(: p (Lonely) (STV 1.0 1.0))")
        assert not ok
        assert err == "zero_arity_atom"

    @pytest.mark.parametrize("stmt", [VALID_FACT, VALID_RULE])
    def test_well_formed_statements_are_unaffected(self, validate, stmt):
        ok, err = validate(stmt)
        assert ok, err

    def test_structural_atoms_may_not_be_empty(self, validate):
        ok, error = validate(
            "(: r (Implication (Premises (And)) (Conclusions (P a))) (STV 1.0 1.0))"
        )
        assert not ok
        assert error == "zero_arity_atom"

    def test_balanced_nested_fact_is_accepted(self, validate):
        ok, error = validate(
            "(: claim (Claims casey (AtLocation camera lab)) (STV 1.0 1.0))"
        )
        assert ok, error


class TestBridgeAtomShape:
    """The C6 constraint, pinned as a test rather than a comment.

    The paper writes similarity links as bare atoms with a lowercase `stv`.
    That form is silently dropped here, so SENF bridge atoms must be emitted in
    the named `(: name body (STV s c))` form instead.
    """

    def test_paper_bare_form_is_rejected(self, validate):
        ok, err = validate("(SimilarityLink cam1 it_s2 (stv 0.84 0.72))")
        assert not ok
        assert err == "missing_prefix"

    def test_named_form_is_accepted(self, validate):
        ok, _ = validate("(: senf_id_1 (SimilarityLink cam1 it_s2) (STV 0.84 0.72))")
        assert ok

    def test_lowercase_stv_alone_does_not_satisfy_the_weight_check(self, validate):
        ok, err = validate("(: senf_id_1 (SimilarityLink cam1 it_s2) (stv 0.84 0.72))")
        assert not ok
        assert err == "missing_weight"


class TestValidateStatements:
    def test_partitions_and_normalizes_whitespace(self):
        service = PLNRAGService.__new__(PLNRAGService)
        valid, rejected = service._validate_statements(
            ["(:   kebede_eats_fish   (Eats kebede fish)\n  (STV 1.0 1.0))", "(P a)"]
        )
        assert valid == [VALID_FACT]
        assert rejected == [{"stmt": "(P a)", "error": "missing_prefix"}]

    def test_blank_entries_are_dropped_without_being_reported(self):
        service = PLNRAGService.__new__(PLNRAGService)
        valid, rejected = service._validate_statements(["", "   ", "\n"])
        assert valid == []
        assert rejected == []

    def test_none_input_is_tolerated(self):
        service = PLNRAGService.__new__(PLNRAGService)
        assert service._validate_statements(None) == ([], [])

    def test_service_delegates_to_the_shared_validator(self):
        service = PLNRAGService.__new__(PLNRAGService)
        given = [VALID_FACT, "(P a)"]
        assert service._validate_statements(given) == validate_statements(given)
