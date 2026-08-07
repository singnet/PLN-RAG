"""Characterization tests for the service-layer statement validator.

`PLNRAGService._validate_statements` is the single gate every atom passes before
reaching the reasoner. Failures here are silent: the atom is dropped and counted,
never raised. Any new producer of atoms (SENF bridge atoms in particular) must
emit shapes this accepts, so the accept/reject table is pinned here verbatim.
"""

import pytest

from core.service import PLNRAGService

VALID_FACT = "(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))"
VALID_RULE = (
    "(: fish_rule (Implication (Premises (Eats $p fish)) (Conclusions (Smart $p)))"
    " (STV 1.0 1.0))"
)


@pytest.fixture
def validate():
    """Bare validator access without constructing the full service graph.

    _is_valid_statement touches no instance state, so an uninitialized instance
    is enough and avoids standing up parsers, Qdrant and the reasoner.
    """
    service = PLNRAGService.__new__(PLNRAGService)
    return service._is_valid_statement


class TestAccepted:
    @pytest.mark.parametrize(
        "stmt",
        [
            VALID_FACT,
            VALID_RULE,
            "(: p (P a) (PointMass 1.0))",
            "(: p (P a) (ParticleFrom x))",
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
        ],
    )
    def test_rejected_with_reason(self, validate, stmt, reason):
        ok, err = validate(stmt)
        assert not ok
        assert err == reason


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
