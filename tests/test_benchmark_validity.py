"""Tests for the benchmark validity guard.

A run whose LLM was dead produced 25/25 `no_query`, wrote a report, and exited 0.
These tests pin the guard that now catches that shape, so a future infrastructure
failure cannot be mistaken for a baseline again.
"""

import sys
import types

import pytest

import benchmark_parsers as bp


def summary(**stats) -> dict[str, dict]:
    base = {
        "cases": 25,
        "correct": 0,
        "proof_found": 0,
        "no_query": 0,
        "weakly_aligned": 0,
        "fallback_used": 0,
        "errors": 0,
    }
    base.update(stats)
    return {"canonical_pln": base}


class TestValidityWarnings:
    def test_all_no_query_is_flagged(self):
        warnings = bp._validity_warnings(summary(no_query=25))
        assert len(warnings) == 1
        assert "no_query" in warnings[0]

    def test_all_errors_is_flagged(self):
        warnings = bp._validity_warnings(summary(errors=25))
        assert any("raised" in w for w in warnings)

    def test_zero_proofs_alone_is_not_flagged(self):
        """A harsh-but-real baseline must stay usable.

        proof_found == 0 with queries being generated is a finding, not a fault;
        only the no_query/error signatures indicate broken infrastructure.
        """
        assert bp._validity_warnings(summary(proof_found=0, no_query=0)) == []

    def test_partial_no_query_is_not_flagged(self):
        assert bp._validity_warnings(summary(no_query=24)) == []

    def test_empty_suite_is_not_flagged(self):
        assert bp._validity_warnings(summary(cases=0, no_query=0)) == []

    def test_flags_only_the_offending_parser(self):
        stats = summary(no_query=25)
        stats["langextract"] = dict(stats["canonical_pln"], no_query=3)
        warnings = bp._validity_warnings(stats)
        assert len(warnings) == 1
        assert warnings[0].startswith("canonical_pln:")


class TestSummarizeParser:
    def test_counts_errors(self):
        results = [
            {"error": "boom", "proof_found": False},
            {"proof_found": True},
        ]
        assert bp._summarize_parser(results)["errors"] == 1

    def test_no_error_key_counts_zero(self):
        assert bp._summarize_parser([{"proof_found": True}])["errors"] == 0


class FakeChainer:
    """Stands in for PeTTaChainer. `proofs` is what backward chaining returns."""

    def __init__(self, proofs=None, raises=None):
        self._proofs = proofs if proofs is not None else ["(Path A C)"]
        self._raises = raises
        self.atoms: list[str] = []
        self.queried: list[str] = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def select_facts(self, facts):
        return facts

    def forward_chain(self, seeds, steps=0):
        return seeds

    def query(self, atom, steps=0, timeout_sec=0):
        self.queried.append(atom)
        if self._raises:
            raise self._raises
        return self._proofs


@pytest.fixture
def fake_pettachainer(monkeypatch):
    """Install a stub `pettachainer` module and hand back its last instance.

    _preflight_reasoner imports inside the function body, so patching sys.modules
    is enough and the real chainer is never constructed.
    """
    made: list[FakeChainer] = []

    def install(**kwargs):
        module = types.ModuleType("pettachainer")
        module.PeTTaChainer = lambda: made[-1] if made else None
        chainer = FakeChainer(**kwargs)
        made.append(chainer)
        monkeypatch.setitem(sys.modules, "pettachainer", module)
        return chainer

    return install


class TestPreflightReasoner:
    """Pins the guard for a dead reasoner.

    The void baseline had no_query=0 and proof_found=0 for all 25 cases: queries
    were generated fine, every proof search came back empty. _validity_warnings
    deliberately does not flag that shape (see
    test_zero_proofs_alone_is_not_flagged), so the only place it can be caught is
    a live smoke test of a fact that must be derivable.
    """

    def test_working_chainer_passes(self, fake_pettachainer):
        fake_pettachainer()
        assert bp._preflight_reasoner() == (True, "")

    def test_empty_proofs_fail(self, fake_pettachainer):
        """The exact regression: query returns [] rather than raising."""
        fake_pettachainer(proofs=[])
        ok, detail = bp._preflight_reasoner()
        assert not ok
        assert "Path A C" in detail

    def test_exception_is_reported_not_raised(self, fake_pettachainer):
        fake_pettachainer(raises=RuntimeError("prolog stack overflow"))
        ok, detail = bp._preflight_reasoner()
        assert not ok
        assert "RuntimeError" in detail
        assert "prolog stack overflow" in detail

    def test_missing_module_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pettachainer", None)
        ok, detail = bp._preflight_reasoner()
        assert not ok
        assert detail

    def test_query_is_backward_chaining_over_a_derived_fact(self, fake_pettachainer):
        """(Path A C) is not asserted — it only exists if chaining worked.

        Guards against someone "fixing" a flaky pre-flight by probing a fact that
        is already in the atomspace, which would pass with chaining fully broken.
        """
        chainer = fake_pettachainer()
        bp._preflight_reasoner()
        assert chainer.queried == ["(: $prf (Path A C) $tv)"]
        assert not any("(Path A C)" in atom for atom in chainer.atoms)
