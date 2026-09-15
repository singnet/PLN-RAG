import threading

import pytest

from core.reasoner import Reasoner


PERSISTENT = "(: seed (Seed alice) (STV 1.0 1.0))"
BACKGROUND = "(: world (World local) (STV 1.0 1.0))"
TRANSIENT_RULE = (
    "(: derive (Implication (Premises (Seed $x) (World local)) "
    "(Conclusions (Ready $x))) (STV 1.0 1.0))"
)
QUERY = "(: $prf (Ready alice) $tv)"


class FakeChainer:
    instances = []

    def __init__(self):
        self.atoms = []
        self.instances.append(self)

    def add_atom(self, atom):
        if "Explodes" in atom:
            raise RuntimeError("bad atom")
        self.atoms.append(atom)

    def query(self, query, timeout_sec):
        if "Failure" in query:
            raise RuntimeError("query failed")
        if query == QUERY and {PERSISTENT, BACKGROUND, TRANSIENT_RULE}.issubset(
            self.atoms
        ):
            return ["transient proof"]
        return []


def fake_reasoner(monkeypatch, tmp_path):
    atomspace = tmp_path / "kb.metta"
    background = tmp_path / "background.metta"
    atomspace.write_text(PERSISTENT + "\n", encoding="utf-8")
    background.write_text(BACKGROUND + "\n", encoding="utf-8")
    FakeChainer.instances = []
    monkeypatch.setattr("core.reasoner.PeTTaChainer", FakeChainer)

    made = Reasoner.__new__(Reasoner)
    made._atomspace_path = str(atomspace)
    made._query_timeout = 3
    made._lock = threading.Lock()
    made._handler = FakeChainer()
    made._background_files = {str(background)}
    made._load_from_disk()
    return made, atomspace


def test_transient_query_uses_fresh_complete_atomspace_without_leaking(
    monkeypatch, tmp_path
):
    reasoner, atomspace = fake_reasoner(monkeypatch, tmp_path)
    persistent_handler = reasoner._handler
    before = atomspace.read_bytes()

    assert reasoner.query(QUERY, transient_statements=[TRANSIENT_RULE]) == [
        "transient proof"
    ]
    assert len(FakeChainer.instances) == 2
    assert PERSISTENT in FakeChainer.instances[-1].atoms
    assert BACKGROUND in FakeChainer.instances[-1].atoms
    assert TRANSIENT_RULE in FakeChainer.instances[-1].atoms
    assert TRANSIENT_RULE not in persistent_handler.atoms
    assert atomspace.read_bytes() == before

    assert reasoner.query(QUERY) == []
    assert len(FakeChainer.instances) == 2


def test_transient_only_query_excludes_persistent_and_background_atoms(
    monkeypatch, tmp_path
):
    reasoner, _ = fake_reasoner(monkeypatch, tmp_path)

    assert reasoner.query_transient_only(QUERY, [TRANSIENT_RULE]) == []
    isolated = FakeChainer.instances[-1].atoms
    assert TRANSIENT_RULE in isolated
    assert PERSISTENT not in isolated
    assert BACKGROUND not in isolated


@pytest.mark.parametrize(
    "transient",
    [
        ["(Malformed alice)"],
        ["(: bad (Explodes alice) (STV 1.0 1.0))"],
    ],
)
def test_transient_validation_and_add_failures_return_no_proof(
    monkeypatch, tmp_path, transient
):
    reasoner, atomspace = fake_reasoner(monkeypatch, tmp_path)
    before = atomspace.read_bytes()

    assert reasoner.query(QUERY, transient_statements=transient) == []
    assert atomspace.read_bytes() == before
    assert all(atom not in reasoner._handler.atoms for atom in transient)


def test_transient_validation_precedes_persistent_exact_fact_lookup(monkeypatch, tmp_path):
    reasoner, _ = fake_reasoner(monkeypatch, tmp_path)
    exact_query = "(: $prf (Seed alice) $tv)"

    assert reasoner.query(exact_query) == [PERSISTENT]
    assert reasoner.query(
        exact_query, transient_statements=["(Malformed alice)"]
    ) == []


def test_persistent_reasoner_boundary_rejects_malformed_statements(monkeypatch, tmp_path):
    reasoner, atomspace = fake_reasoner(monkeypatch, tmp_path)
    before = atomspace.read_bytes()

    added, rejected = reasoner.add_statements_report(["(Malformed alice)"])

    assert added == []
    assert rejected == [{"stmt": "(Malformed alice)", "error": "missing_prefix"}]
    assert atomspace.read_bytes() == before
    assert "(Malformed alice)" not in reasoner._handler.atoms


def test_transient_query_failure_returns_no_proof(monkeypatch, tmp_path):
    reasoner, atomspace = fake_reasoner(monkeypatch, tmp_path)
    before = atomspace.read_bytes()

    assert reasoner.query(
        "(: $prf (Failure alice) $tv)", transient_statements=[TRANSIENT_RULE]
    ) == []
    assert atomspace.read_bytes() == before
    assert TRANSIENT_RULE not in reasoner._handler.atoms


def test_real_chainer_derives_transiently_without_mutation_or_leakage(
    reasoner, temp_atomspace, tmp_path
):
    background = tmp_path / "background.metta"
    background.write_text(BACKGROUND + "\n", encoding="utf-8")
    reasoner.add_statements([PERSISTENT])
    reasoner.load_background_file(str(background))
    before = temp_atomspace.read_bytes()

    proof = reasoner.query(QUERY, transient_statements=[TRANSIENT_RULE])

    assert proof
    assert temp_atomspace.read_bytes() == before
    assert reasoner.query(QUERY) == []
