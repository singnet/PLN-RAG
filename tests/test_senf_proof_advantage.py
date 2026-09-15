import re
from dataclasses import replace
from types import SimpleNamespace

from core.parser import ParseResult
from core.senf.bridge import executable_bridge_atoms
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.temporal import compile_branch_theory
from core.senf.types import Context
from core.senf.weave import weave
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


SOURCE_ATOM = "(: source (AtLocation nikon_camera lab) (STV 1.0 1.0))"
CANONICAL_QUERY = "(: $prf (LocatedIn it lab) $tv)"


def _case_e():
    source = extract_senf(
        "s1", "The Nikon camera is at the lab.", [SOURCE_ATOM]
    )
    query = extract_senf(
        "s2", "Is it located in the lab?", [CANONICAL_QUERY]
    )
    graph = resolve_identity([source, query])
    result = weave(query, [source], identity_graph=graph)
    return source, query, graph, result


def _tv(atom: str) -> tuple[float, float]:
    match = re.search(r"\(STV ([0-9.]+) ([0-9.]+)\)\)$", atom)
    assert match
    return float(match.group(1)), float(match.group(2))


def test_case_e_canonical_fails_but_justified_senf_adapter_proves(reasoner):
    source, query, graph, result = _case_e()
    adapters = executable_bridge_atoms(result, source, query, graph)
    reasoner.add_statements([SOURCE_ATOM])

    assert graph.same_entity(
        result.entity_maps[0].source_entity_id,
        result.entity_maps[0].target_entity_id,
    )
    assert reasoner.query(CANONICAL_QUERY) == []
    assert adapters
    assert reasoner.query(CANONICAL_QUERY, transient_statements=adapters)
    assert reasoner.query(CANONICAL_QUERY) == []


def test_case_f_low_cost_adapter_has_higher_tv_and_is_consumed_transiently(reasoner):
    source, query, graph, result = _case_e()
    low = executable_bridge_atoms(replace(result, total_cost=0.1), source, query, graph)[0]
    high = executable_bridge_atoms(replace(result, total_cost=1.2), source, query, graph)[0]
    reasoner.add_statements([SOURCE_ATOM])

    assert _tv(low)[0] > _tv(high)[0]
    assert _tv(low)[1] > _tv(high)[1]
    low_proof = reasoner.query(CANONICAL_QUERY, transient_statements=[low])
    high_proof = reasoner.query(CANONICAL_QUERY, transient_statements=[high])
    assert low_proof
    assert high_proof
    assert low_proof != high_proof
    assert str(_tv(low)[0]) in str(low_proof)
    assert str(_tv(high)[0]) in str(high_proof)
    assert reasoner.query(CANONICAL_QUERY) == []


def test_actual_case_e_parser_execution_proves_canonical_target_transiently(
    reasoner, monkeypatch
):
    lent_atoms = [
        "(: lend (Lent alex sam nikon_camera) (STV 1 1))",
        "(: lent_kind (IsA nikon_camera Camera) (STV 1 1))",
    ]
    returned_atoms = [
        "(: return (Returned sam it) (STV 1 1))",
        "(: returned_kind (IsA it Camera) (STV 1 1))",
    ]
    target = "(: $prf (Returned sam nikon_camera) $tv)"

    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    parser._use_vector_context = False
    parser._emit_bridge_atoms = True
    parser._pln_spec = None
    parser._nl2pln = None
    parser._generation_backend = SimpleNamespace(generate=lambda **_: ParseResult(
        queries=[target]
    ))

    for text, atoms in (
        ("Alex lent Sam the Nikon camera.", lent_atoms),
        ("Sam returned it.", returned_atoms),
    ):
        statements, _ = parser._post_filter_hook([text], atoms, [], [], False)
        ingest = ParseResult(statements=statements, parser_state=parser._pending_ingest)
        parser.prepare_ingest(ingest, statements)
        parser.commit_ingest(ingest)
        reasoner.add_statements(statements)

    assert reasoner.query(target) == []

    parsed = parser.parse_query("Did Sam return the Nikon camera?", returned_atoms)

    assert parsed.original_query == target
    assert parsed.queries[0] == target
    assert all("(Returned sam it)" not in query for query in parsed.queries)
    assert len(parsed.candidate_trusted_transient_statements) == len(parsed.queries)
    adapters = parsed.candidate_trusted_transient_statements[parsed.queries.index(target)]
    assert adapters
    assert all("senf_adapter" in adapter for adapter in adapters)
    assert reasoner.query(parsed.queries[0], transient_statements=adapters)
    assert reasoner.query(target) == []


def test_inverse_predicate_does_not_create_a_false_proof(reasoner):
    source_atom = "(: source (HasPart car wheel) (STV 1 1))"
    query_atom = "(: $prf (PartOf car wheel) $tv)"
    source = extract_senf("s1", "The car has a wheel.", [source_atom])
    query = extract_senf("q1", "Is the car part of the wheel?", [query_atom])
    result = weave(query, [source])
    adapters = executable_bridge_atoms(result, source, query)
    reasoner.add_statements([source_atom])

    assert result.pairs == ()
    assert adapters == []
    assert reasoner.query(query_atom, transient_statements=adapters) == []


def test_attributed_claim_cannot_prove_its_unqualified_proposition(reasoner):
    claim_atom = "(: claim (Claims casey (AtLocation camera lab)) (STV 1 1))"
    query_atom = "(: $prf (LocatedIn camera lab) $tv)"
    source = extract_senf("s1", "Casey claims the camera is at the lab.", [claim_atom])
    query = extract_senf("q1", "Is the camera in the lab?", [query_atom])
    result = weave(query, [source])
    adapters = executable_bridge_atoms(result, source, query)
    reasoner.add_statements([claim_atom])

    assert adapters == []
    assert reasoner.query(query_atom, transient_statements=adapters) == []


def test_counterfactual_rules_execute_only_inside_the_selected_branch(
    reasoner, monkeypatch
):
    branch_atoms = [
        "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
        "(: dry (BranchContext dry actual_root counterfactual 0.6) (STV 1 1))",
        "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
        "(: rule (InContext rain none (Implication (Premises (Wet $x)) "
        "(Conclusions (Slippery $x)))) (STV 1 1))",
    ]
    branch_query = "(: $prf (InContext rain none (Slippery ground)) $tv)"
    actual_query = "(: $prf (Slippery ground) $tv)"

    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    parser._use_vector_context = False
    parser._counterfactual_enabled = True
    parser._emit_bridge_atoms = False
    parser._pln_spec = None
    parser._nl2pln = None
    parser._generation_backend = SimpleNamespace(
        generate=lambda **_: ParseResult(queries=[branch_query])
    )

    statements, _ = parser._post_filter_hook(
        ["If it rained, the ground would be slippery."], branch_atoms, [], [], False
    )
    ingest = ParseResult(statements=statements, parser_state=parser._pending_ingest)
    assert parser.prepare_ingest(ingest, statements)
    parser.commit_ingest(ingest)
    reasoner.add_statements(statements)

    parsed = parser.parse_query(
        "In the rain counterfactual, would the ground be slippery?", branch_atoms
    )
    support = parsed.candidate_trusted_transient_statements[0]

    assert parsed.original_query == branch_query
    assert parsed.queries[0] == actual_query
    assert any("(Wet ground)" in atom for atom in support)
    assert any("(Slippery $x)" in atom for atom in support)
    assert reasoner.query(actual_query) == []
    assert reasoner.query(parsed.queries[0], transient_statements=support)
    assert reasoner.query(actual_query) == []


def test_counterfactual_fact_cannot_support_an_actual_query(reasoner):
    source = extract_senf(
        "s1",
        "In one counterfactual, the ground is wet.",
        [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
        ],
    )
    actual = extract_senf(
        "q1", "Is the ground wet?", ["(: $prf (Wet ground) $tv)"]
    )
    graph = resolve_identity([source, actual])
    result = weave(actual, [source], identity_graph=graph)

    assert not result.aligned
    assert executable_bridge_atoms(result, source, actual, graph) == []


def test_counterfactual_planner_rejects_unknown_intervals(monkeypatch):
    branch_atom = "(: rain (BranchContext rain actual_root counterfactual 0.5) (STV 1 1))"
    fact_atom = "(: wet (InContext rain none (Wet ground)) (STV 1 1))"
    query = "(: $prf (InContext rain missing (Wet ground)) $tv)"
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    parser._use_vector_context = False
    parser._counterfactual_enabled = True
    statements, _ = parser._post_filter_hook(
        ["There is a rain branch."], [branch_atom, fact_atom], [], [], False
    )
    ingest = ParseResult(statements=statements, parser_state=parser._pending_ingest)
    parser.prepare_ingest(ingest, statements)
    parser.commit_ingest(ingest)
    query_statements, queries = parser._post_filter_hook(
        ["Is it wet in the missing interval?"], [], [query], [], True
    )

    assert parser._plan_queries(
        "Is it wet in the missing interval?", queries, query_statements, []
    ) == []


def test_counterfactual_planner_rejects_multi_context_queries(monkeypatch):
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    parser._counterfactual_enabled = True
    _, queries = parser._post_filter_hook(
        ["Compare rain and dry."],
        [],
        [
            "(: $a (InContext rain none (Wet ground)) $tv)",
            "(: $b (InContext dry none (Wet ground)) $tv)",
        ],
        [],
        True,
    )

    assert queries == []
    assert parser.senf_telemetry()["stage7_rejection"] == "query spans multiple contexts"


def test_branch_query_uses_isolated_degraded_actual_ancestor_facts(reasoner):
    actual_fact = "(: actual_wet (Wet ground) (STV 1 1))"
    senf = extract_senf(
        "s1", "The ground is wet in actuality.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            actual_fact,
        ],
    )
    query = "(: $prf (Wet ground) $tv)"
    theory, _ = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain")
    )
    reasoner.add_statements([actual_fact])

    assert reasoner.query(query)
    assert reasoner.query_transient_only(query, []) == []
    assert any("(STV 0.4 0.4)" in atom for atom in theory)
    assert reasoner.query_transient_only(query, theory)
