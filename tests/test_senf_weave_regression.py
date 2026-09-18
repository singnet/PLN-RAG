import re

from config import get_settings
from core.senf.bridge import executable_bridge_atoms, transport_truth
from core.senf.extractor import extract_senf
from core.senf.identity import IdentityEdge, IdentityGraph
from core.senf.weave import build_weaves
from parsers.canonical_pln_parser import CanonicalPLNParser
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


def _senf(sentence_id: str, text: str, atoms: list[str]):
    return extract_senf(sentence_id, text, atoms)


def _truth_value(atom: str) -> tuple[float, float]:
    match = re.search(r"\(STV ([0-9.]+) ([0-9.]+)\)\)$", atom)
    assert match is not None
    return float(match.group(1)), float(match.group(2))


def test_reversing_sources_preserves_beam_outputs(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_per_source_k", 2)
    monkeypatch.setattr(settings, "senf_weave_beam_width", 3)
    query = _senf("q1", "Did someone move?", ["(: q (Moved person) $tv)"])
    sources = [
        _senf(
            sentence_id,
            "Two things moved.",
            [
                "(: a (Moved alpha) (STV 1 1))",
                "(: b (Moved beta) (STV 1 1))",
            ],
        )
        for sentence_id in ("s2", "s1")
    ]

    forward = build_weaves(query, sources, k=3)
    reversed_order = build_weaves(query, list(reversed(sources)), k=3)

    assert forward == reversed_order
    assert [result.guard for result in forward] == [
        result.guard for result in reversed_order
    ]


def test_beam_results_do_not_claim_solver_residuals():
    query = _senf(
        "q1",
        "Did alpha move and beta leave?",
        ["(: q1 (Moved alpha) $tv)", "(: q2 (Left beta) $tv)"],
    )
    source = _senf(
        "s1", "Alpha moved.", ["(: a (Moved alpha) (STV 1 1))"]
    )

    results = build_weaves(query, [source], k=3)

    assert {result.distortion for result in results} == {0.5, 1.0}
    assert all(
        result.residuals is None for result in results
    )


def test_executable_bridge_cost_includes_aggregate_and_mapping_costs():
    source = _senf(
        "s1",
        "The camera is at the lab.",
        ["(: source (AtLocation camera lab) (STV 1 1))"],
    )
    query = _senf(
        "q1",
        "Is it located in the lab?",
        ["(: query (LocatedIn it lab) $tv)"],
    )
    edges = tuple(
        IdentityEdge(
            source_mention,
            query_mention,
            strength=1.0,
            confidence=1.0,
            positive_cost=0.2,
        )
        for source_mention, query_mention in zip(source.mentions, query.mentions)
    )
    graph = IdentityGraph(
        edges=edges,
        merged=edges,
        representatives={
            source_mention.entity_id: query_mention.entity_id
            for source_mention, query_mention in zip(source.mentions, query.mentions)
        },
    )
    result = next(
        item
        for item in build_weaves(query, [source], k=3, identity_graph=graph)
        if item.aligned
    )

    adapter = executable_bridge_atoms(result, source, query, graph)
    matching_mapping_cost = sum(mapping.cost for mapping in result.entity_maps)
    expected = transport_truth(
        result.branch_probability,
        result.branch_probability,
        result.total_cost + matching_mapping_cost,
    )

    assert len(adapter) == 1
    assert matching_mapping_cost > 0.0
    assert _truth_value(adapter[0]) == (expected.strength, expected.weight)


def test_build_weaves_merges_bounded_per_source_results(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_per_source_k", 2)
    query = _senf("q1", "Did alpha move?", ["(: q (Moved alpha) $tv)"])
    sources = [
        _senf(sentence_id, "Alpha moved.", ["(: a (Moved alpha) (STV 1 1))"])
        for sentence_id in ("s2", "s1")
    ]

    per_source = [build_weaves(query, [source], k=99) for source in sources]
    combined = build_weaves(query, sources, k=3)
    pooled = [result for source_results in per_source for result in source_results]
    expected = sorted(
        pooled,
        key=lambda result: (
            bool(result.rejected_pairs),
            result.total_cost,
            result.distortion,
            result.guard,
        ),
    )[:3]

    assert [len(results) for results in per_source] == [2, 2]
    assert all(
        all(result.guard.startswith(f"{source.sentence_id}->") for result in results)
        for source, results in zip(sources, per_source)
    )
    assert combined == tuple(expected)


def test_no_prior_senf_planning_matches_base_parser(monkeypatch):
    monkeypatch.setattr(CanonicalPLNParser, "__init__", lambda self: None)
    parser = CanonicalSENFPLNParser()
    candidates = [
        "(: $prf (Smart kebede) $tv)",
        "(: $prf (Tall kebede) $tv)",
    ]
    context = [
        "(: smart (Smart kebede) (STV 1 1))",
        "(: tall (Tall kebede) (STV 1 1))",
    ]

    expected = CanonicalPLNParser._plan_queries(
        parser, "Is Kebede smart?", candidates, [], context
    )

    assert parser._query_prior == ()
    assert parser._plan_queries(
        "Is Kebede smart?", candidates, [], context
    ) == expected
