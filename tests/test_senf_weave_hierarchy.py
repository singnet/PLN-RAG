import pytest
from pydantic import ValidationError

from config import Settings, get_settings
from core.senf.bridge import executable_bridge_atoms
from core.senf.extractor import extract_senf
from core.senf.identity import IdentityEdge, IdentityGraph
from core.senf.weave import EntityMap, FramePair, PredicateMap, WeaveResult, build_weaves, weave
from core.senf.weave_hierarchy import HierarchyLimits, _round, _solve
from core.senf.weave_model import (
    FrameAlignment,
    PairComponentCosts,
    SourceFrameKey,
)
from parsers.canonical_senf_pln_parser import CanonicalSENFPLNParser


def _senf(sentence_id: str, atoms: list[str]):
    return extract_senf(sentence_id, sentence_id, atoms)


@pytest.fixture(autouse=True)
def hierarchical_engine(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_engine", "hierarchical")
    monkeypatch.setattr(settings, "senf_weave_max_cost", 2.0)


def test_global_polish_combines_support_from_multiple_sources():
    query = _senf(
        "q1",
        ["(: q1 (Arrived alpha) $tv)", "(: q2 (Left beta) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Left beta) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert {pair.query_frame_id for pair in result.pairs} == {"q1:f0", "q1:f1"}
    assert {item.source_frame.source_id for item in result.alignments} == {
        "senf:s1", "senf:s2"
    }
    assert result.guard == "s1+s2->q1"
    assert result.polish is not None
    assert result.polish.engine == "hierarchical"


def test_global_source_competition_rounds_one_qualified_source_frame():
    query = _senf("q1", ["(: q (Moved alpha) $tv)"])
    sources = [
        _senf("s2", ["(: b (Moved beta) (STV 1 1))"]),
        _senf("s1", ["(: a (Moved alpha) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert len(result.alignments) == 1
    assert result.alignments[0].source_frame.source_id == "senf:s1"


def test_hierarchy_is_deterministic_under_source_reversal():
    query = _senf(
        "q1", ["(: q1 (Moved alpha) $tv)", "(: q2 (Moved beta) $tv)"],
    )
    sources = [
        _senf("s2", ["(: b (Moved beta) (STV 1 1))"]),
        _senf("s1", ["(: a (Moved alpha) (STV 1 1))"]),
    ]

    assert build_weaves(query, sources, k=4) == build_weaves(
        query, list(reversed(sources)), k=4
    )


def test_residuals_are_actual_stage_residuals():
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert result.polish is not None
    assert result.residuals == result.polish.residuals
    assert result.polish.local.solves == 1
    assert result.polish.block.solves == 1
    assert result.polish.global_stage.solves == 1
    assert max(result.residuals) <= get_settings().senf_weave_sinkhorn_tolerance


def test_hard_predicate_exclusions_are_not_relaxed_by_polish():
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Left alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert not result.aligned
    assert result.polish is not None
    assert result.polish.fallback_reason == "no_hard_compatible_candidates"


def test_global_candidate_cap_is_enforced(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_global_candidate_cap", 1)
    query = _senf("q1", ["(: q (Moved person) $tv)"])
    source = _senf(
        "s1",
        [
            "(: a (Moved alpha) (STV 1 1))",
            "(: b (Moved beta) (STV 1 1))",
        ],
    )

    result = weave(query, [source])

    assert result.polish is not None
    assert result.polish.candidate_count == 1


def test_cell_bound_falls_back_unaligned_without_beam(monkeypatch):
    monkeypatch.setattr(get_settings(), "senf_weave_max_cells", 1)
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert not result.aligned
    assert result.alignments == ()
    assert result.polish is not None
    assert result.polish.fallback_reason == "sinkhorn_max_cells"


def test_fallback_respects_total_seed_bound_and_retains_unaligned(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_max_cells", 1)
    monkeypatch.setattr(settings, "senf_weave_max_seeds", 1)
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    results = build_weaves(query, [source], k=10)

    assert len(results) == 1
    assert not results[0].aligned
    assert results[0].rejected_pairs == ()
    assert results[0].polish is not None
    assert results[0].polish.fallback_reason == "sinkhorn_max_cells"


def test_hierarchical_engine_never_runs_legacy_beam(monkeypatch):
    import core.senf.weave as weave_module

    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    def fail(*args, **kwargs):
        raise AssertionError("legacy beam must not run in hierarchical mode")

    monkeypatch.setattr(weave_module, "_weave_source", fail)

    assert weave_module.weave(query, [source]).aligned


def test_source_cap_is_checked_before_pair_hypotheses(monkeypatch):
    import core.senf.weave as weave_module

    monkeypatch.setattr(get_settings(), "senf_query_max_priors", 1)
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    sources = [
        _senf(sentence_id, ["(: a (Arrived alpha) (STV 1 1))"])
        for sentence_id in ("s1", "s2")
    ]

    def fail(*args, **kwargs):
        raise AssertionError("pair hypotheses must not run beyond the source cap")

    monkeypatch.setattr(weave_module, "_pair_hypotheses", fail)
    result = weave_module.weave(query, sources)

    assert not result.aligned
    assert result.polish is not None
    assert result.polish.fallback_reason == "hierarchy_source_cap"


def test_source_frame_cap_is_checked_before_pair_hypotheses(monkeypatch):
    import core.senf.weave as weave_module

    monkeypatch.setattr(get_settings(), "senf_query_max_source_frames", 1)
    query = _senf("q1", ["(: q (Moved alpha) $tv)"])
    source = _senf(
        "s1",
        [
            "(: a (Moved alpha) (STV 1 1))",
            "(: b (Moved beta) (STV 1 1))",
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("pair hypotheses must not run beyond the frame cap")

    monkeypatch.setattr(weave_module, "_pair_hypotheses", fail)
    result = weave_module.weave(query, [source])

    assert not result.aligned
    assert result.polish is not None
    assert result.polish.fallback_reason == "hierarchy_source_frame_cap"


def test_pair_work_cap_is_checked_before_pair_hypotheses(monkeypatch):
    import core.senf.weave as weave_module

    settings = get_settings()
    monkeypatch.setattr(settings, "senf_query_max_priors", 1)
    monkeypatch.setattr(settings, "senf_weave_global_candidate_cap", 1)
    query = _senf(
        "q1", ["(: q1 (Moved alpha) $tv)", "(: q2 (Left alpha) $tv)"]
    )
    source = _senf("s1", ["(: a (Moved alpha) (STV 1 1))"])

    def fail(*args, **kwargs):
        raise AssertionError("pair hypotheses must not run beyond the work cap")

    monkeypatch.setattr(weave_module, "_pair_hypotheses", fail)
    result = weave_module.weave(query, [source])

    assert not result.aligned
    assert result.polish is not None
    assert result.polish.fallback_reason == "hierarchy_pair_work_cap"


def test_hierarchy_exposes_typed_alignment_costs():
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert isinstance(result.alignments[0], FrameAlignment)
    assert isinstance(result.alignments[0].costs, PairComponentCosts)


def test_empty_sources_preserve_build_weaves_contract():
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])

    assert build_weaves(query, []) == ()


def test_candidate_cap_considers_better_later_sources(monkeypatch):
    monkeypatch.setattr(get_settings(), "senf_weave_global_candidate_cap", 1)
    query = _senf("q1", ["(: q (Moved alpha) $tv)"])
    sources = [
        _senf("s1", ["(: a (Moved beta) (STV 1 1))"]),
        _senf("s2", ["(: b (Moved alpha) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert result.pairs[0].source_id == "senf:s2"


def test_hierarchy_candidates_do_not_depend_on_per_source_beam(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_per_source_k", 1)
    query = _senf(
        "q1",
        ["(: q1 (LocatedIn alpha room) $tv)", "(: q2 (Left gamma) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (AtLocation beta room) (STV 1 1))"]),
        _senf("s2", ["(: b (Left gamma) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert {pair.query_frame_id for pair in result.pairs} == {"q1:f0", "q1:f1"}


def test_global_cap_allocates_first_candidates_across_queries(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_global_candidate_cap", 2)
    query = _senf(
        "q1", ["(: q1 (Moved alpha) $tv)", "(: q2 (Moved beta) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (Moved alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Moved beta) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert result.polish is not None
    assert result.polish.candidate_count == 2
    assert {item.query_frame_id for item in result.alignments} == {"q1:f0", "q1:f1"}


def test_global_cap_allocates_first_candidates_across_sources(monkeypatch):
    from core.senf import weave_hierarchy

    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_global_candidate_cap", 2)
    query = _senf("q1", ["(: q (Moved person) $tv)"])
    sources = [
        _senf("s1", ["(: a1 (Moved alpha) (STV 1 1))", "(: a2 (Moved other) (STV 1 1))"]),
        _senf("s2", ["(: b1 (Moved beta) (STV 1 1))", "(: b2 (Moved another) (STV 1 1))"]),
    ]
    captured = []
    original = weave_hierarchy.polish_hierarchy

    def capture(candidates, query_frame_ids, limits):
        captured.extend(candidates)
        return original(candidates, query_frame_ids, limits)

    monkeypatch.setattr(weave_hierarchy, "polish_hierarchy", capture)

    weave(query, sources)

    assert {item.source_frame.source_id for item in captured} == {"senf:s1", "senf:s2"}


def test_cross_source_corroboration_is_not_an_entity_conflict():
    query = _senf(
        "q1",
        ["(: q1 (Arrived alpha) $tv)", "(: q2 (Left alpha) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Left alpha) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert len(result.pairs) == 2
    assert result.distortion_cost == 0.0


def test_cross_source_incompatible_entities_are_a_consistency_conflict():
    query = _senf(
        "q1",
        ["(: q1 (Arrived alpha) $tv)", "(: q2 (Left alpha) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Left beta) (STV 1 1))"]),
    ]

    result = weave(query, sources)

    assert len(result.pairs) == 2
    assert result.distortion_cost > 0.0


def test_cross_source_identity_allows_different_canonical_symbols():
    query = _senf(
        "q1",
        ["(: q1 (Arrived person) $tv)", "(: q2 (Left person) $tv)"],
    )
    sources = [
        _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Left alias) (STV 1 1))"]),
    ]
    edges = tuple(
        IdentityEdge(source.mentions[0], query.mentions[0], 1.0, 1.0, 0.0)
        for source in sources
    )
    graph = IdentityGraph(
        edges=edges,
        merged=edges,
        representatives={
            source.mentions[0].entity_id: query.mentions[0].entity_id
            for source in sources
        },
    )

    result = weave(query, sources, identity_graph=graph)

    assert len(result.pairs) == 2
    assert result.distortion_cost == 0.0


def test_multisource_bridges_resolve_each_qualified_pair():
    query = _senf(
        "q1",
        [
            "(: q1 (Arrived it) $tv)",
            "(: q2 (Left them) $tv)",
        ],
    )
    sources = [
        _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"]),
        _senf("s2", ["(: b (Left beta) (STV 1 1))"]),
    ]
    edges = tuple(
        IdentityEdge(source.mentions[0], query.mentions[index], 1.0, 1.0, 0.0)
        for index, source in enumerate(sources)
    )
    graph = IdentityGraph(
        edges=edges,
        merged=edges,
        representatives={
            source.mentions[0].entity_id: query.mentions[index].entity_id
            for index, source in enumerate(sources)
        },
    )
    result = weave(query, sources, identity_graph=graph)

    adapters = executable_bridge_atoms(result, sources, query, graph)

    assert {pair.source_id for pair in result.pairs} == {"senf:s1", "senf:s2"}
    assert len(adapters) == 2
    assert any("(Arrived alpha)" in atom for atom in adapters)
    assert any("(Left beta)" in atom for atom in adapters)


def test_sinkhorn_unmatched_assignment_costs_exactly_one():
    limits = HierarchyLimits(8, 64, 4, 200, 1e-7, 0.25)
    candidate = FrameAlignment(
        "q", SourceFrameKey("senf:s", "s:f0"), 0.0,
        PairComponentCosts(structural=1.0),
    )

    _, weights = _solve((candidate,), limits)

    assert weights[("q", candidate.source_frame)] == pytest.approx(0.5)


def test_later_polish_does_not_reapply_semantic_cost():
    limits = HierarchyLimits(8, 64, 4, 200, 1e-7, 0.25)
    cheap = FrameAlignment(
        "q1", SourceFrameKey("senf:s", "s:f0"), 1.0,
        PairComponentCosts(),
    )
    expensive = FrameAlignment(
        "q2", SourceFrameKey("senf:s", "s:f1"), 0.0,
        PairComponentCosts(structural=1.0),
    )
    prior = {
        (cheap.query_frame_id, cheap.source_frame): 0.25,
        (expensive.query_frame_id, expensive.source_frame): 0.25,
    }

    _, weights = _solve((cheap, expensive), limits, prior)

    assert weights[(cheap.query_frame_id, cheap.source_frame)] == pytest.approx(
        weights[(expensive.query_frame_id, expensive.source_frame)]
    )


def test_hard_rounding_uses_polished_mass_before_semantic_cost():
    cheap = FrameAlignment(
        "q", SourceFrameKey("senf:s1", "s1:f0"), 1.0,
        PairComponentCosts(),
    )
    polished = FrameAlignment(
        "q", SourceFrameKey("senf:s2", "s2:f0"), 0.0,
        PairComponentCosts(structural=0.8),
    )
    weights = {
        ("q", cheap.source_frame): 0.1,
        ("q", polished.source_frame): 0.8,
        ("q", None): 0.1,
    }

    selections = _round((cheap, polished), weights, ("q",), 2, 16)

    assert selections == ((polished,), ())


def test_hard_rounding_finds_global_optimum_without_prefix_pruning():
    a = SourceFrameKey("senf:s", "a")
    b = SourceFrameKey("senf:s", "b")
    c = SourceFrameKey("senf:s", "c")
    candidates = (
        FrameAlignment("q1", a, 1.0, PairComponentCosts()),
        FrameAlignment("q1", b, 1.0, PairComponentCosts()),
        FrameAlignment("q2", a, 1.0, PairComponentCosts()),
        FrameAlignment("q2", c, 1.0, PairComponentCosts()),
        FrameAlignment("q3", a, 1.0, PairComponentCosts()),
        FrameAlignment("q3", b, 1.0, PairComponentCosts()),
        FrameAlignment("q3", c, 1.0, PairComponentCosts()),
    )
    weights = {
        ("q1", a): 0.40,
        ("q1", b): 0.35,
        ("q1", None): 0.25,
        ("q2", a): 0.45,
        ("q2", c): 0.30,
        ("q2", None): 0.25,
        ("q3", a): 0.15,
        ("q3", b): 0.30,
        ("q3", c): 0.40,
        ("q3", None): 0.15,
    }

    selections = _round(candidates, weights, ("q1", "q2", "q3"), 2, 64)

    assert {
        (item.query_frame_id, item.source_frame.frame_id)
        for item in selections[0]
    } == {("q1", "b"), ("q2", "a"), ("q3", "c")}
    assert selections[-1] == ()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("senf_weave_global_candidate_cap", 0),
        ("senf_weave_max_cells", 0),
        ("senf_weave_max_seeds", 0),
        ("senf_weave_max_iterations", 0),
        ("senf_weave_sinkhorn_tolerance", float("nan")),
        ("senf_weave_sinkhorn_regularization", float("inf")),
    ],
)
def test_hierarchy_settings_reject_invalid_values(name, value):
    with pytest.raises(ValidationError):
        Settings(openai_api_key="test", **{name: value})


@pytest.mark.parametrize(
    "name",
    ["senf_weave_sinkhorn_tolerance", "senf_weave_sinkhorn_regularization"],
)
def test_runtime_validation_rejects_nonfinite_numeric_settings(monkeypatch, name):
    monkeypatch.setattr(get_settings(), name, float("inf"))
    query = _senf("q1", ["(: q (Arrived alpha) $tv)"])
    source = _senf("s1", ["(: a (Arrived alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert not result.aligned
    assert result.polish is not None
    assert result.polish.fallback_reason == "invalid_hierarchy_numeric_setting"


def test_max_seeds_one_keeps_unaligned_over_rejected_result(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "senf_weave_max_seeds", 1)
    monkeypatch.setattr(settings, "senf_weave_max_cost", 0.05)
    query = _senf("q1", ["(: q (Moved alpha) $tv)"])
    source = _senf("s1", ["(: a (Moved alpha) (STV 1 1))"])

    result = weave(query, [source])

    assert not result.aligned
    assert result.rejected_pairs == ()
    assert result.polish is not None
    assert result.polish.seed_count == 1
    assert len(build_weaves(query, [source], k=10)) == 1


def test_candidate_variants_never_mix_mappings_from_different_pairs():
    candidate = "(: $prf (LocatedIn it lab) $tv)"
    query = _senf("q1", [candidate])
    sources = [
        _senf("s1", ["(: a (AtLocation alpha lab) (STV 1 1))"]),
        _senf("s2", ["(: b (LocatedAt beta lab) (STV 1 1))"]),
    ]
    edges = tuple(
        IdentityEdge(source.mentions[0], query.mentions[0], 1.0, 1.0, 0.0)
        for source in sources
    )
    graph = IdentityGraph(
        edges=edges,
        merged=edges,
        representatives={
            source.mentions[0].entity_id: query.mentions[0].entity_id
            for source in sources
        },
    )
    pairs = tuple(
        FramePair(
            query.frames[0].frame_id,
            source.frames[0].frame_id,
            1.0,
            source_id=source.senf_id,
        )
        for source in sources
    )
    entity_maps = tuple(
        EntityMap(
            source.mentions[0].entity_id,
            query.mentions[0].entity_id,
            source.mentions[0].mention_id,
            query.mentions[0].mention_id,
            source.mentions[0].canonical_symbol,
            query.mentions[0].canonical_symbol,
            0.0,
            True,
            source.senf_id,
            source.frames[0].frame_id,
            query.frames[0].frame_id,
        )
        for source in sources
    )
    predicate_maps = tuple(
        PredicateMap(
            source.frames[0].predicate_head,
            query.frames[0].predicate_head,
            0.25,
            source.senf_id,
            source.frames[0].frame_id,
            query.frames[0].frame_id,
        )
        for source in sources
    )
    result = WeaveResult(
        pairs=pairs,
        entity_maps=entity_maps,
        predicate_maps=predicate_maps,
    )
    parser = object.__new__(CanonicalSENFPLNParser)
    parser._emit_bridge_atoms = False

    variants = parser._candidate_variants(candidate, query, graph, (result,))

    assert "(: $prf (AtLocation alpha lab) $tv)" in variants
    assert "(: $prf (LocatedAt beta lab) $tv)" in variants
    assert "(: $prf (AtLocation beta lab) $tv)" not in variants
    assert "(: $prf (LocatedAt alpha lab) $tv)" not in variants
