import pytest

from core.senf.extractor import extract_senf
from core.senf.temporal import (
    BranchingContextTree,
    assess_transport,
    compile_branch_theory,
    interval_relation,
    temporal_decay,
)
from core.senf.types import (
    ACTUAL_BRANCH_ID,
    BranchContext,
    Context,
    EntityPersistence,
    ValidityInterval,
    senf_from_payload,
    senf_to_payload,
)


def branch_tree():
    return BranchingContextTree([
        BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0),
        BranchContext("forecast", ACTUAL_BRANCH_ID, "projected", 0.8),
        BranchContext("rain", "forecast", "counterfactual", 0.5),
        BranchContext("dry", "forecast", "counterfactual", 0.5),
    ])


def test_branch_tree_has_deterministic_lineage_lca_and_probability():
    tree = branch_tree()

    assert tree.lineage("rain") == (ACTUAL_BRANCH_ID, "forecast", "rain")
    assert tree.least_common_ancestor("rain", "dry") == "forecast"
    assert tree.path_probability(ACTUAL_BRANCH_ID, "rain") == 0.4


@pytest.mark.parametrize(
    "branches",
    [
        [
            BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0),
            BranchContext("orphan", "missing", "projected", 0.5),
        ],
        [
            BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0),
            BranchContext("left", "right", "projected", 0.5),
            BranchContext("right", "left", "projected", 0.5),
        ],
    ],
)
def test_invalid_branch_ancestry_fails_closed(branches):
    with pytest.raises(ValueError):
        BranchingContextTree(branches)


def test_interval_relations_and_decay_are_typed_and_monotonic():
    early = ValidityInterval("early", "2026-01-01", "2026-01-02")
    near = ValidityInterval("near", "2026-01-03", "2026-01-04")
    far = ValidityInterval("far", "2026-02-01", "2026-02-02")

    assert interval_relation(early, near) == "before"
    assert temporal_decay(early, near, 0.1) > temporal_decay(early, far, 0.1)


def test_fact_transport_inherits_downward_but_never_into_actual_or_siblings():
    tree = branch_tree()
    actual = Context("u0")
    rain = Context("u1", branch_id="rain")
    dry = Context("u2", branch_id="dry")

    assert assess_transport(tree, actual, rain, {}).allowed
    assert not assess_transport(tree, rain, actual, {}).allowed
    assert not assess_transport(tree, rain, dry, {}).allowed


def test_rigid_identity_does_not_authorize_cross_world_fact_transport():
    tree = branch_tree()
    rigid = EntityPersistence("e1", "rigid", branch_id="rain")
    rain = Context("u1", branch_id="rain")
    dry = Context("u2", branch_id="dry")

    assert assess_transport(tree, rain, dry, {}, rigid, purpose="identity").allowed
    assert not assess_transport(tree, rain, dry, {}, rigid, purpose="fact").allowed


def test_stage7_extraction_and_v5_round_trip():
    atoms = [
        "(: branch (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
        "(: interval (ValidityInterval t1 2026-01-01 2026-12-31) (STV 1 1))",
        "(: policy (EntityPersistence camera rigid realized rain t1) (STV 1 1))",
        "(: event (InContext rain t1 (Returned sam camera)) (STV 1 1))",
    ]

    senf = extract_senf("s1", "Sam would return the camera.", atoms)
    frame = senf.frames[0]

    assert frame.context.branch_id == "rain"
    assert frame.context.validity_interval_id == "t1"
    assert senf.entity_persistence[0].persistence_type == "rigid"
    assert senf_from_payload(senf_to_payload(senf)) == senf


def test_branch_theory_compiles_only_the_target_lineage_with_degraded_tvs():
    senf = extract_senf(
        "s1",
        "In the rain branch, the ground is wet and therefore slippery.",
        [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: dry (BranchContext dry actual_root counterfactual 0.6) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
            "(: rule (InContext rain none (Implication (Premises (Wet $x)) "
            "(Conclusions (Slippery $x)))) (STV 1 1))",
            "(: sibling (InContext dry none (Dry ground)) (STV 1 1))",
        ],
    )

    theory, decisions = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain")
    )

    assert len(theory) == 2
    assert all("(STV 0.4 0.4)" in atom for atom in theory)
    assert any("(Wet ground)" in atom for atom in theory)
    assert any("(Slippery $x)" in atom for atom in theory)
    assert all("(Dry ground)" not in atom for atom in theory)
    assert any(not decision.allowed for decision in decisions)


def test_v4_payload_is_rejected_without_migration():
    senf = extract_senf(
        "s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"]
    )
    payload = senf_to_payload(senf)
    payload["senf_version"] = 4

    assert senf_from_payload(payload) is None


def test_unfulfilled_entity_in_actual_branch_is_invalid():
    senf = extract_senf(
        "s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1 1))"]
    )
    senf.entity_persistence = [
        EntityPersistence(senf.entities[0].entity_id, "contingent", "unfulfilled")
    ]

    assert senf_from_payload(senf_to_payload(senf)) is None


def test_branch_theory_uses_the_source_branch_entity_status():
    senf = extract_senf(
        "s1", "A camera exists only in one branch.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.5) (STV 1 1))",
            "(: dry (BranchContext dry actual_root counterfactual 0.5) (STV 1 1))",
            "(: rain_camera (EntityPersistence camera contingent unfulfilled rain none) (STV 1 1))",
            "(: dry_camera (EntityPersistence camera contingent realized dry none) (STV 1 1))",
            "(: exists (InContext rain none (Exists camera)) (STV 1 1))",
        ],
    )

    theory, decisions = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain")
    )

    assert theory == []
    assert any(decision.reason == "unfulfilled_entity" for decision in decisions)


def test_branch_theory_rejects_entity_outside_its_persistence_interval():
    senf = extract_senf(
        "s1", "A camera exists during the early interval.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.5) (STV 1 1))",
            "(: early (ValidityInterval early 2026-01-01 2026-01-31) (STV 1 1))",
            "(: late (ValidityInterval late 2026-03-01 2026-03-31) (STV 1 1))",
            "(: camera_policy (EntityPersistence camera temporal realized rain early) (STV 1 1))",
            "(: exists (InContext rain late (Exists camera)) (STV 1 1))",
        ],
    )

    theory, decisions = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain", validity_interval_id="late")
    )

    assert theory == []
    assert any(
        decision.reason == "persistence_interval_mismatch" for decision in decisions
    )


def test_conflicting_declarations_fail_extraction_closed():
    senf = extract_senf(
        "s1", "Conflicting branch declarations.", [
            "(: first (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: second (BranchContext rain actual_root counterfactual 0.6) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
        ],
    )

    assert senf.is_empty


@pytest.mark.parametrize("probability", ["-0.1", "1.1", "nan", "inf"])
def test_invalid_branch_probabilities_fail_extraction_closed(probability):
    senf = extract_senf(
        "s1", "An invalid branch.", [
            f"(: rain (BranchContext rain actual_root counterfactual {probability}) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
        ],
    )

    assert senf.is_empty


def test_zero_probability_branch_compiles_no_proof_witness():
    senf = extract_senf(
        "s1", "An impossible rain branch.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.0) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV 1 1))",
        ],
    )

    theory, _ = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain")
    )

    assert theory == []


def test_nonfinite_truth_values_compile_no_proof_witness():
    senf = extract_senf(
        "s1", "An invalid weighted branch fact.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.5) (STV 1 1))",
            "(: wet (InContext rain none (Wet ground)) (STV nan inf))",
        ],
    )

    theory, _ = compile_branch_theory(
        [senf], Context("query:u0", branch_id="rain")
    )

    assert theory == []


def test_interval_relation_normalizes_mixed_timezone_forms():
    naive = ValidityInterval("naive", "2026-01-01", "2026-01-02")
    aware = ValidityInterval(
        "aware", "2026-01-03T00:00:00+03:00", "2026-01-04T00:00:00+03:00"
    )

    assert interval_relation(naive, aware) == "before"


def test_target_branch_status_blocks_inherited_actual_facts_across_records():
    actual = extract_senf(
        "actual", "The camera exists.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: exists (Exists camera) (STV 1 1))",
        ],
    )
    branch_policy = extract_senf(
        "policy", "The camera is unfulfilled in rain.", [
            "(: rain (BranchContext rain actual_root counterfactual 0.4) (STV 1 1))",
            "(: policy (EntityPersistence camera contingent unfulfilled rain none) (STV 1 1))",
            "(: imagined (InContext rain none (Imagined camera)) (STV 1 1))",
        ],
    )

    theory, decisions = compile_branch_theory(
        [actual, branch_policy], Context("query:u0", branch_id="rain")
    )

    assert theory == []
    assert any(decision.reason == "unfulfilled_entity" for decision in decisions)
