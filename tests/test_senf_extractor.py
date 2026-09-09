import pytest

from core.senf.extractor import SENFExtractor, extract_senf
from core.senf.types import EntityRef, FrameRef, KindRef, SENF, ValueRef


def _symbol(senf: SENF, ref: EntityRef) -> str:
    entity = senf.entity(ref.entity_id)
    assert entity is not None
    return entity.canonical_symbol


def test_entities_mentions_and_typed_fillers_are_separate():
    senf = extract_senf(
        "s1", "Kebede ate 3 fish.", ["(: ate (Eats kebede 3 fish) (STV 1 1))"],
    )
    frame = senf.frames[0]
    assert [_symbol(senf, frame.roles[i].filler) for i in (0, 2)] == ["kebede", "fish"]
    assert frame.roles[1].filler == ValueRef("3", "number")
    assert [role.position for role in frame.roles] == [0, 1, 2]
    assert all(mention.entity_id for mention in senf.mentions)
    assert {mention.canonical_symbol for mention in senf.mentions} == senf.symbols()


def test_generic_roles_are_positional_and_safe_overrides_remain():
    generic = extract_senf("s1", "a helps b", ["(: x (Helps a b) (STV 1 1))"])
    isa = extract_senf("s2", "a is a thing", ["(: y (IsA a thing) (STV 1 1))"])
    assert [role.name for role in generic.frames[0].roles] == ["Arg0", "Arg1"]
    assert [role.name for role in isa.frames[0].roles] == ["Instance", "Class"]


def test_atom_symbols_are_authoritative_and_never_canonicalized_again():
    senf = extract_senf(
        "s1", "Fish eaters use tools.",
        ["(: x (Uses Fish-Eaters digitalTools) (STV 1 1))"],
    )
    assert senf.symbols() == {"Fish-Eaters", "digitalTools"}
    assert {mention.canonical_symbol for mention in senf.mentions} == senf.symbols()


def test_isa_class_is_kind_ref_not_entity_and_assertions_are_multi_valued():
    senf = extract_senf(
        "s1",
        "Kebede is a researcher and clinician.",
        [
            "(: a (IsA kebede researcher) (STV 1 1))",
            "(: b (IsA kebede clinician) (STV 1 1))",
        ],
    )
    assert senf.symbols() == {"kebede"}
    assert isinstance(senf.frames[0].roles[1].filler, KindRef)
    assert [
        (assertion.kind.canonical_symbol, assertion.polarity, assertion.source_frame_id)
        for assertion in senf.kind_assertions
    ] == [("researcher", True, "s1:f0"), ("clinician", True, "s1:f1")]


def test_case_a_entity_kind_separation():
    senf = extract_senf(
        "case-a",
        "The camera is a device.",
        ["(: a (IsA camera device) (STV 1 1))"],
    )

    assert senf.symbols() == {"camera"}
    assert senf.frames[0].roles[0].filler == EntityRef("case-a:e0", "case-a:m0")
    assert senf.frames[0].roles[1].filler == KindRef("device")


def test_negated_isa_records_negative_kind_assertion():
    senf = extract_senf(
        "s1", "Kebede is not a doctor.",
        ["(: a (Not (IsA kebede doctor)) (STV 1 1))"],
    )
    assert senf.frames[0].polarity is False
    assert senf.kind_assertions[0].polarity is False
    assert senf.kind_assertions[0].kind == KindRef("doctor")


def test_nested_predication_preserves_parent_arity_order_and_reference():
    senf = extract_senf(
        "s1", "Alice believes Bob runs.",
        ["(: a (Believes alice (Runs bob)) (STV 1 1))"],
    )
    assert [frame.predicate_head for frame in senf.frames] == ["Runs", "Believes"]
    parent = senf.frames[1]
    assert [role.position for role in parent.roles] == [0, 1]
    assert parent.roles[1].filler == FrameRef(senf.frames[0].frame_id)


def test_rule_context_and_source_atom_are_retained():
    senf = extract_senf(
        "s1",
        "Edges make paths.",
        [
            "(: rule_7 (Implication (Premises (Edge $x $y)) "
            "(Conclusions (Path $x $y))) (STV 1 1))"
        ],
    )
    assert [
        (frame.predicate_head, frame.clause_role, frame.source_atom_id)
        for frame in senf.frames
    ] == [("Edge", "premise", "rule_7"), ("Path", "conclusion", "rule_7")]
    assert all(
        isinstance(role.filler, ValueRef)
        for frame in senf.frames
        for role in frame.roles
    )


def test_repeated_source_occurrences_get_distinct_role_refs():
    senf = extract_senf(
        "s1",
        "one camera beside another camera",
        ["(: a (Beside camera camera) (STV 1 1))"],
    )
    refs = [role.filler for role in senf.frames[0].roles]
    assert refs == [EntityRef("s1:e0", "s1:m0"), EntityRef("s1:e1", "s1:m1")]
    assert refs[0].entity_id != refs[1].entity_id
    assert refs[0].mention_id != refs[1].mention_id


def test_one_source_occurrence_is_reused_stably():
    senf = extract_senf(
        "s1", "camera beside itself", ["(: a (Beside camera camera) (STV 1 1))"]
    )
    refs = [role.filler for role in senf.frames[0].roles]
    assert refs == [EntityRef("s1:e0", "s1:m0"), EntityRef("s1:e0", "s1:m0")]


def test_configured_mention_cap_skips_the_whole_atom():
    senf = SENFExtractor(max_mentions_per_sentence=1).extract(
        "s1", "Camera saw camera.", ["(: a (Sees camera lens) (STV 1 1))"],
    )
    assert senf.frames == []
    assert senf.entities == []
    assert senf.mentions == []


def test_capped_atom_rollback_preserves_prior_frames_without_dangling_refs():
    senf = SENFExtractor(max_mentions_per_sentence=1).extract(
        "s1",
        "camera sees one lens beside another lens",
        [
            "(: a (Sees camera) (STV 1 1))",
            "(: b (Beside lens lens) (STV 1 1))",
        ],
    )
    assert [frame.source_atom_id for frame in senf.frames] == ["a"]
    assert len(senf.entities) == len(senf.mentions) == 1
    ref = senf.frames[0].roles[0].filler
    assert ref == EntityRef(senf.entities[0].entity_id, senf.mentions[0].mention_id)


def test_default_mention_limit_is_unbounded():
    atoms = [f"(: a{i} (Sees entity_{i}) (STV 1 1))" for i in range(40)]
    senf = extract_senf("s1", "", atoms)
    assert len(senf.frames) == len(senf.mentions) == 40


def test_no_dangling_role_references():
    senf = extract_senf(
        "s1", "a relates b", ["(: a (Relates a (Knows b)) (STV 1 1))"],
    )
    entity_ids = {entity.entity_id for entity in senf.entities}
    frame_ids = {frame.frame_id for frame in senf.frames}
    for frame in senf.frames:
        for role in frame.roles:
            if isinstance(role.filler, EntityRef):
                assert role.filler.entity_id in entity_ids
            if isinstance(role.filler, FrameRef):
                assert role.filler.frame_id in frame_ids


@pytest.mark.parametrize(
    "statement",
    [None, "", "not an atom", "(: bad (Unclosed a", "(: x () (STV 1 1))"],
)
def test_malformed_atoms_never_raise(statement):
    assert isinstance(extract_senf("s1", "text", [statement]), SENF)


def test_empty_input_yields_empty_senf():
    senf = extract_senf("s1", "", [])
    assert senf.is_empty
    assert senf.senf_id == "senf:s1"
