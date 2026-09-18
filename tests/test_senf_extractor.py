import pytest

from core.senf.extractor import SENFExtractor, extract_senf
from core.senf.types import (
    EntityRef,
    FrameRef,
    KindRef,
    SENF,
    ValueRef,
    SourceSpan,
    senf_from_payload,
    senf_to_payload,
)


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


def test_source_and_frame_spans_are_typed_and_conservative():
    exact = extract_senf(
        "s1", "Kebede works.", ["(: a (Works kebede) (STV 1 1))"]
    )
    inflected = extract_senf(
        "s2", "Kebede ate fish.", ["(: a (Eats kebede fish) (STV 1 1))"]
    )

    assert exact.source_units[0].char_span == SourceSpan(0, 13)
    assert exact.frames[0].frame_span == SourceSpan(0, 12)
    assert exact.frames[0].clause_span == SourceSpan(0, 13)
    assert inflected.frames[0].frame_span is None
    assert inflected.frames[0].clause_span == SourceSpan(0, 16)


def test_generic_roles_are_positional_and_safe_overrides_remain():
    generic = extract_senf("s1", "a helps b", ["(: x (Helps a b) (STV 1 1))"])
    isa = extract_senf("s2", "a is a thing", ["(: y (IsA a thing) (STV 1 1))"])
    assert [role.name for role in generic.frames[0].roles] == ["Arg0", "Arg1"]
    assert [role.name for role in isa.frames[0].roles] == ["Instance", "Class"]


@pytest.mark.parametrize(
    ("predicate", "arguments", "roles"),
    [
        ("Lent", "alex sam camera", ["Agent", "Recipient", "Theme"]),
        ("Lends", "alex sam camera", ["Agent", "Recipient", "Theme"]),
        ("Loaned", "alex sam camera", ["Agent", "Recipient", "Theme"]),
        ("Returned", "sam camera", ["Agent", "Theme"]),
        ("Returns", "sam camera", ["Agent", "Theme"]),
        ("Borrowed", "sam lens monday", ["Agent", "Theme", "Time"]),
        ("Borrows", "sam lens", ["Agent", "Theme"]),
        ("Says", "casey claim", ["Speaker", "Content"]),
        ("Claims", "casey claim", ["Speaker", "Content"]),
    ],
)
def test_bounded_predicate_arity_role_registry(predicate, arguments, roles):
    senf = extract_senf("s1", arguments, [f"(: a ({predicate} {arguments}) (STV 1 1))"])
    assert [role.name for role in senf.frames[0].roles] == roles


def test_registered_predicate_with_unknown_arity_falls_back_to_arg_roles():
    senf = extract_senf("s1", "alex camera", ["(: a (Lent alex camera) (STV 1 1))"])
    assert [role.name for role in senf.frames[0].roles] == ["Arg0", "Arg1"]


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


def test_source_units_definiteness_and_explicit_claim_context_are_typed():
    text = "Source A: Alex lent Sam a lens. Source B: Casey claims Sam borrowed the lens on Monday."
    senf = extract_senf(
        "case-d",
        text,
        [
            "(: a (Lent alex sam lens) (STV 1 1))",
            "(: b (Claims casey (Borrowed sam lens monday)) (STV 1 1))",
        ],
    )
    assert [(unit.source_unit_id, unit.text) for unit in senf.source_units] == [
        ("case-d:u0", "Source A: Alex lent Sam a lens."),
        ("case-d:u1", "Source B: Casey claims Sam borrowed the lens on Monday."),
    ]
    lenses = [mention for mention in senf.mentions if mention.canonical_symbol == "lens"]
    assert [(mention.source_unit_id, mention.definiteness) for mention in lenses] == [
        ("case-d:u0", "indefinite"), ("case-d:u1", "definite"),
    ]
    borrowed = next(frame for frame in senf.frames if frame.predicate_head == "Borrowed")
    assert borrowed.context.speaker == "casey"
    assert borrowed.context.modality == borrowed.modality == "claim"
    assert borrowed.context.time_ref == borrowed.time_ref == "monday"
    assert any(
        constraint.kind == "time_ref" and constraint.frame_id == borrowed.frame_id
        for constraint in senf.constraints
    )


def test_context_is_not_inferred_from_words_but_explicit_modality_is_propagated():
    lexical = extract_senf(
        "s1", "Sam may borrow the lens.",
        ["(: a (Borrowed sam lens) (STV 1 1))"],
    )
    explicit = extract_senf(
        "s1", "Sam may borrow the lens.",
        ["(: a (May (Borrowed sam lens)) (STV 1 1))"],
    )
    assert next(frame for frame in lexical.frames if frame.predicate_head == "Borrowed").modality is None
    borrowed = next(frame for frame in explicit.frames if frame.predicate_head == "Borrowed")
    assert borrowed.modality == borrowed.context.modality == "possible"


def test_failed_atom_rolls_back_constraints_and_all_graph_nodes(monkeypatch):
    extractor = SENFExtractor()
    original = extractor._apply_explicit_context

    def fail_after_context(frame, senf):
        original(frame, senf)
        if frame.source_atom_id == "bad":
            raise RuntimeError("failed after constraints")

    monkeypatch.setattr(extractor, "_apply_explicit_context", fail_after_context)
    senf = extractor.extract(
        "s1",
        "Sam borrowed the lens on Monday. Alex waits.",
        [
            "(: bad (Borrowed sam lens monday) (STV 1 1))",
            "(: good (Waits alex) (STV 1 1))",
        ],
    )

    assert [frame.source_atom_id for frame in senf.frames] == ["good"]
    assert senf.constraints == []
    assert senf.symbols() == {"alex"}


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


def test_frame_can_reference_a_mention_from_an_earlier_source_unit():
    senf = extract_senf(
        "s1",
        "Alice arrived. Bob greeted Alice.",
        ["(: greeting (Greeted bob alice) (STV 1 1))"],
    )

    assert {mention.source_unit_id for mention in senf.mentions} == {"s1:u0", "s1:u1"}
    assert senf.frames[0].context.source_unit_id == "s1:u1"
    assert senf_from_payload(senf_to_payload(senf)) == senf


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
