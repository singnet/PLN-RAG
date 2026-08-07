"""Unit tests for the deterministic SENF extractor.

Table-driven where the cases are genuinely parallel. Atoms are taken from shapes
the parsers actually emit — see the stress25 baseline artifact, where IsA (58
occurrences) and InGroup (18) dominate and arities run from 1 to 5.
"""

import pytest

from core.senf.extractor import SENFExtractor, extract_senf
from core.senf.types import Literal, Mention, SENF, SENFFrame
from core.symbol_normalization import canonical_symbol


def roles_of(frame: SENFFrame) -> list[tuple[str, str]]:
    out = []
    for role in frame.roles:
        value = (
            role.filler.canonical_symbol
            if isinstance(role.filler, Mention)
            else role.filler.value
        )
        out.append((role.name, value))
    return out


# (label, text, statements, expected [(head, polarity, [(role, filler)])])
FRAME_CASES = [
    (
        "binary_default_roles",
        "Digital tools help self-management.",
        ["(: f (Helps digital_tool self_management) (STV 1.0 1.0))"],
        [("Helps", True, [("Agent", "digital_tool"), ("Patient", "self_management")])],
    ),
    (
        "unary_is_theme_not_agent",
        "Self-management is substantial.",
        ["(: f (Substantial self_management) (STV 1.0 1.0))"],
        [("Substantial", True, [("Theme", "self_management")])],
    ),
    (
        "isa_override",
        "Kebede is a researcher.",
        ["(: f (IsA kebede researcher) (STV 1.0 1.0))"],
        [("IsA", True, [("Instance", "kebede"), ("Class", "researcher")])],
    ),
    (
        "at_location_override",
        "The clinic is located in Nairobi.",
        ["(: f (AtLocation clinic nairobi) (STV 1.0 1.0))"],
        [("AtLocation", True, [("Theme", "clinic"), ("Location", "nairobi")])],
    ),
    (
        "has_property_override",
        "The vial is sterile.",
        ["(: f (HasProperty vial sterile) (STV 1.0 1.0))"],
        [("HasProperty", True, [("Theme", "vial"), ("Property", "sterile")])],
    ),
    (
        "in_group_override",
        "Patient 4 is in the control group.",
        ["(: f (InGroup patient_4 control_group) (STV 1.0 1.0))"],
        [("InGroup", True, [("Member", "patient_4"), ("Group", "control_group")])],
    ),
    (
        "negation_flips_polarity_and_keeps_head",
        "The intervention group did not reduce HbA1c.",
        ["(: f (Not (ReducesHbA1c intervention_group)) (STV 1.0 1.0))"],
        [("ReducesHbA1c", False, [("Theme", "intervention_group")])],
    ),
    (
        "high_arity_numbers_roles",
        "We analyzed 342 participants in the trial.",
        ["(: f (Analyzed participant_data 342 diabete rct) (STV 1.0 1.0))"],
        [
            (
                "Analyzed",
                True,
                [
                    ("Agent", "participant_data"),
                    ("Patient", "342"),
                    ("Instrument", "diabete"),
                    ("Arg3", "rct"),
                ],
            )
        ],
    ),
]


@pytest.mark.parametrize(
    "label,text,statements,expected", FRAME_CASES, ids=[c[0] for c in FRAME_CASES]
)
def test_frame_extraction(label, text, statements, expected):
    senf = extract_senf("s1", text, statements)
    assert [(f.predicate_head, f.polarity, roles_of(f)) for f in senf.frames] == expected


MALFORMED = [
    "",
    "   ",
    "not an atom at all",
    "(: broken (Unclosed a b (STV 1.0 1.0)",
    "(: empty () (STV 1.0 1.0))",
    "(: nohead ($x $y) (STV 1.0 1.0))",
    "(Helps a b)",  # missing the (: name ...) wrapper
    "(: weird (123Bad a) (STV 1.0 1.0))",
    None,
]


@pytest.mark.parametrize("statement", MALFORMED, ids=[repr(s)[:28] for s in MALFORMED])
def test_malformed_atoms_never_raise(statement):
    senf = extract_senf("s1", "some text", [statement])
    assert isinstance(senf, SENF)


def test_malformed_atom_does_not_lose_its_neighbours():
    """One unparseable atom must not discard the rest of the sentence."""
    senf = extract_senf(
        "s1",
        "Kebede eats fish.",
        [
            "(: broken (Unclosed a b",
            "(: good (Eats kebede fish) (STV 1.0 1.0))",
        ],
    )
    assert [f.predicate_head for f in senf.frames] == ["Eats"]


def test_multi_frame_sentence_keeps_order_and_unique_frame_ids():
    senf = extract_senf(
        "s2",
        "Kebede is a researcher who eats fish and lives in Nairobi.",
        [
            "(: a (IsA kebede researcher) (STV 1.0 1.0))",
            "(: b (Eats kebede fish) (STV 1.0 1.0))",
            "(: c (AtLocation kebede nairobi) (STV 1.0 1.0))",
        ],
    )
    assert [f.predicate_head for f in senf.frames] == ["IsA", "Eats", "AtLocation"]
    assert [f.frame_id for f in senf.frames] == ["s2:f0", "s2:f1", "s2:f2"]
    assert len({f.frame_id for f in senf.frames}) == 3


def test_mentions_are_deduplicated_across_frames():
    senf = extract_senf(
        "s1",
        "Kebede eats fish. Kebede is smart.",
        [
            "(: a (Eats kebede fish) (STV 1.0 1.0))",
            "(: b (Smart kebede) (STV 1.0 1.0))",
        ],
    )
    assert sorted(senf.symbols()) == ["fish", "kebede"]


def test_pronouns_are_typed_as_pronoun():
    senf = extract_senf(
        "s1", "They eat fish and it helps them.", ["(: a (Eats they fish) (STV 1.0 1.0))"]
    )
    kinds = {m.canonical_symbol: m.mention_type for m in senf.mentions}
    assert kinds["they"] == "pronoun"
    assert kinds["fish"] == "common"


def test_proper_noun_detected_from_non_initial_capital():
    senf = extract_senf(
        "s1",
        "The clinic is in Nairobi.",
        ["(: a (AtLocation clinic nairobi) (STV 1.0 1.0))"],
    )
    types = {m.canonical_symbol: m.mention_type for m in senf.mentions}
    assert types["nairobi"] == "proper"
    assert types["clinic"] == "common"


def test_sentence_initial_capital_is_not_treated_as_proper():
    """Every first word is capitalized, so it carries no proper-noun evidence."""
    senf = extract_senf(
        "s1", "Aspirin was approved.", ["(: a (ApprovedFor aspirin trial) (STV 1.0 1.0))"]
    )
    types = {m.canonical_symbol: m.mention_type for m in senf.mentions}
    assert types["aspirin"] == "common"


def test_rule_variables_become_literals_not_mentions():
    """Rule variables keep their role slot but are not entities."""
    senf = extract_senf(
        "s1",
        "Fish eaters are smart.",
        [
            "(: r (Implication (Premises (Eats $person fish)) "
            "(Conclusions (Smart $person))) (CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
        ],
    )
    assert [f.predicate_head for f in senf.frames] == ["Eats", "Smart"]
    eats = senf.frames[0]
    assert eats.role("Agent").filler == Literal("$person", "variable")
    assert senf.symbols() == {"fish"}


def test_implication_premises_and_conclusions_both_yield_frames():
    senf = extract_senf(
        "s1",
        "Treatment requiring substantial self-management is helped by digital tools.",
        [
            "(: r (Implication (Premises (Require treatment diabete) "
            "(Substantial self_management)) (Conclusions "
            "(Helps digital_tool self_management))) (STV 1.0 1.0))"
        ],
    )
    assert [f.predicate_head for f in senf.frames] == [
        "Require",
        "Substantial",
        "Helps",
    ]


def test_kinds_populated_from_isa_only():
    senf = extract_senf(
        "s1",
        "Kebede is a researcher who eats fish.",
        [
            "(: a (IsA kebede researcher) (STV 1.0 1.0))",
            "(: b (Eats kebede fish) (STV 1.0 1.0))",
        ],
    )
    assert senf.kinds == {"kebede": "researcher"}


def test_kinds_keeps_first_assignment_for_stability():
    senf = extract_senf(
        "s1",
        "Kebede is a researcher. Kebede is a clinician.",
        [
            "(: a (IsA kebede researcher) (STV 1.0 1.0))",
            "(: b (IsA kebede clinician) (STV 1.0 1.0))",
        ],
    )
    assert senf.kinds == {"kebede": "researcher"}


def test_symbols_match_canonical_symbol_exactly():
    """Symbol-space divergence is the top integration risk; assert it directly."""
    senf = extract_senf(
        "s1",
        "Fish-Eaters use Digital Tools.",
        ["(: a (Uses Fish-Eaters digitalTools) (STV 1.0 1.0))"],
    )
    for mention in senf.mentions:
        assert mention.canonical_symbol == canonical_symbol(mention.canonical_symbol)
    assert sorted(senf.symbols()) == ["digital_tool", "fish_eater"]


def test_head_lemma_is_the_compound_head():
    senf = extract_senf(
        "s1", "Digital tools help.", ["(: a (Helps digital_tool user) (STV 1.0 1.0))"]
    )
    lemmas = {m.canonical_symbol: m.head_lemma for m in senf.mentions}
    assert lemmas["digital_tool"] == "tool"


def test_extraction_is_deterministic_across_repeated_calls():
    text = "Kebede is a researcher who eats fish in Nairobi."
    statements = [
        "(: a (IsA kebede researcher) (STV 1.0 1.0))",
        "(: b (Eats kebede fish) (STV 1.0 1.0))",
        "(: c (AtLocation kebede nairobi) (STV 1.0 1.0))",
    ]
    first = extract_senf("s1", text, statements)
    for _ in range(3):
        again = extract_senf("s1", text, statements)
        assert [f.frame_id for f in again.frames] == [f.frame_id for f in first.frames]
        assert [roles_of(f) for f in again.frames] == [roles_of(f) for f in first.frames]
        assert again.mentions == first.mentions
        assert again.kinds == first.kinds


def test_empty_input_yields_empty_senf():
    senf = extract_senf("s1", "", [])
    assert senf.is_empty
    assert senf.senf_id == "senf:s1"
    assert senf.sentence_id == "s1"


def test_max_mentions_is_respected():
    statements = [
        f"(: a{i} (Mentions e{i}) (STV 1.0 1.0))" for i in range(10)
    ]
    senf = SENFExtractor(max_mentions_per_sentence=4).extract("s1", "text", statements)
    assert len(senf.mentions) == 4
    assert len(senf.frames) == 10, "capping mentions must not drop frames"


def test_truth_value_is_not_mistaken_for_a_frame():
    senf = extract_senf(
        "s1", "Kebede eats fish.", ["(: a (Eats kebede fish) (STV 1.0 1.0))"]
    )
    assert [f.predicate_head for f in senf.frames] == ["Eats"]
    assert "stv" not in senf.symbols()


def test_ctv_weighted_rule_is_not_mistaken_for_a_frame():
    senf = extract_senf(
        "s1",
        "Edges make paths.",
        [
            "(: r (Implication (Premises (Edge $x $y)) (Conclusions (Path $x $y))) "
            "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
        ],
    )
    assert [f.predicate_head for f in senf.frames] == ["Edge", "Path"]
    assert senf.symbols() == set()


def test_frame_helpers():
    senf = extract_senf(
        "s1", "Kebede eats fish.", ["(: a (Eats kebede fish) (STV 1.0 1.0))"]
    )
    frame = senf.frames[0]
    assert frame.filler_symbols() == ["kebede", "fish"]
    assert frame.role("Agent").filler.canonical_symbol == "kebede"
    assert frame.role("Nonexistent") is None
    assert frame.source_sentence_id == "s1"
    assert frame.source_text == "Kebede eats fish."
