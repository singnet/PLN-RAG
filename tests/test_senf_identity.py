from dataclasses import replace

import pytest

from core.senf.extractor import extract_senf
from core.senf.exemplars import score_exemplars
from core.senf.identity import (
    DEFAULT_IDENTITY_THRESHOLD,
    IdentityResolver,
    IdentityWeights,
    resolve_identity,
)
from core.senf.types import Context, Entity, EntityRef, KindAssertion, KindRef, Role, SENF, SENFFrame, Mention
from core.symbol_normalization import canonical_symbol


def edge_for(graph, left: str, right: str):
    wanted = tuple(sorted((left, right)))
    for edge in graph.edges:
        if edge.symbols == wanted:
            return edge
    return None


def merged_symbols(graph) -> list[tuple[str, str]]:
    return [edge.symbols for edge in graph.merged]


# --- the worked example ------------------------------------------------------


def _camera_and_pronoun() -> list[SENF]:
    return [
        extract_senf(
            "s1",
            "The camera has a wide lens.",
            ["(: a (HasProperty camera wide_lens) (STV 1.0 1.0))"],
        ),
        extract_senf(
            "s2", "It is expensive.", ["(: b (HasProperty it expensive) (STV 1.0 1.0))"]
        ),
    ]


def test_pronoun_resolves_to_its_only_prominent_antecedent():
    graph = resolve_identity(_camera_and_pronoun())

    assert merged_symbols(graph) == [("camera", "it")]
    assert graph.resolve("it") == "camera"


def test_case_b_camera_and_pronoun_remain_separate_mentions_before_identity():
    camera, pronoun = _camera_and_pronoun()

    assert camera.entities[0].entity_id == "s1:e0"
    assert pronoun.entities[0].entity_id == "s2:e0"
    assert camera.mentions[0].canonical_symbol == "camera"
    assert pronoun.mentions[0].canonical_symbol == "it"
    assert pronoun.mentions[0].surface == "It"
    assert pronoun.mentions[0].char_span == (0, 2)
    assert resolve_identity([camera, pronoun]).same_entity("s1:e0", "s2:e0")


def test_guide_case_a_lending_roles_support_pronoun_identity():
    lent = extract_senf(
        "s1", "Alex lent Sam the Nikon camera.",
        [
            "(: lend (Lent alex sam nikon_camera) (STV 1 1))",
            "(: kind (IsA nikon_camera Camera) (STV 1 1))",
        ],
    )
    returned = extract_senf(
        "s2", "Sam returned it.",
        [
            "(: return (Returned sam it) (STV 1 1))",
            "(: kind (IsA it Camera) (STV 1 1))",
        ],
    )
    graph = resolve_identity([lent, returned])
    edge = edge_for(graph, "it", "nikon_camera")

    assert edge is not None
    assert "semantic_role_continuity" in edge.evidence
    assert edge.negative_strength == 0.0
    assert graph.same_entity(
        next(m.entity_id for m in lent.mentions if m.canonical_symbol == "nikon_camera"),
        next(m.entity_id for m in returned.mentions if m.canonical_symbol == "it"),
    )


def test_guide_case_b_definite_description_is_evidence_not_hardcoded_equality():
    first = extract_senf(
        "s1", "Alex lent Sam a camera.",
        ["(: a (Lent alex sam camera) (STV 1 1))"],
    )
    second = extract_senf(
        "s2", "Later, the camera was found at Alex's house.",
        ["(: b (AtLocation camera alex_house) (STV 1 1))"],
    )
    graph = resolve_identity([first, second])
    edge = next(
        edge for edge in graph.edges
        if edge.left.canonical_symbol == edge.right.canonical_symbol == "camera"
    )

    assert {"exact_symbol", "definiteness", "recency"} <= set(edge.evidence)
    assert "surface_match" not in edge.evidence
    assert edge.guard.source_unit_ids == ("s1:u0", "s2:u0")
    assert edge in graph.merged


@pytest.mark.parametrize(("later_unit", "expected_recency"), [("doc:u1", True), ("doc:u2", False)])
def test_recency_uses_ordered_source_units(later_unit, expected_recency):
    first = extract_senf(
        "s1", "A camera appeared.",
        ["(: a (Appeared camera) (STV 1 1))"],
    )
    later = extract_senf(
        "s2", "The camera broke.",
        ["(: b (Broke camera) (STV 1 1))"],
    )
    first.mentions[0] = replace(
        first.mentions[0], source_unit_id="doc:u0", definiteness="indefinite"
    )
    later.mentions[0] = replace(
        later.mentions[0], source_unit_id=later_unit, definiteness="definite"
    )
    first.mentions[0] = replace(first.mentions[0], sentence_id="doc")
    later.mentions[0] = replace(later.mentions[0], sentence_id="doc")

    graph = resolve_identity([first, later])
    edge = next(edge for edge in graph.edges if edge.symbols == ("camera", "camera"))

    assert ("recency" in edge.evidence) is expected_recency
    assert (edge in graph.merged) is expected_recency


def test_the_merge_rests_on_three_independent_signals():
    """Pins *why* it merged. A weight tweak must not reduce this to one signal."""
    edge = edge_for(resolve_identity(_camera_and_pronoun()), "camera", "it")

    assert edge is not None
    assert set(edge.evidence) == {
        "pronoun_antecedent", "role_prominence", "semantic_role_continuity", "unambiguous",
    }
    assert edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    assert 0.0 < edge.confidence <= 1.0


def test_distractors_in_the_same_sentence_stay_far_below_threshold():
    graph = resolve_identity(_camera_and_pronoun())

    for other in ("expensive", "wide_len"):
        edge = edge_for(graph, "it", other)
        if edge is not None:
            assert edge.strength < DEFAULT_IDENTITY_THRESHOLD


def test_representative_election_prefers_the_entity_over_the_pronoun():
    graph = resolve_identity(_camera_and_pronoun())

    assert graph.resolve("camera") == "camera"
    assert graph.resolve_entity("s2:e0") == "s1:e0"
    assert graph.same_entity("s1:m0", "s2:m0")
    assert {"s1:e0", "s2:e0"} in graph.clusters()
    assert graph.merge_count == 1


def test_a_qualifier_named_organisation_does_not_merge_with_its_first_token():
    """Observed on stress25/A15: `human` merged into the consortium's name.

    Both are proper and share a leading token, so `proper_compat` + `name_extension`
    cleared the threshold — rewriting "influences human health" into "influences
    consortium health". Properness cannot discriminate here; token distance can.
    """
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "The Human Microbiome Action Consortium met.",
                ["(: a (Met human_microbiome_action_consortium) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s2",
                "A survey was initiated by Human.",
                ["(: b (Initiated human delphi_survey) (STV 1.0 1.0))"],
            ),
        ]
    )

    assert merged_symbols(graph) == []
    assert graph.resolve("human") == "human"


def test_specificity_outranks_mention_type():
    """Observed on stress25/A03: `hsc` parsed as proper, `mouse_hsc` as nominal.

    Type priority alone therefore preferred the bare symbol and dropped the
    `mouse` qualifier, which distinguishes the entity.
    """
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "Mouse HSC were edited.",
                ["(: a (IsA mouse_hsc hematopoietic_stem_cell) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s2",
                "HSC were transplanted.",
                ["(: b (IsA hsc hematopoietic_stem_cell) (STV 1.0 1.0))"],
            ),
        ]
    )

    assert graph.resolve("hsc") == "mouse_hsc"


def test_a_pronoun_never_wins_however_compound_it_looks():
    """Specificity is ranked first, so the pronoun guard must not be shape-based."""
    graph = resolve_identity(_camera_and_pronoun())

    assert graph.resolve("it") == "camera"


# --- the non-merges ----------------------------------------------------------


def test_two_plausible_antecedents_produce_two_subthreshold_edges():
    """The core anaphora safety property: ambiguity must not resolve at all."""
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "Kebede works and Almaz works.",
                [
                    "(: a (Works kebede) (STV 1.0 1.0))",
                    "(: b (Works almaz) (STV 1.0 1.0))",
                ],
            ),
            extract_senf("s2", "He was late.", ["(: c (Late he) (STV 1.0 1.0))"]),
        ]
    )

    assert merged_symbols(graph) == []
    for antecedent in ("kebede", "almaz"):
        edge = edge_for(graph, "he", antecedent)
        assert edge is not None, "the evidence should be recorded, just not acted on"
        assert edge.strength < DEFAULT_IDENTITY_THRESHOLD
        assert "unambiguous" not in edge.evidence
    assert graph.resolve("he") == "he"


def test_two_distinct_proper_names_of_the_same_kind_never_merge():
    graph = resolve_identity(
        [
            extract_senf(
                "s1", "Kebede is a researcher.", ["(: a (IsA kebede researcher) (STV 1.0 1.0))"]
            ),
            extract_senf(
                "s2", "Almaz is a researcher.", ["(: b (IsA almaz researcher) (STV 1.0 1.0))"]
            ),
        ]
    )

    assert merged_symbols(graph) == []
    assert graph.resolve("kebede") == "kebede"
    assert graph.resolve("almaz") == "almaz"


def test_compounds_contrasted_by_their_modifier_are_vetoed():
    """fish_eater and meat_eater share a head precisely because they contrast."""
    graph = resolve_identity(
        [
            extract_senf("s1", "Fish eaters are smart.", ["(: a (Smart fish_eater) (STV 1.0 1.0))"]),
            extract_senf("s2", "Meat eaters are strong.", ["(: b (Strong meat_eater) (STV 1.0 1.0))"]),
        ]
    )

    edge = edge_for(graph, "fish_eater", "meat_eater")
    assert edge is not None
    assert edge.negative_strength >= 0.8
    assert "modifier_conflict" in edge.negative_evidence
    assert merged_symbols(graph) == []


def test_conflicting_kinds_veto_even_with_other_evidence():
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "Kebede is a researcher.",
                ["(: a (IsA kebede researcher) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s2",
                "Kebede Clinic is a hospital.",
                ["(: b (IsA kebede_clinic hospital) (STV 1.0 1.0))"],
            ),
        ]
    )

    edge = edge_for(graph, "kebede", "kebede_clinic")
    assert edge is not None
    assert edge.negative_strength >= 0.9
    assert "kind_conflict" in edge.negative_evidence
    assert edge not in graph.merged


def test_two_pronouns_never_merge_with_each_other():
    graph = resolve_identity(
        [
            extract_senf("s1", "It arrived.", ["(: a (Arrived it) (STV 1.0 1.0))"]),
            extract_senf("s2", "They left.", ["(: b (Left they) (STV 1.0 1.0))"]),
        ]
    )

    edge = edge_for(graph, "it", "they")
    assert edge is not None
    assert edge.negative_strength >= 0.9
    assert "both_pronouns" in edge.negative_evidence
    assert edge not in graph.merged


def test_same_named_entities_keep_positive_and_negative_evidence_without_merging():
    left = Mention("camera", "camera", "s1", "s1:e0", "s1:m0", (2, 8))
    right = Mention("camera", "camera", "s1", "s1:e1", "s1:m1", (28, 34))
    senf = SENF(
        "senf:s1",
        "s1",
        entities=[Entity("s1:e0", "camera"), Entity("s1:e1", "camera")],
        mentions=[left, right],
        frames=[SENFFrame(
            "s1:f0",
            "Beside",
            [
                Role("Arg0", EntityRef("s1:e0", "s1:m0"), 0),
                Role("Arg1", EntityRef("s1:e1", "s1:m1"), 1),
            ],
            source_text="A camera was beside another camera.",
        )],
    )

    graph = resolve_identity([senf])
    edge = next(
        edge
        for edge in graph.edges
        if edge.left.canonical_symbol == edge.right.canonical_symbol == "camera"
    )
    assert "exact_symbol" in edge.evidence
    assert "same_frame_distinct_roles" in edge.negative_evidence
    assert "contrastive_language" in edge.negative_evidence
    assert edge not in graph.merged
    assert not graph.same_entity(left.mention_id, right.mention_id)
    assert graph.resolve_entity(left.entity_id) == left.entity_id
    assert graph.resolve_entity(right.entity_id) == right.entity_id


def test_two_indefinite_cameras_do_not_merge_from_symbol_role_and_recency():
    first = extract_senf(
        "s1", "A camera has a red light.",
        ["(: a (HasProperty camera red_light) (STV 1 1))"],
    )
    second = extract_senf(
        "s2", "A camera has a blue light.",
        ["(: b (HasProperty camera blue_light) (STV 1 1))"],
    )
    first.mentions[0] = replace(
        first.mentions[0], sentence_id="doc", source_unit_id="doc:u0", definiteness="indefinite"
    )
    second.mentions[0] = replace(
        second.mentions[0], sentence_id="doc", source_unit_id="doc:u1", definiteness="indefinite"
    )

    graph = resolve_identity([first, second])
    edge = next(edge for edge in graph.edges if edge.symbols == ("camera", "camera"))

    assert {"exact_symbol", "semantic_role_continuity", "recency"} <= set(edge.evidence)
    assert edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    assert edge not in graph.merged
    assert edge in graph.alternatives


def test_guide_case_d_preserves_exemplar_and_contrast_conflict_as_an_alternative():
    attached = score_exemplars(extract_senf(
        "s1", "Source A: The lens attached to the camera was cracked.",
        ["(: a (Attached lens camera) (STV 1 1))"],
    ))
    borrowed = score_exemplars(extract_senf(
        "s2", "Source B: Casey says Sam borrowed the lens separately.",
        ["(: b (Says casey (Borrowed sam lens)) (STV 1 1))"],
    ))
    graph = resolve_identity([attached, borrowed])
    edge = next(
        edge for edge in graph.edges
        if edge.left.canonical_symbol == edge.right.canonical_symbol == "lens"
    )

    assert "exact_symbol" in edge.evidence
    assert {"exemplar_conflict", "contrastive_language"} <= set(edge.negative_evidence)
    assert edge.strength > 0 and edge.negative_strength > 0
    assert edge in graph.alternatives
    assert edge not in graph.merged
    assert edge.guard.speakers == ("casey",)


def test_first_person_pronouns_take_no_antecedent():
    """Deictic reference points outside the text; there is nothing to bind."""
    graph = resolve_identity(
        [
            extract_senf("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1.0 1.0))"]),
            extract_senf("s2", "I am pleased.", ["(: b (Pleased i) (STV 1.0 1.0))"]),
        ]
    )

    assert edge_for(graph, "camera", "i") is None
    assert merged_symbols(graph) == []


def test_pronoun_two_sentences_back_is_out_of_window():
    graph = resolve_identity(
        [
            extract_senf("s1", "The camera arrived.", ["(: a (Arrived camera) (STV 1.0 1.0))"]),
            extract_senf("s2", "Prices rose.", ["(: b (Rose price) (STV 1.0 1.0))"]),
            extract_senf("s3", "It is expensive.", ["(: c (Expensive it) (STV 1.0 1.0))"]),
        ]
    )

    assert edge_for(graph, "camera", "it") is None


def test_cataphora_is_not_resolved():
    """A pronoun before its referent is out of scope; forward binding is unsafe."""
    graph = resolve_identity(
        [
            extract_senf("s1", "It is expensive.", ["(: a (Expensive it) (STV 1.0 1.0))"]),
            extract_senf("s2", "The camera arrived.", ["(: b (Arrived camera) (STV 1.0 1.0))"]),
        ]
    )

    assert merged_symbols(graph) == []


def test_cataphora_in_the_same_source_unit_is_not_resolved():
    senf = extract_senf(
        "s1",
        "Before it broke, the camera arrived.",
        [
            "(: a (Broke it) (STV 1 1))",
            "(: b (Arrived camera) (STV 1 1))",
        ],
    )

    graph = resolve_identity([senf])

    assert edge_for(graph, "camera", "it") is None
    assert merged_symbols(graph) == []


# --- named entities across sentences -----------------------------------------


def test_a_name_gaining_tokens_merges_towards_the_fuller_name():
    """`kebede` / `kebede_alemu` (one token gained) stays a merge, so the token
    bound does not amputate the genuine pattern."""
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "Dr Kebede Alemu leads the trial.",
                ["(: a (IsA kebede_alemu researcher) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s2",
                "Kebede published results.",
                ["(: b (IsA kebede researcher) (STV 1.0 1.0))"],
            ),
        ]
    )

    assert merged_symbols(graph) == [("kebede", "kebede_alemu")]
    assert graph.resolve("kebede") == "kebede_alemu"


def test_a_modifier_prefix_that_is_not_a_name_does_not_merge():
    """fish / fish_eater has the same shape as kebede / kebede_alemu and must not."""
    graph = resolve_identity(
        [
            extract_senf("s1", "Fish is healthy.", ["(: a (Healthy fish) (STV 1.0 1.0))"]),
            extract_senf("s2", "Fish eaters are smart.", ["(: b (Smart fish_eater) (STV 1.0 1.0))"]),
        ]
    )

    assert merged_symbols(graph) == []


def test_two_people_sharing_a_first_name_do_not_merge():
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "The trial lists Kebede Alemu.",
                ["(: a (IsA kebede_alemu researcher) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s2",
                "The trial lists Kebede Bekele.",
                ["(: b (IsA kebede_bekele researcher) (STV 1.0 1.0))"],
            ),
        ]
    )

    assert merged_symbols(graph) == []


def test_short_name_cannot_bridge_conflicting_proper_name_components():
    graph = resolve_identity(
        [
            extract_senf(
                "s1", "Dr Kebede Alemu leads the trial.",
                ["(: a (IsA kebede_alemu researcher) (STV 1 1))"],
            ),
            extract_senf(
                "s2", "Kebede published results.",
                ["(: b (IsA kebede researcher) (STV 1 1))"],
            ),
            extract_senf(
                "s3", "Dr Kebede Bekele leads the trial.",
                ["(: c (IsA kebede_bekele researcher) (STV 1 1))"],
            ),
        ]
    )
    rejected = [
        edge for edge in graph.alternatives
        if "kebede" in edge.symbols and edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    ]

    assert not graph.same_entity("s1:e0", "s3:e0")
    assert len(graph.merged) == 1
    assert rejected, "the other plausible short-name edge remains inspectable"
    conflict = edge_for(graph, "kebede_alemu", "kebede_bekele")
    assert conflict is not None
    assert conflict.negative_strength >= 0.5


def _contextual_camera_pair() -> list[SENF]:
    first = extract_senf(
        "s1", "A camera appeared.", ["(: a (Appeared camera) (STV 1 1))"]
    )
    second = extract_senf(
        "s2", "The camera appeared.", ["(: b (Appeared camera) (STV 1 1))"]
    )
    return [first, second]


@pytest.mark.parametrize(
    ("field_name", "left_value", "right_value"),
    [
        ("speaker", "alice", "bob"),
        ("modality", "actual", "hypothetical"),
        ("time_ref", "morning", "evening"),
        ("location_ref", "lab", "field"),
    ],
)
def test_conflicting_identity_context_guards_reject_merge(
    field_name, left_value, right_value
):
    first, second = _contextual_camera_pair()
    first.frames[0].context = replace(
        first.frames[0].context, **{field_name: left_value}
    )
    second.frames[0].context = replace(
        second.frames[0].context, **{field_name: right_value}
    )

    graph = resolve_identity([first, second])
    edge = next(edge for edge in graph.edges if edge.symbols == ("camera", "camera"))

    assert edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    assert edge not in graph.merged
    assert edge in graph.alternatives


def test_missing_context_does_not_block_safe_definite_identity():
    first, second = _contextual_camera_pair()
    second.frames[0].context = Context(
        second.frames[0].context.source_unit_id, speaker="alice"
    )

    graph = resolve_identity([first, second])

    assert merged_symbols(graph) == [("camera", "camera")]


def test_matching_context_allows_safe_pronoun_identity():
    camera, pronoun = _camera_and_pronoun()
    camera.frames[0].context = replace(camera.frames[0].context, speaker="alice")
    pronoun.frames[0].context = replace(pronoun.frames[0].context, speaker="alice")

    graph = resolve_identity([camera, pronoun])

    assert merged_symbols(graph) == [("camera", "it")]


# --- the merge gate ----------------------------------------------------------


def _same_surface_pair(second_sentence: str) -> list[SENF]:
    """Two mentions sharing a surface but not a symbol — exactly one evidence type.

    Built directly rather than through the extractor because a single-evidence pair
    strong enough to clear the threshold cannot arise from the default weights,
    and the gate has to be testable independently of them.
    """
    left = Mention("Camera", "camera", "s1", "s1:e0", "s1:m0")
    right_suffix = "1" if second_sentence == "s1" else "0"
    right = Mention(
        "Camera", "camera_unit", second_sentence,
        f"{second_sentence}:e{right_suffix}", f"{second_sentence}:m{right_suffix}",
    )
    return [
        SENF("senf:s1", "s1", entities=[Entity(left.entity_id, left.canonical_symbol)], mentions=[left]),
        SENF(f"senf:{second_sentence}", second_sentence, entities=[Entity(right.entity_id, right.canonical_symbol)], mentions=[right]),
    ]


def test_one_evidence_type_cannot_merge_across_a_sentence_boundary():
    resolver = IdentityResolver(weights=IdentityWeights(surface_match=0.9))
    graph = resolver.resolve(_same_surface_pair("s2"))

    edge = edge_for(graph, "camera", "camera_unit")
    assert edge is not None
    assert edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    assert edge.evidence == ("surface_match",)
    assert merged_symbols(graph) == [], "cross-sentence merges require corroboration"


def test_the_same_single_evidence_type_is_enough_within_one_sentence():
    resolver = IdentityResolver(weights=IdentityWeights(surface_match=0.9))
    graph = resolver.resolve(_same_surface_pair("s1"))

    assert merged_symbols(graph) == [("camera", "camera_unit")]


def test_threshold_of_one_disables_merging_entirely():
    """The escape hatch: turn identity off without a redeploy."""
    graph = IdentityResolver(threshold=1.0).resolve(_camera_and_pronoun())

    assert merged_symbols(graph) == []
    assert graph.resolve("it") == "it"
    assert graph.edges, "edges are still scored and reported, just never acted on"


def test_confidence_grows_with_independent_evidence_but_never_exceeds_one():
    graph = resolve_identity(_camera_and_pronoun())
    strong = edge_for(graph, "camera", "it")
    weak = edge_for(graph, "it", "expensive")

    assert weak is None or strong.confidence > weak.confidence
    assert all(0.0 <= edge.confidence <= 1.0 for edge in graph.edges)
    assert all(0.0 <= edge.strength <= 1.0 for edge in graph.edges)


# --- symbol safety -----------------------------------------------------------


def test_a_representative_is_always_an_existing_cluster_member():
    """Inventing a symbol here would fragment the space and unprove everything."""
    graph = resolve_identity(_camera_and_pronoun() + _same_surface_pair("s1"))

    clusters = graph.clusters()
    assert clusters, "the fixtures should produce at least one cluster"

    for cluster in clusters:
        representatives = {graph.resolve_entity(entity_id) for entity_id in cluster}
        assert len(representatives) == 1, "a cluster must agree on one representative"
        representative = representatives.pop()
        assert representative in cluster, "the representative must be a member"
        assert graph.entity_symbols[representative] == canonical_symbol(
            graph.entity_symbols[representative]
        )


def test_resolve_is_total_for_symbols_it_has_never_seen():
    """Statement rewriting must be able to resolve every token without a guard."""
    graph = resolve_identity(_camera_and_pronoun())

    assert graph.resolve("never_mentioned") == "never_mentioned"
    assert graph.resolve("") == ""


# --- degenerate input --------------------------------------------------------


@pytest.mark.parametrize(
    "senfs",
    [
        [],
        [SENF(senf_id="senf:s1", sentence_id="s1")],
        [extract_senf("s1", "", [])],
        [extract_senf("s1", "Nothing parses here.", ["not an atom"])],
    ],
    ids=["no_senfs", "empty_senf", "empty_extraction", "unparseable_atom"],
)
def test_degenerate_input_yields_an_empty_graph(senfs):
    graph = resolve_identity(senfs)

    assert graph.edges == ()
    assert graph.merged == ()
    assert graph.representatives == {}


def test_a_single_mention_has_no_pairs_to_score():
    graph = resolve_identity(
        [extract_senf("s1", "Rain fell.", ["(: a (Fell rain) (STV 1.0 1.0))"])]
    )

    assert graph.edges == ()
    assert graph.resolve("rain") == "rain"


def test_mentions_are_capped_per_sentence_because_scoring_is_quadratic():
    statements = [f"(: a{i} (Mentions e{i}) (STV 1.0 1.0))" for i in range(20)]
    graph = IdentityResolver(max_mentions_per_sentence=4).resolve(
        [extract_senf("s1", "text", statements)]
    )

    assert len(graph.nodes) == 4


def test_mention_cap_is_cumulative_for_duplicate_sentence_records():
    statements = [f"(: a{i} (Mentions e{i}) (STV 1.0 1.0))" for i in range(4)]
    first = extract_senf("s1", "text", statements)
    duplicate = extract_senf("s1", "text", statements)

    graph = IdentityResolver(max_mentions_per_sentence=4).resolve([first, duplicate])

    assert len(graph.nodes) == 4


def test_resolution_is_deterministic_across_repeated_calls():
    senfs = _camera_and_pronoun()
    first = resolve_identity(senfs)

    for _ in range(3):
        again = resolve_identity(senfs)
        assert [e.symbols for e in again.edges] == [e.symbols for e in first.edges]
        assert [e.strength for e in again.edges] == [e.strength for e in first.edges]
        assert again.representatives == first.representatives


def test_repeated_mentions_of_one_antecedent_are_not_rival_candidates():
    """A symbol recurring across retrieved SENFs must not look like ambiguity.

    Prior SENFs come from the vector store, so the same entity routinely appears in
    several of them. Counting each occurrence as a separate candidate would fabricate
    a tie and silently suppress every pronoun resolution in production.
    """
    graph = resolve_identity(
        [
            extract_senf(
                "s1",
                "The camera has a wide lens.",
                ["(: a (HasProperty camera wide_lens) (STV 1.0 1.0))"],
            ),
            extract_senf(
                "s1", "The camera shipped.", ["(: b (Shipped camera) (STV 1.0 1.0))"]
            ),
            extract_senf(
                "s2", "It is expensive.", ["(: c (HasProperty it expensive) (STV 1.0 1.0))"]
            ),
        ]
    )

    assert merged_symbols(graph) == [("camera", "it")]


# --- the optional embedding hook ---------------------------------------------


def _near_miss_pair() -> list[SENF]:
    """camera / device: same kind, same role slot — deliberately just short."""
    return [
        extract_senf("s1", "The camera is a gadget.", ["(: a (IsA camera gadget) (STV 1.0 1.0))"]),
        extract_senf("s2", "The device is a gadget.", ["(: b (IsA device gadget) (STV 1.0 1.0))"]),
    ]


def test_no_embedder_means_no_round_trips_and_no_embed_evidence():
    graph = resolve_identity(_near_miss_pair())

    edge = edge_for(graph, "camera", "device")
    assert edge is not None
    assert "embed_sim" not in edge.evidence
    assert edge.strength < DEFAULT_IDENTITY_THRESHOLD


def test_embedder_is_consulted_only_for_near_misses():
    """Each call is an Ollama round trip on the ingest path, so the band matters."""
    calls: list[str] = []

    def embedder(text: str):
        calls.append(text)
        return [1.0, 0.0]

    resolver = IdentityResolver(embedder=embedder)
    graph = resolver.resolve(_near_miss_pair())

    edge = edge_for(graph, "camera", "device")
    assert "embed_sim" in edge.evidence
    assert edge.strength >= DEFAULT_IDENTITY_THRESHOLD
    assert set(calls) == {
        "The camera is a gadget.",
        "The device is a gadget.",
    }, "embedding input must retain each mention's context"


def test_kind_evidence_uses_all_positive_kind_assertions():
    left, right = _near_miss_pair()
    left.kind_assertions.append(KindAssertion("s1:e0", KindRef("device"), True, "s1:f0"))
    right.kind_assertions.append(KindAssertion("s2:e0", KindRef("instrument"), True, "s2:f0"))

    edge = edge_for(resolve_identity([left, right]), "camera", "device")

    assert edge is not None
    assert "kind_match" in edge.evidence
    assert "kind_conflict" not in edge.negative_evidence


def test_a_pair_already_over_the_threshold_is_not_embedded():
    calls: list[str] = []

    def embedder(text: str):
        calls.append(text)
        return [1.0, 0.0]

    IdentityResolver(embedder=embedder).resolve(_camera_and_pronoun())

    assert calls == [], "spending a round trip on a decided pair is pure latency"


def test_a_failing_embedder_degrades_to_deterministic_evidence():
    def embedder(text: str):
        raise RuntimeError("ollama down")

    graph = IdentityResolver(embedder=embedder).resolve(_near_miss_pair())

    edge = edge_for(graph, "camera", "device")
    assert edge is not None
    assert "embed_sim" not in edge.evidence
