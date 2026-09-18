import json
from pathlib import Path

import pytest

from core.senf import (
    SENF_PAYLOAD_KEY,
    SENF_PAYLOAD_VERSION,
    senf_from_payload,
    senf_to_payload,
)
from core.senf.extractor import extract_senf
from core.senf.types import EntityRef, FrameRef, KindRef, ValueRef
from core.senf.feature_binding import apply_feature_bindings, bind_features
from core.senf.features import CoreferenceEvidence, ExactSpan, FeatureBatch, SpanFeature


def _sample():
    return extract_senf(
        "s1",
        "Kebede is a researcher and eats 3 fish.",
        [
            "(: kind (IsA kebede researcher) (STV 1 1))",
            "(: eat (Eats kebede 3 (Fresh fish)) (STV 1 1))",
        ],
    )


def test_v7_payload_is_json_safe_and_round_trips_every_field():
    original = _sample()
    payload = senf_to_payload(original)
    assert payload["senf_version"] == 7 == SENF_PAYLOAD_VERSION
    assert json.loads(json.dumps(payload)) == payload
    assert senf_from_payload(json.loads(json.dumps(payload))) == original


def test_round_trip_preserves_all_reference_types():
    restored = senf_from_payload(senf_to_payload(_sample()))
    assert restored is not None
    fillers = [role.filler for frame in restored.frames for role in frame.roles]
    assert any(isinstance(filler, EntityRef) for filler in fillers)
    assert any(isinstance(filler, KindRef) for filler in fillers)
    assert any(isinstance(filler, ValueRef) for filler in fillers)
    assert any(isinstance(filler, FrameRef) for filler in fillers)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 8, 99, "7", True, None])
def test_absent_old_future_and_malformed_versions_are_unsupported(version):
    payload = senf_to_payload(_sample())
    payload["senf_version"] = version
    assert senf_from_payload(payload) is None


def test_static_v5_fixture_migrates_to_typed_v7_model():
    fixture = Path(__file__).with_name("fixtures") / "senf_v5.json"
    payload = json.loads(fixture.read_text())

    restored = senf_from_payload(payload)

    assert restored is not None
    assert restored.mentions[0].char_span.start == 0
    assert restored.frames[0].frame_span == (0, 12)
    assert restored.frames[0].clause_span == (0, 13)
    assert senf_to_payload(restored)["senf_version"] == 7


def test_static_v6_fixture_migrates_with_empty_feature_audit():
    fixture = Path(__file__).with_name("fixtures") / "senf_v6.json"
    payload = json.loads(fixture.read_text())

    restored = senf_from_payload(payload)

    assert restored is not None
    assert restored.applied_mention_features == []
    assert restored.coreference_evidence == []
    assert senf_to_payload(restored)["senf_version"] == 7


def test_v7_requires_span_and_guard_fields():
    payload = senf_to_payload(_sample())
    del payload["frames"][0]["frame_span"]
    assert senf_from_payload(payload) is None

    from core.senf.exemplars import score_exemplars

    scored = extract_senf(
        "camera", "The camera arrived.",
        ["(: a (Arrived camera) (STV 1 1))"],
    )
    payload = senf_to_payload(score_exemplars(scored))
    score = next(iter(payload["exemplar_scores"].values()))[0]
    del score["guard"]
    assert senf_from_payload(payload) is None


@pytest.mark.parametrize("blob", [None, {}, "bad", [], 42])
def test_absent_or_non_payload_senf_is_safe(blob):
    assert senf_from_payload(blob) is None


def test_malformed_v5_payload_fails_closed():
    payload = senf_to_payload(_sample())
    payload["frames"][0]["roles"][0]["filler"] = {
        "type": "entity_ref",
        "entity_id": "missing",
    }
    assert senf_from_payload(payload) is None


def test_dangling_frame_ref_fails_closed():
    payload = senf_to_payload(_sample())
    nested = next(
        role
        for frame in payload["frames"]
        for role in frame["roles"]
        if role["filler"]["type"] == "frame_ref"
    )
    nested["filler"]["frame_id"] = "missing"
    assert senf_from_payload(payload) is None


def test_cyclic_frame_ref_fails_closed():
    payload = senf_to_payload(_sample())
    parent = next(
        frame
        for frame in payload["frames"]
        if any(role["filler"]["type"] == "frame_ref" for role in frame["roles"])
    )
    child = next(
        frame
        for frame in payload["frames"]
        if frame["frame_id"] == next(
            role["filler"]["frame_id"]
            for role in parent["roles"]
            if role["filler"]["type"] == "frame_ref"
        )
    )
    child["roles"].append({
        "name": "Arg2",
        "position": len(child["roles"]),
        "filler": {"type": "frame_ref", "frame_id": parent["frame_id"]},
    })

    assert senf_from_payload(payload) is None


def test_frame_ref_cannot_cross_atom_provenance():
    payload = senf_to_payload(_sample())
    parent = next(
        frame
        for frame in payload["frames"]
        if any(role["filler"]["type"] == "frame_ref" for role in frame["roles"])
    )
    parent["source_atom_id"] = "forged"

    assert senf_from_payload(payload) is None


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("senf_id",), ""),
        (("sentence_id",), ""),
        (("entities", 0, "entity_id"), ""),
        (("mentions", 0, "mention_id"), ""),
        (("mentions", 0, "sentence_id"), "other"),
        (("mentions", 0, "char_span"), [-1, 2]),
        (("mentions", 0, "char_span"), [4, 2]),
        (("mentions", 0, "char_span"), [0, 10_000]),
        (("frames", 0, "frame_id"), ""),
        (("frames", 0, "source_sentence_id"), "other"),
        (("frames", 0, "source_text"), 7),
        (("frames", 0, "modality"), 7),
        (("frames", 0, "time_ref"), 7),
        (("frames", 0, "location_ref"), 7),
    ],
)
def test_v5_rejects_invalid_ids_sources_and_spans(path, bad_value):
    payload = senf_to_payload(_sample())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    assert senf_from_payload(payload) is None


def test_v5_rejects_kind_assertion_not_backed_by_matching_isa_frame():
    payload = senf_to_payload(_sample())
    payload["kind_assertions"][0]["kind"] = "clinician"
    assert senf_from_payload(payload) is None


def test_v5_rejects_isa_frame_without_kind_assertion():
    payload = senf_to_payload(_sample())
    payload["kind_assertions"] = []
    assert senf_from_payload(payload) is None


def test_v5_rejects_surface_that_does_not_match_its_span():
    payload = senf_to_payload(_sample())
    mention = next(item for item in payload["mentions"] if item["char_span"])
    mention["surface"] = "forged"
    assert senf_from_payload(payload) is None


@pytest.mark.parametrize("distance", [-0.1, 1.1, float("inf"), float("nan")])
def test_v5_rejects_unbounded_or_nonfinite_exemplar_distance(distance):
    senf = _sample()
    mention_id = senf.mentions[0].mention_id
    payload = senf_to_payload(senf)
    payload["exemplar_scores"] = {
        mention_id: [{
            "kind": "researcher", "exemplar": "scientist", "distance": distance,
            "reasons": [],
        }]
    }
    assert senf_from_payload(payload) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exemplar_scores", {"missing": []}),
        ("nearest_exemplars", {"missing": "scientist"}),
        ("constraints", [7]),
    ],
)
def test_v5_rejects_annotation_keys_and_types_that_are_not_mentions(field, value):
    payload = senf_to_payload(_sample())
    payload[field] = value
    assert senf_from_payload(payload) is None


def test_v5_requires_exemplar_reasons_to_be_a_string_list():
    payload = senf_to_payload(_sample())
    mention_id = payload["mentions"][0]["mention_id"]
    payload["exemplar_scores"] = {
        mention_id: [{
            "kind": "researcher", "exemplar": "scientist", "distance": 0.2,
            "reasons": "not-a-list",
        }]
    }
    assert senf_from_payload(payload) is None


def test_v5_rejects_forged_cross_kind_exemplar():
    senf = extract_senf(
        "s1", "The camera is professional.",
        ["(: kind (IsA camera camera) (STV 1 1))"],
    )
    payload = senf_to_payload(senf)
    mention_id = payload["mentions"][0]["mention_id"]
    payload["exemplar_scores"] = {mention_id: [{
        "kind": "game", "exemplar": "chess_game", "distance": 0.2,
        "reasons": ["professional"],
    }]}

    assert senf_from_payload(payload) is None


def test_v5_allows_registered_lexical_applicability_without_isa():
    from core.senf.exemplars import score_exemplars

    senf = score_exemplars(extract_senf(
        "s1", "The Nikon camera arrived.",
        ["(: arrived (Arrived camera) (STV 1 1))"],
    ))

    assert senf.kind_assertions == []
    assert senf.exemplar_scores
    assert senf_from_payload(senf_to_payload(senf)) == senf


def test_unknown_keys_are_ignored_only_on_otherwise_valid_v5():
    payload = senf_to_payload(_sample())
    payload["unknown_future_key"] = {"ignored": True}
    assert senf_from_payload(payload) == _sample()


def test_v3_payload_fails_closed_without_migration():
    payload = senf_to_payload(_sample())
    payload["senf_version"] = 3
    assert senf_from_payload(payload) is None


@pytest.mark.parametrize(
    "field", [
        "source_units", "active_exemplars", "constraints", "exemplar_scores",
        "branches", "validity_intervals", "entity_persistence", "source_atoms",
    ]
)
def test_v5_requires_all_typed_fields(field):
    payload = senf_to_payload(_sample())
    del payload[field]
    assert senf_from_payload(payload) is None


def test_v5_rejects_orphan_entities_and_mentions():
    payload = senf_to_payload(_sample())
    payload["entities"].append({"entity_id": "s1:orphan", "canonical_symbol": "ghost"})
    assert senf_from_payload(payload) is None

    payload = senf_to_payload(_sample())
    mention = dict(payload["mentions"][0])
    mention.update({
        "mention_id": "s1:orphan", "char_span": None, "surface": "ghost",
        "source_unit_id": payload["source_units"][0]["source_unit_id"],
    })
    payload["mentions"].append(mention)
    assert senf_from_payload(payload) is None


@pytest.mark.parametrize("mutation", ["missing_context", "wrong_unit", "bad_constraint"])
def test_v5_rejects_inconsistent_frame_context_and_typed_constraints(mutation):
    senf = extract_senf(
        "s1", "Sam borrowed the lens on Monday.",
        ["(: a (Borrowed sam lens monday) (STV 1 1))"],
    )
    payload = senf_to_payload(senf)
    if mutation == "missing_context":
        payload["frames"][0]["context"] = None
    elif mutation == "wrong_unit":
        payload["source_units"].append({
            "source_unit_id": "s1:u1", "sentence_id": "s1", "text": "", "char_span": [0, 0],
        })
        payload["frames"][0]["context"]["source_unit_id"] = "s1:u1"
    else:
        payload["constraints"][0]["value"] = "tuesday"

    assert senf_from_payload(payload) is None


def test_store_merges_senf_without_disturbing_nl_and_pln(fake_vector_store):
    senf = _sample()
    atoms = ["(: eat (Eats kebede fish) (STV 1 1))"]
    fake_vector_store.store(
        "Kebede eats fish.",
        atoms,
        fake_vector_store.embed("Kebede eats fish."),
        metadata={SENF_PAYLOAD_KEY: senf_to_payload(senf)},
    )
    payload = fake_vector_store.points[-1]["payload"]
    assert payload["nl"] == "Kebede eats fish."
    assert payload["pln"] == atoms
    assert senf_from_payload(payload[SENF_PAYLOAD_KEY]) == senf
    assert fake_vector_store.retrieve_senf_context("fish", 1) == [{
        SENF_PAYLOAD_KEY: payload[SENF_PAYLOAD_KEY],
        "nl": "Kebede eats fish.",
        "pln": atoms,
    }]


def test_store_without_metadata_writes_no_senf_key(fake_vector_store):
    fake_vector_store.store("plain", [], [0.0])
    payload = fake_vector_store.points[-1]["payload"]
    assert SENF_PAYLOAD_KEY not in payload
    assert senf_from_payload(payload.get(SENF_PAYLOAD_KEY)) is None


def test_v7_round_trip_preserves_applied_feature_and_coreference_audit():
    text = "Alice arrived. She smiled."
    senf = extract_senf(
        "s1", text,
        ["(: a (Arrived alice) (STV 1 1))", "(: b (Smiled she) (STV 1 1))"],
    )
    alice, she = senf.mentions
    batch = FeatureBatch(
        (SpanFeature(ExactSpan(*she.char_span, she.surface), "mention_type", "pronoun"),),
        (CoreferenceEvidence(
            ExactSpan(*she.char_span, she.surface),
            ExactSpan(*alice.char_span, alice.surface), 0.8, ("test",),
        ),),
    )
    apply_feature_bindings(
        senf, bind_features(text, senf.mentions, batch), provider="test"
    )

    restored = senf_from_payload(senf_to_payload(senf))
    assert restored == senf
    assert restored.applied_mention_features[0].mention_id == she.mention_id
    assert restored.coreference_evidence[0].antecedent_mention_id == alice.mention_id


def test_v7_rejects_forged_applied_feature_audit():
    payload = senf_to_payload(_sample())
    mention = payload["mentions"][0]
    payload["applied_mention_features"] = [{
        "mention_id": mention["mention_id"],
        "entity_id": mention["entity_id"],
        "name": "mention_type",
        "value": "pronoun",
        "confidence": 1.0,
        "provider": "test",
        "source_span": mention["char_span"],
        "source_text": mention["surface"],
        "evidence": [],
    }]
    assert senf_from_payload(payload) is None
