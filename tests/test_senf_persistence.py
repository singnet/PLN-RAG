import json

import pytest

from core.senf import (
    SENF_PAYLOAD_KEY,
    SENF_PAYLOAD_VERSION,
    senf_from_payload,
    senf_to_payload,
)
from core.senf.extractor import extract_senf
from core.senf.types import Literal, Mention, Role, SENF, SENFFrame


def _sample_senf() -> SENF:
    kebede = Mention(
        surface="Kebede",
        canonical_symbol="kebede",
        sentence_id="s1",
        mention_id="s1:m0",
        char_span=(0, 6),
        mention_type="proper",
        head_lemma="kebede",
    )
    fish = Mention(
        surface="fish",
        canonical_symbol="fish",
        sentence_id="s1",
        mention_id="s1:m1",
        char_span=(12, 16),
        mention_type="common",
        head_lemma="fish",
    )
    frame = SENFFrame(
        frame_id="f1",
        predicate_head="Eats",
        roles=[
            Role("arg0", kebede),
            Role("arg1", fish),
            Role("arg2", Literal("3", "number")),
        ],
        polarity=False,
        modality="possible",
        time_ref="past",
        location_ref="addis_ababa",
        source_sentence_id="s1",
        source_text="Kebede eats fish.",
    )
    return SENF(
        senf_id="senf-1",
        sentence_id="s1",
        frames=[frame],
        mentions=[kebede, fish],
        kinds={"kebede": "human"},
    )


def test_payload_is_json_serializable():
    """Qdrant payloads cross the wire as JSON, so tuples must already be lists."""
    payload = senf_to_payload(_sample_senf())
    assert json.loads(json.dumps(payload)) == payload


def test_round_trip_preserves_every_field():
    original = _sample_senf()
    restored = senf_from_payload(senf_to_payload(original))

    assert restored is not None
    assert restored.senf_id == original.senf_id
    assert restored.sentence_id == original.sentence_id
    assert restored.kinds == original.kinds
    assert restored.mentions == original.mentions
    assert restored.frames == original.frames


def test_round_trip_survives_a_real_json_hop():
    """A tuple char_span becomes a list in transit; it must come back a tuple."""
    original = _sample_senf()
    restored = senf_from_payload(json.loads(json.dumps(senf_to_payload(original))))

    assert restored is not None
    assert restored.mentions[0].char_span == (0, 6)
    assert restored.frames == original.frames


def test_round_trip_of_extractor_output():
    """Extractor output must round-trip through the persistence format."""
    senf = extract_senf(
        "s1",
        "Kebede eats fish.",
        ["(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))"],
    )
    restored = senf_from_payload(senf_to_payload(senf))

    assert restored is not None
    assert restored.frames == senf.frames
    assert restored.mentions == senf.mentions
    assert restored.symbols() == senf.symbols()


def test_frame_mention_fillers_are_the_same_objects_as_the_mention_list():
    """Fillers are stored by mention reference, not duplicated inline.

    If rehydration built fresh Mentions per role, a frame filler would lose the
    span and type carried by the mention list and identity resolution would see
    two different objects for one entity.
    """
    restored = senf_from_payload(senf_to_payload(_sample_senf()))

    assert restored is not None
    by_symbol = {m.canonical_symbol: m for m in restored.mentions}
    filler = restored.frames[0].roles[0].filler
    assert filler == by_symbol["kebede"]
    assert filler.mention_type == "proper"
    assert filler.char_span == (0, 6)


def test_literal_fillers_survive_as_literals():
    restored = senf_from_payload(senf_to_payload(_sample_senf()))

    assert restored is not None
    filler = restored.frames[0].roles[2].filler
    assert isinstance(filler, Literal)
    assert (filler.value, filler.literal_type) == ("3", "number")


def test_polarity_false_is_not_lost():
    """False is the one boolean a sloppy `or` default would silently flip."""
    restored = senf_from_payload(senf_to_payload(_sample_senf()))

    assert restored is not None
    assert restored.frames[0].polarity is False


def test_version_is_stamped():
    assert senf_to_payload(_sample_senf())["senf_version"] == SENF_PAYLOAD_VERSION


@pytest.mark.parametrize(
    "blob",
    [
        None,
        {},
        "not-a-dict",
        [],
        42,
        {"senf_version": "1"},
        {"senf_version": SENF_PAYLOAD_VERSION + 1, "senf_id": "future"},
        {"mentions": "not-a-list"},
        {"senf_id": "x", "mentions": [None, 7], "frames": ["nope"]},
    ],
)
def test_unusable_blobs_read_back_as_none(blob):
    """A legacy or corrupt point must degrade to pre-SENF behavior, not raise."""
    assert senf_from_payload(blob) is None


def test_partial_blob_recovers_what_it_can():
    """Forward compatibility: an unknown key is ignored, known ones still load."""
    restored = senf_from_payload(
        {
            "senf_version": SENF_PAYLOAD_VERSION,
            "senf_id": "senf-2",
            "sentence_id": "s9",
            "mentions": [{"surface": "Abebe", "symbol": "abebe", "sentence_id": "s9"}],
            "frames": [],
            "unknown_future_key": {"ignored": True},
        }
    )

    assert restored is not None
    assert restored.senf_id == "senf-2"
    assert restored.symbols() == {"abebe"}


def test_frame_referencing_an_absent_mention_keeps_the_role():
    """Dropping the role would silently change a frame's arity."""
    restored = senf_from_payload(
        {
            "senf_version": SENF_PAYLOAD_VERSION,
            "senf_id": "senf-3",
            "sentence_id": "s3",
            "mentions": [],
            "frames": [
                {
                    "frame_id": "f1",
                    "predicate_head": "Eats",
                    "roles": [{"name": "arg0", "filler": {"kind": "mention", "symbol": "ghost"}}],
                }
            ],
        }
    )

    assert restored is not None
    assert restored.frames[0].filler_symbols() == ["ghost"]


def test_store_merges_senf_without_disturbing_nl_and_pln(fake_vector_store):
    senf = _sample_senf()
    atoms = ["(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))"]

    fake_vector_store.store(
        "Kebede eats fish.",
        atoms,
        fake_vector_store.embed("Kebede eats fish."),
        metadata={SENF_PAYLOAD_KEY: senf_to_payload(senf)},
    )

    payload = fake_vector_store.points[-1]["payload"]
    assert payload["nl"] == "Kebede eats fish."
    assert payload["pln"] == atoms
    assert senf_from_payload(payload[SENF_PAYLOAD_KEY]).frames == senf.frames


def test_store_without_metadata_writes_no_senf_key(fake_vector_store):
    """Parsers that never produce SENF must keep writing the old payload shape."""
    fake_vector_store.store("Kebede eats fish.", ["(: a (Eats kebede fish) (STV 1.0 1.0))"], [0.0])

    payload = fake_vector_store.points[-1]["payload"]
    assert SENF_PAYLOAD_KEY not in payload
    assert senf_from_payload(payload.get(SENF_PAYLOAD_KEY)) is None


def test_atom_context_retrieval_is_unaffected_by_senf(fake_vector_store):
    """retrieve_context must return the same atoms whether or not SENF is present."""
    atoms = ["(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))"]
    fake_vector_store.store("plain.", atoms, fake_vector_store.embed("plain."))
    fake_vector_store.store(
        "with senf.",
        atoms,
        fake_vector_store.embed("with senf."),
        metadata={SENF_PAYLOAD_KEY: senf_to_payload(_sample_senf())},
    )

    context, _ = fake_vector_store.retrieve_context("anything", top_k=2)
    assert context == atoms + atoms
