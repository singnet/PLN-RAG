from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from core.senf.feature_binding import bind_features
from core.senf.feature_provider import (
    FailOpenFeatureProvider,
    NoOpFeatureProvider,
    provide_features,
)
from core.senf.features import (
    CoreferenceEvidence,
    ExactSpan,
    FeatureBatch,
    SpanFeature,
    validate_feature_batch,
)
from core.senf.langextract_features import features_from_langextract
from core.senf.types import Mention


def _extraction(cls, text, start, end, attrs, alignment="MATCH_EXACT"):
    return SimpleNamespace(
        extraction_class=cls,
        extraction_text=text,
        char_interval=SimpleNamespace(start_pos=start, end_pos=end),
        attributes=attrs,
        alignment_status=SimpleNamespace(name=alignment),
    )


def test_feature_models_are_immutable_and_validate_exact_source_slices():
    feature = SpanFeature(ExactSpan(0, 5, "Alice"), "mention_type", "proper")
    with pytest.raises(FrozenInstanceError):
        feature.value = "common"

    result = validate_feature_batch(
        "Alice runs", FeatureBatch((feature, SpanFeature(ExactSpan(6, 10, "walk"), "lemma", "walk")))
    )
    assert result.accepted.features == (feature,)
    assert result.rejected[0].reason == "span does not exactly match source"


def test_binding_requires_exact_span_and_coreference_remains_evidence_only():
    text = "Alice arrived. She smiled."
    alice = Mention("Alice", "parser_owned_alice", "s1", "e1", "m1", (0, 5))
    she = Mention("She", "parser_owned_she", "s1", "e2", "m2", (15, 18))
    link = CoreferenceEvidence(ExactSpan(15, 18, "She"), ExactSpan(0, 5, "Alice"), 0.8)
    feature = SpanFeature(ExactSpan(15, 18, "She"), "mention_type", "pronoun")

    result = bind_features(text, [alice, she], FeatureBatch((feature,), (link,)))

    assert result.features[0].mention_id == "m2"
    assert result.coreferences[0].anaphor_mention_id == "m2"
    assert result.coreferences[0].antecedent_mention_id == "m1"
    assert (alice.canonical_symbol, she.canonical_symbol) == (
        "parser_owned_alice", "parser_owned_she"
    )
    assert alice.entity_id != she.entity_id


def test_binding_rejects_nonmatching_mention_surface():
    mention = Mention("ALICE", "alice", "s1", "e1", "m1", (0, 5))
    feature = SpanFeature(ExactSpan(0, 5, "Alice"), "mention_type", "proper")
    result = bind_features("Alice", [mention], FeatureBatch((feature,)))
    assert not result.features
    assert result.rejected[0].reason == "no exact mention span"


def test_binding_rejects_ambiguous_exact_mentions():
    feature = SpanFeature(ExactSpan(0, 2, "It"), "mention_type", "pronoun")
    mentions = [
        Mention("It", "it", "s1", "e1", "m1", (0, 2)),
        Mention("It", "it", "s1", "e2", "m2", (0, 2)),
    ]

    result = bind_features("It moved.", mentions, FeatureBatch((feature,)))

    assert result.features == ()
    assert result.rejected[0].reason == "ambiguous exact mention span"


def test_optional_provider_is_noop_and_fail_open():
    class BrokenProvider:
        def provide(self, source_text):
            raise RuntimeError("offline")

    assert provide_features("text").accepted == FeatureBatch.empty()
    assert NoOpFeatureProvider().provide("text") == FeatureBatch.empty()
    assert FailOpenFeatureProvider(BrokenProvider()).provide("text") == FeatureBatch.empty()


def test_fail_open_rejects_wrong_provider_return_type():
    class MalformedProvider:
        def provide(self, source_text):
            return {"canonical_symbol": "provider_must_not_own_this"}

    assert provide_features("text", MalformedProvider()).accepted == FeatureBatch.empty()


def test_langextract_accepts_only_exact_feature_output_and_builds_coref_evidence():
    text = "Alice arrived. She smiled."
    extractions = [
        _extraction("senf_feature", "She", 15, 18, {"name": "mention_type", "value": "pronoun"}),
        _extraction("senf_coreference", "Alice", 0, 5, {"group": "person-1"}),
        _extraction("senf_coreference", "She", 15, 18, {"group": "person-1", "confidence": 0.7}),
    ]
    result = features_from_langextract(text, extractions)
    assert [(item.name, item.value) for item in result.accepted.features] == [
        ("mention_type", "pronoun")
    ]
    assert result.accepted.coreferences[0].antecedent.text == "Alice"
    assert result.accepted.coreferences[0].anaphor.text == "She"
    assert result.accepted.coreferences[0].confidence == 0.7


@pytest.mark.parametrize(
    "extraction",
    [
        _extraction("senf_feature", "Alic", 0, 5, {"name": "lemma", "value": "alice"}),
        _extraction("senf_feature", "Alice", 0, 5, {"name": "lemma", "value": "alice"}, "MATCH_FUZZY"),
        _extraction("senf_feature", "Alice", 0, 5, {"name": "lemma"}),
        _extraction("senf_feature", "Alice", 0, 5, {"name": "lemma", "value": "alice", "canonical_symbol": "alice"}),
        _extraction("senf_feature", "Alice", 0, 5, {"name": "canonical-symbol", "value": "alice"}),
        _extraction("fact", "Alice", 0, 5, {"predicate": "is-a"}),
    ],
)
def test_langextract_rejects_malformed_fuzzy_nonexact_and_symbol_output(extraction):
    result = features_from_langextract("Alice arrived", [extraction])
    assert result.accepted == FeatureBatch.empty()
    assert result.rejected
