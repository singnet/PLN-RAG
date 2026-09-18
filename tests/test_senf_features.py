from dataclasses import FrozenInstanceError
import multiprocessing
import time
from types import SimpleNamespace

import pytest

from core.senf.feature_binding import apply_feature_bindings, bind_features
from core.senf.extractor import extract_senf
from core.senf.identity import resolve_identity
from core.senf.feature_provider import (
    FailOpenFeatureProvider,
    NoOpFeatureProvider,
    create_feature_provider,
    provide_features,
)
from core.senf.features import (
    CoreferenceEvidence,
    ExactSpan,
    FeatureBatch,
    FeatureRejection,
    FeatureValidation,
    SpanFeature,
    validate_feature_batch,
)
from core.senf.langextract_features import LangExtractFeatureProvider, features_from_langextract
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


def test_optional_provider_is_noop_and_fail_open(caplog):
    class BrokenProvider:
        def provide(self, source_text):
            raise RuntimeError("offline token=super-secret")

    assert provide_features("text").accepted == FeatureBatch.empty()
    assert NoOpFeatureProvider().provide("text") == FeatureBatch.empty()
    result = FailOpenFeatureProvider(BrokenProvider()).provide("text")
    assert result.accepted == FeatureBatch.empty()
    assert result.rejected[0].category == "provider"
    assert result.rejected[0].reason == "feature provider failed"
    assert "super-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_fail_open_rejects_wrong_provider_return_type():
    class MalformedProvider:
        def provide(self, source_text):
            return {"canonical_symbol": "provider_must_not_own_this"}

    assert provide_features("text", MalformedProvider()).accepted == FeatureBatch.empty()


def test_provider_validation_is_preserved_and_accepted_batch_is_revalidated():
    invalid = SpanFeature(ExactSpan(0, 4, "text"), "mention_type", "common")

    class ValidatedProvider:
        def provide(self, source_text):
            return FeatureValidation(
                FeatureBatch((invalid,)),
                (FeatureRejection("LangExtract", 7, "feature limit exceeded"),),
            )

    result = provide_features("different", ValidatedProvider())

    assert result.accepted == FeatureBatch.empty()
    assert [item.reason for item in result.rejected] == [
        "feature limit exceeded",
        "span does not exactly match source",
    ]
    assert result.rejected[0].category == "langextract"


def test_provider_rejection_text_is_redacted_and_malformed_entries_are_dropped():
    class UntrustedProvider:
        def provide(self, source_text):
            return FeatureValidation(
                FeatureBatch.empty(),
                (
                    FeatureRejection("Custom-Provider", 4, "token=super-secret"),
                    {"reason": "token=another-secret"},
                    FeatureRejection("provider", -1, "token=third-secret"),
                ),
            )

    result = provide_features("text", UntrustedProvider())

    assert result.rejected == (
        FeatureRejection("provider", 4, "provider rejection details redacted"),
    )
    assert "secret" not in repr(result)


def test_combined_provider_rejections_are_bounded():
    class NoisyProvider:
        def provide(self, source_text):
            return FeatureValidation(
                FeatureBatch.empty(),
                tuple(
                    FeatureRejection("provider", i, "feature limit exceeded")
                    for i in range(10)
                ),
            )

    assert len(provide_features("text", NoisyProvider(), max_rejections=3).rejected) == 3


def test_fail_open_does_not_hide_strict_replay_integrity_errors():
    class ReplayError(RuntimeError):
        fail_closed = True

    class BrokenReplay:
        def provide(self, source_text):
            raise ReplayError("hash mismatch")

    with pytest.raises(ReplayError, match="hash mismatch"):
        provide_features("text", BrokenReplay())


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


def test_applied_features_are_allowlisted_audited_and_do_not_change_symbols_or_roles():
    text = "The professional camera arrived."
    senf = extract_senf("s1", text, ["(: a (Arrived professional_camera) (STV 1 1))"])
    mention = senf.mentions[0]
    original_symbol = mention.canonical_symbol
    original_roles = list(senf.frames[0].roles)
    start, end = mention.char_span
    batch = FeatureBatch((
        SpanFeature(ExactSpan(start, end, mention.surface), "definiteness", "definite"),
        SpanFeature(ExactSpan(start, end, mention.surface), "exemplar_cue", "professional"),
        SpanFeature(ExactSpan(start, end, mention.surface), "location_ref", "studio"),
        SpanFeature(ExactSpan(start, end, mention.surface), "canonical_symbol", "forged"),
    ))

    rejected = apply_feature_bindings(
        senf, bind_features(text, senf.mentions, batch), provider="test"
    )

    assert senf.mentions[0].canonical_symbol == original_symbol
    assert senf.frames[0].roles == original_roles
    assert senf.mentions[0].definiteness == "definite"
    assert senf.frames[0].location_ref == "studio"
    assert {item.name for item in senf.applied_mention_features} == {
        "definiteness", "exemplar_cue", "location_ref"
    }
    assert any(item.reason == "feature cannot augment SENF" for item in rejected)


def test_exemplar_cue_must_occur_in_the_exact_bound_span():
    text = "The camera arrived."
    senf = extract_senf("s1", text, ["(: a (Arrived camera) (STV 1 1))"])
    mention = senf.mentions[0]
    feature = SpanFeature(
        ExactSpan(*mention.char_span, mention.surface), "exemplar_cue", "professional"
    )

    rejected = apply_feature_bindings(
        senf, bind_features(text, senf.mentions, FeatureBatch((feature,))), provider="test"
    )

    assert not senf.applied_mention_features
    assert rejected[0].reason == "exemplar cue is not present in its exact source span"


def test_zero_confidence_feature_is_audit_only():
    text = "The camera arrived."
    senf = extract_senf("s1", text, ["(: a (Arrived camera) (STV 1 1))"])
    mention = senf.mentions[0]
    original = mention.mention_type
    feature = SpanFeature(
        ExactSpan(*mention.char_span, mention.surface),
        "mention_type", "pronoun", confidence=0.0,
    )

    apply_feature_bindings(
        senf, bind_features(text, senf.mentions, FeatureBatch((feature,))), provider="test"
    )

    assert senf.mentions[0].mention_type == original
    assert senf.applied_mention_features[0].confidence == 0.0


def test_langextract_provider_controls_exactness_and_max_features():
    fuzzy = _extraction(
        "senf_feature", "She", 0, 3,
        {"name": "mention_type", "value": "pronoun"}, "MATCH_FUZZY",
    )
    exact = _extraction(
        "senf_feature", "She", 0, 3,
        {"name": "definiteness", "value": "pronoun"},
    )
    provider = LangExtractFeatureProvider(
        lambda _: [fuzzy, exact], exact_only=False, max_features=1,
    )
    assert provider.provide("She").accepted.features == (
        SpanFeature(ExactSpan(0, 3, "She"), "mention_type", "pronoun", 1.0, ("langextract",)),
    )


def test_langextract_provider_bounds_an_unbounded_injected_iterable():
    exact = _extraction(
        "senf_feature", "She", 0, 3,
        {"name": "mention_type", "value": "pronoun"},
    )

    def unbounded(_):
        while True:
            yield exact

    result = LangExtractFeatureProvider(unbounded, max_features=2).provide("She")

    assert len(result.accepted.features) == 2
    assert [(item.index, item.reason) for item in result.rejected] == [
        (2, "feature limit exceeded")
    ]


def test_langextract_timeout_terminates_worker_process():
    def slow_extractor(_):
        time.sleep(10)
        return []

    before = {child.pid for child in multiprocessing.active_children()}
    provider = LangExtractFeatureProvider(slow_extractor, timeout=0.05)

    with pytest.raises(TimeoutError, match="timed out"):
        provider.provide("She")

    assert {child.pid for child in multiprocessing.active_children()} == before


def test_feature_provider_factory_is_lazy_and_honors_none_or_langextract():
    settings = SimpleNamespace(
        senf_feature_provider="none",
        senf_feature_exact_only=True,
        senf_feature_max_features=4,
        senf_feature_timeout=3.0,
        senf_feature_model="feature-model",
        senf_feature_examples_path="features.json",
        langextract_model_id="fallback-model",
        langextract_model_url="http://model",
        langextract_api_key="key",
        openai_api_key="other",
    )
    assert isinstance(create_feature_provider(settings), NoOpFeatureProvider)
    settings.senf_feature_provider = "langextract"
    provider = create_feature_provider(settings)
    assert isinstance(provider, LangExtractFeatureProvider)
    assert provider._model_id == "feature-model"
    assert provider._examples_path == "features.json"


@pytest.mark.parametrize(
    ("polarity", "evidence_name"),
    [("positive", "provider_coreference"), ("negative", "provider_coreference_conflict")],
)
def test_provider_coreference_augments_identity_evidence_without_direct_merge(
    polarity, evidence_name
):
    text = "Alice arrived. She smiled."
    senf = extract_senf(
        "s1", text,
        ["(: a (Arrived alice) (STV 1 1))", "(: b (Smiled she) (STV 1 1))"],
    )
    alice, she = senf.mentions
    link = CoreferenceEvidence(
        ExactSpan(*she.char_span, she.surface),
        ExactSpan(*alice.char_span, alice.surface),
        0.9, ("provider",), polarity,
    )
    apply_feature_bindings(
        senf,
        bind_features(text, senf.mentions, FeatureBatch(coreferences=(link,))),
        provider="test",
    )

    graph = resolve_identity([senf])
    edge = next(item for item in graph.edges if set(item.mention_ids) == {
        alice.mention_id, she.mention_id
    })
    assert evidence_name in edge.evidence + edge.negative_evidence
    record = next(item for item in edge.identity_evidence if item.kind == evidence_name)
    expected_weight = 0.4 if polarity == "positive" else 0.9
    assert record.weight == pytest.approx(expected_weight * 0.9)
    assert record.confidence == 0.9
    assert alice.entity_id != she.entity_id


def test_zero_confidence_coreference_has_no_identity_effect():
    text = "Alice arrived. She smiled."
    senf = extract_senf(
        "s1", text,
        ["(: a (Arrived alice) (STV 1 1))", "(: b (Smiled she) (STV 1 1))"],
    )
    alice, she = senf.mentions
    link = CoreferenceEvidence(
        ExactSpan(*she.char_span, she.surface),
        ExactSpan(*alice.char_span, alice.surface),
        0.0, ("provider",), "negative",
    )
    apply_feature_bindings(
        senf, bind_features(text, senf.mentions, FeatureBatch(coreferences=(link,))),
        provider="test",
    )

    graph = resolve_identity([senf])

    assert all("provider_coreference_conflict" not in edge.negative_evidence for edge in graph.edges)
