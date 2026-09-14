import threading
import unittest

from core.coreference import (
    CoreferenceBackend,
    CoreferenceCluster,
    CoreferenceMention,
    CoreferencePrediction,
    CoreferenceResolver,
)


class FakeBackend(CoreferenceBackend):
    name = "fake"
    model_name = "fake/model"
    loaded = True
    load_error = None

    def __init__(self, clusters=(), error=None, score_available=False):
        self.clusters = tuple(clusters)
        self.error = error
        self.score_available = score_available
        self.calls = 0
        self.lock = threading.Lock()

    def predict(self, text):
        with self.lock:
            self.calls += 1
        if self.error:
            raise self.error
        return CoreferencePrediction(
            backend=self.name,
            model=self.model_name,
            clusters=self.clusters,
            score_available=self.score_available,
        )


def mention(text, value, occurrence=0, score=None):
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.index(value, cursor)
        cursor = start + len(value)
    return CoreferenceMention(start, start + len(value), value, score)


def cluster(cluster_id, *mentions):
    return CoreferenceCluster(cluster_id, tuple(mentions))


class CoreferenceResolverTests(unittest.TestCase):
    def resolve_with(self, text, *clusters):
        backend = FakeBackend(clusters)
        resolver = CoreferenceResolver(enabled=True, backend=backend)
        return resolver, resolver.resolve(text)

    def test_disabled_and_empty_inputs_do_not_call_backend(self):
        backend = FakeBackend()
        disabled = CoreferenceResolver(enabled=False, backend=backend)
        self.assertEqual(disabled.resolve("Alice said she left.").status, "disabled")
        self.assertEqual(backend.calls, 0)

        enabled = CoreferenceResolver(enabled=True, backend=backend)
        self.assertEqual(enabled.resolve("  ").status, "unchanged")
        self.assertEqual(backend.calls, 0)

    def test_plain_and_possessive_pronouns_are_rewritten(self):
        text = "Alice arrived. She checked her notes because they mattered to her."
        resolver, result = self.resolve_with(
            text,
            cluster(0, mention(text, "Alice"), mention(text, "She")),
            cluster(1, mention(text, "notes"), mention(text, "they")),
        )

        self.assertEqual(
            result.resolved,
            "Alice arrived. Alice checked her notes because notes mattered to her.",
        )
        self.assertEqual(len(result.replacements), 2)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(resolver.status().documents_changed, 1)

    def test_ambiguous_her_is_never_rewritten(self):
        text = "Alice met Bob. Bob thanked her and praised her work."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "Alice"),
                mention(text, "her", 0),
                mention(text, "her", 1),
            ),
        )
        self.assertEqual(result.resolved, text)
        self.assertEqual(result.replacements, ())

    def test_unambiguous_possessives_use_correct_apostrophe(self):
        text = "James entered. His coat was wet. The coat was his."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "James"),
                mention(text, "His"),
                mention(text, "his", 0),
            ),
        )
        self.assertEqual(
            result.resolved,
            "James entered. James' coat was wet. The coat was James'.",
        )

    def test_structurally_noisy_possessive_antecedents_are_not_rewritten(self):
        cases = [
            ("A total of 200 patients arrived. Their results varied.", "A total of 200 patients"),
            ("The other contact lens materials remained stable. Their values changed.", "The other contact lens materials"),
            ("A polymer, serafilcon A, was tested. Its reserve declined.", "A polymer, serafilcon A,"),
            ("The devices that reached peak efficiency aged. Their stability varied.", "The devices that reached peak efficiency"),
            ("Patients with COVID-19 at one hospital enrolled. Their outcomes differed.", "Patients with COVID-19 at one hospital"),
            ("Serafilcon A gradually depleted. Its reserve declined.", "Serafilcon A gradually"),
        ]
        for text, antecedent in cases:
            with self.subTest(antecedent=antecedent):
                pronoun = "Their" if "Their" in text else "Its"
                _, result = self.resolve_with(
                    text,
                    cluster(0, mention(text, antecedent), mention(text, pronoun)),
                )
                self.assertEqual(result.resolved, text)
                self.assertTrue(
                    any("unsafe_possessive_antecedent" in item for item in result.diagnostics)
                )

    def test_clean_possessive_antecedents_have_no_length_cap(self):
        antecedent = "individual members of the commensal community"
        text = f"The {antecedent} were cataloged. Their contributions matter."
        _, result = self.resolve_with(
            text,
            cluster(0, mention(text, antecedent), mention(text, "Their")),
        )
        self.assertEqual(
            result.resolved,
            f"The {antecedent} were cataloged. {antecedent}'s contributions matter.",
        )

    def test_plain_pronouns_ignore_possessive_safety_policy(self):
        antecedent = "114 patients with COVID-19 at the Jinyintan Hospital, Wuhan"
        text = f"We enrolled {antecedent}. They were categorized."
        _, result = self.resolve_with(
            text,
            cluster(0, mention(text, antecedent), mention(text, "They")),
        )
        self.assertEqual(
            result.resolved,
            f"We enrolled {antecedent}. {antecedent} were categorized.",
        )

    def test_pronoun_before_nominal_antecedent_is_not_rewritten(self):
        text = "Before he spoke, Daniel paused. He then answered."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "he"),
                mention(text, "Daniel"),
                mention(text, "He"),
            ),
        )
        self.assertEqual(
            result.resolved,
            "Before he spoke, Daniel paused. Daniel then answered.",
        )

    def test_non_pronominal_aliases_remain_unchanged(self):
        text = "Acme opened. The company expanded. It hired staff."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "Acme"),
                mention(text, "The company"),
                mention(text, "It"),
            ),
        )
        self.assertEqual(result.resolved, "Acme opened. The company expanded. Acme hired staff.")

    def test_trailing_antecedent_punctuation_is_not_copied(self):
        text = "A polymer, changed. It hardened."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                CoreferenceMention(0, 10, "A polymer,"),
                mention(text, "It"),
            ),
        )
        self.assertEqual(result.resolved, "A polymer, changed. A polymer hardened.")
        self.assertEqual(result.replacements[0].antecedent, "A polymer")

    def test_invalid_and_mismatched_spans_are_skipped(self):
        text = "Alice left. She waved."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "Alice"),
                CoreferenceMention(-1, 2, "xx"),
                CoreferenceMention(12, 15, "Not"),
                CoreferenceMention(500, 503, "She"),
            ),
        )
        self.assertEqual(result.resolved, text)
        self.assertEqual(len(result.diagnostics), 3)

    def test_duplicate_replacement_is_applied_once(self):
        text = "Alice left. She waved."
        she = mention(text, "She")
        _, result = self.resolve_with(
            text,
            cluster(0, mention(text, "Alice"), she),
            cluster(1, mention(text, "Alice"), she),
        )
        self.assertEqual(result.resolved, "Alice left. Alice waved.")
        self.assertEqual(len(result.replacements), 1)
        self.assertTrue(any(item.startswith("duplicate_span") for item in result.diagnostics))

    def test_right_to_left_replacement_preserves_offsets(self):
        text = "Alice arrived. She smiled because she won."
        _, result = self.resolve_with(
            text,
            cluster(
                0,
                mention(text, "Alice"),
                mention(text, "She"),
                mention(text, "she"),
            ),
        )
        self.assertEqual(result.resolved, "Alice arrived. Alice smiled because Alice won.")

    def test_scores_are_reported_but_not_filtered(self):
        text = "Alice arrived. She smiled."
        backend = FakeBackend(
            [cluster(0, mention(text, "Alice"), mention(text, "She", score=-10.0))],
            score_available=True,
        )
        result = CoreferenceResolver(
            enabled=True,
            min_confidence=0.99,
            backend=backend,
        ).resolve(text)
        self.assertEqual(result.resolved, "Alice arrived. Alice smiled.")
        self.assertEqual(result.replacements[0].score, -10.0)
        self.assertTrue(result.score_available)

    def test_fail_open_is_observable_and_cumulative(self):
        backend = FakeBackend(error=RuntimeError("model unavailable\nsecret detail"))
        resolver = CoreferenceResolver(enabled=True, backend=backend)
        result = resolver.resolve("Alice left. She waved.")

        self.assertEqual(result.status, "failed_open")
        self.assertEqual(result.resolved, result.original)
        self.assertEqual(result.error, "model unavailable secret detail")
        status = resolver.status()
        self.assertEqual(status.state, "degraded")
        self.assertEqual(status.failures, 1)
        self.assertEqual(status.documents_processed, 1)

    def test_fail_closed_propagates_backend_error(self):
        resolver = CoreferenceResolver(
            enabled=True,
            fail_open=False,
            backend=FakeBackend(error=RuntimeError("broken")),
        )
        with self.assertRaisesRegex(RuntimeError, "broken"):
            resolver.resolve("Alice left. She waved.")
        status = resolver.status()
        self.assertEqual(status.state, "degraded")
        self.assertEqual(status.failures, 1)
        self.assertEqual(status.documents_processed, 1)

    def test_rewriting_is_idempotent_for_model_output_without_new_spans(self):
        text = "Alice arrived. She smiled."
        _, first = self.resolve_with(
            text,
            cluster(0, mention(text, "Alice"), mention(text, "She")),
        )
        resolver = CoreferenceResolver(enabled=True, backend=FakeBackend())
        second = resolver.resolve(first.resolved)
        self.assertEqual(second.resolved, first.resolved)


if __name__ == "__main__":
    unittest.main()
