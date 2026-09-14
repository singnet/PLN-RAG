import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_coreference import (
    DEFAULT_DATASET,
    DatasetError,
    _make_resolver,
    evaluate,
    load_dataset,
    main,
)


class CoreferenceEvaluatorTests(unittest.TestCase):
    def test_dataset_is_synthetic_sized_and_all_spans_validate(self):
        dataset = load_dataset(DEFAULT_DATASET)

        self.assertGreaterEqual(len(dataset["cases"]), 30)
        self.assertLessEqual(len(dataset["cases"]), 40)
        self.assertTrue(dataset["redistribution"]["safe"])
        categories = {case["category"] for case in dataset["cases"]}
        self.assertTrue(
            {
                "subject",
                "object",
                "possessive",
                "ambiguous_her_abstention",
                "plural",
                "multiple_clusters",
                "cataphora",
                "negative",
                "science",
                "biomedical",
            }.issubset(categories)
        )

    def test_fixture_evaluates_deterministic_rewriter_without_models(self):
        artifact = evaluate(DEFAULT_DATASET, "fixture")

        self.assertEqual(artifact["schema"]["version"], "1.0")
        self.assertEqual(artifact["resolver"]["backend"], "fixture")
        self.assertEqual(artifact["resolver"]["invocations"], 36)
        self.assertEqual(artifact["summary"]["exact_accuracy"], 1.0)
        self.assertEqual(artifact["summary"]["change_accuracy"], 1.0)
        self.assertEqual(artifact["summary"]["replacement_f1"], 1.0)
        self.assertEqual(artifact["summary"]["failures"], 0)
        self.assertEqual(len(artifact["cases"]), 36)
        self.assertIn("timing", artifact["cases"][0])
        self.assertIn("replacements", artifact["cases"][0])

    def test_span_text_mismatch_is_rejected(self):
        source = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        source["cases"][0]["clusters"][0]["mentions"][0]["end"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "text mismatch"):
                load_dataset(path)

    def test_none_backend_require_effective_writes_artifact_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            exit_code = main(
                [
                    "--dataset",
                    str(DEFAULT_DATASET),
                    "--output",
                    str(output),
                    "--backend",
                    "none",
                    "--require-effective",
                ]
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 3)
        self.assertEqual(artifact["resolver"]["invocations"], 0)
        self.assertEqual(artifact["resolver"]["state"], "disabled")

    def test_require_effective_rejects_failed_open_run(self):
        artifact = {
            "resolver": {"invocations": 1},
            "summary": {
                "exact_accuracy": 0.0,
                "replacement_f1": 0.0,
                "failures": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-open.json"
            with patch("scripts.evaluate_coreference.evaluate", return_value=artifact):
                exit_code = main(["--output", str(output), "--require-effective"])

        self.assertEqual(exit_code, 3)

    def test_real_backend_is_configured_for_cpu(self):
        dataset = load_dataset(DEFAULT_DATASET)
        with patch("scripts.evaluate_coreference.CoreferenceResolver") as resolver_class:
            _make_resolver("lingmess", dataset, "example/model", 0.7)

        resolver_class.assert_called_once_with(
            enabled=True,
            backend_name="lingmess",
            model_name="example/model",
            min_confidence=0.7,
            device="cpu",
            fail_open=True,
        )


if __name__ == "__main__":
    unittest.main()
