import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dspy
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from core.lm import (
    CassetteError,
    cassette_status,
    create_lm,
    reset_cassette_state,
)


class CreateLMTests(unittest.TestCase):
    def setUp(self):
        reset_cassette_state()
        self.env = patch.dict(os.environ, {"LLM_CASSETTE_MODE": "off"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        reset_cassette_state()

    def test_uses_provider_default_endpoint_when_base_url_is_unset(self):
        settings = SimpleNamespace(
            openai_api_key="test-key",
            openai_model="openai/test-model",
            openai_base_url=None,
        )

        with (
            patch("core.lm.get_settings", return_value=settings),
            patch("core.lm.dspy.LM", return_value="lm") as lm_constructor,
        ):
            result = create_lm()

        self.assertEqual(result, "lm")
        lm_constructor.assert_called_once_with(
            "openai/test-model",
            api_key="test-key",
            cache=False,
        )

    def test_forwards_configured_base_url(self):
        settings = SimpleNamespace(
            openai_api_key="openrouter-key",
            openai_model="openrouter/openai/gpt-4o-mini",
            openai_base_url="https://openrouter.ai/api/v1",
        )

        with (
            patch("core.lm.get_settings", return_value=settings),
            patch("core.lm.dspy.LM", return_value="lm") as lm_constructor,
        ):
            result = create_lm()

        self.assertEqual(result, "lm")
        lm_constructor.assert_called_once_with(
            "openrouter/openai/gpt-4o-mini",
            api_key="openrouter-key",
            cache=False,
            base_url="https://openrouter.ai/api/v1",
        )

    def test_litellm_response_model_constructs_on_runtime_python(self):
        self.assertIsNotNone(ModelResponse())


class CassetteLMTests(unittest.TestCase):
    def setUp(self):
        reset_cassette_state()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cassette_path = Path(self.temp_dir.name) / "lm-cassette.json"
        self.settings = SimpleNamespace(
            openai_api_key="api-key-must-not-be-written",
            openai_model="openai/test-model",
            openai_base_url=None,
        )

    def tearDown(self):
        reset_cassette_state()
        self.temp_dir.cleanup()

    def _env(self, mode, scope="test-scope"):
        return patch.dict(
            os.environ,
            {
                "LLM_CASSETTE_MODE": mode,
                "LLM_CASSETTE_PATH": str(self.cassette_path),
                "LLM_CASSETTE_SCOPE": scope,
            },
        )

    @staticmethod
    def _response(content, response_id="response-id"):
        return ModelResponse(
            id=response_id,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    def test_capture_reuses_response_pool_after_reset_without_live_call(self):
        captured = self._response("captured")
        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(dspy.LM, "forward", return_value=captured) as provider,
        ):
            first = create_lm(purpose="parser").forward(prompt="private prompt")

        self.assertIs(first, captured)
        provider.assert_called_once()
        reset_cassette_state()

        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(
                dspy.LM, "forward", side_effect=AssertionError("provider called")
            ),
        ):
            replayed_from_pool = create_lm(purpose="parser").forward(
                prompt="private prompt"
            )
            status = cassette_status()

        self.assertIsInstance(replayed_from_pool, ModelResponse)
        self.assertEqual(replayed_from_pool.choices[0].message.content, "captured")
        self.assertEqual(status["hits"], 1)
        self.assertEqual(status["live_calls"], 0)
        self.assertEqual(status["misses"], 0)

    def test_multiple_lms_share_per_hash_occurrences_and_trace(self):
        responses = [self._response("first", "one"), self._response("second", "two")]
        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(dspy.LM, "forward", side_effect=responses) as provider,
        ):
            messages = [{"role": "user", "content": "same"}]
            create_lm(purpose="shared").forward(messages=messages)
            create_lm(purpose="shared").forward(messages=messages)

        self.assertEqual(provider.call_count, 2)
        cassette = json.loads(self.cassette_path.read_text(encoding="utf-8"))
        trace = cassette["scopes"]["test-scope"]
        self.assertEqual([entry["occurrence"] for entry in trace], [0, 1])
        self.assertEqual(len(cassette["responses"]), 1)

    def test_capture_canonicalizes_pydantic_model_class_forward_kwarg(self):
        class ResponseFormat(BaseModel):
            answer: str

        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(
                dspy.LM, "forward", return_value=self._response("answer")
            ) as provider,
        ):
            create_lm(purpose="parser").forward(
                prompt="question", response_format=ResponseFormat
            )

        provider.assert_called_once()

    def test_replay_reconstructs_model_response_and_never_calls_provider(self):
        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(dspy.LM, "forward", return_value=self._response("answer")),
        ):
            create_lm(purpose="answer").forward(prompt="question")
        reset_cassette_state()

        with (
            self._env("replay"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(
                dspy.LM, "forward", side_effect=AssertionError("provider called")
            ) as provider,
        ):
            result = create_lm(purpose="answer").forward(prompt="question")
            status = cassette_status()

        provider.assert_not_called()
        self.assertIsInstance(result, ModelResponse)
        self.assertEqual(result.choices[0].message.content, "answer")
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(status["calls"], 1)
        self.assertEqual(status["hits"], 1)
        self.assertTrue(status["replay_valid"])
        self.assertEqual(status["unconsumed"], 0)

    def test_replay_rejects_out_of_order_request_without_provider_fallback(self):
        with (
            self._env("capture"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(dspy.LM, "forward", return_value=self._response("answer")),
        ):
            create_lm(purpose="parser").forward(prompt="expected")
        reset_cassette_state()

        with (
            self._env("replay"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(
                dspy.LM, "forward", side_effect=AssertionError("provider called")
            ) as provider,
        ):
            with self.assertRaisesRegex(CassetteError, "request mismatch"):
                create_lm(purpose="parser").forward(prompt="different")
            status = cassette_status()

        provider.assert_not_called()
        self.assertFalse(status["replay_valid"])
        self.assertEqual(status["misses"], 1)
        self.assertEqual(status["unconsumed"], 1)

    def test_cassette_contains_hashes_responses_and_safe_metadata_only(self):
        prompt = "prompt-body-must-not-be-written"
        with (
            self._env("capture", scope="safe-scope"),
            patch("core.lm.get_settings", return_value=self.settings),
            patch.object(dspy.LM, "forward", return_value=self._response("safe output")),
        ):
            create_lm(purpose="safe-purpose").forward(
                messages=[{"role": "user", "content": prompt}]
            )

        raw_cassette = self.cassette_path.read_text(encoding="utf-8")
        self.assertNotIn(prompt, raw_cassette)
        self.assertNotIn(self.settings.openai_api_key, raw_cassette)
        self.assertIn("safe output", raw_cassette)
        cassette = json.loads(raw_cassette)
        entry = cassette["scopes"]["safe-scope"][0]
        self.assertRegex(entry["request_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(set(entry), {"request_hash", "occurrence", "purpose"})

    def test_non_off_mode_requires_path(self):
        env = {
            "LLM_CASSETTE_MODE": "replay",
            "LLM_CASSETTE_SCOPE": "scope",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("core.lm.get_settings", return_value=self.settings),
            self.assertRaisesRegex(ValueError, "LLM_CASSETTE_PATH"),
        ):
            create_lm(purpose="parser")


class ComponentLMOwnershipTests(unittest.TestCase):
    def test_nl2pln_parser_binds_its_lm_to_the_loaded_module(self):
        from parsers.nl2pln_parser import NL2PLNParser

        fake_module = self._fake_nl2pln_module()
        settings = SimpleNamespace(nl2pln_module_path="data/test.json")

        with (
            patch.dict(
                sys.modules,
                {
                    "nl2pln": fake_module,
                    "pettachainer": types.SimpleNamespace(get_language_spec=object()),
                },
            ),
            patch("parsers.nl2pln_parser.get_settings", return_value=settings),
            patch(
                "parsers.nl2pln_parser.create_lm", return_value="parser-lm"
            ) as create_lm_mock,
        ):
            parser = NL2PLNParser()

        self.assertEqual(parser._module.loaded_path, "data/test.json")
        self.assertEqual(parser._module.lm, "parser-lm")
        create_lm_mock.assert_called_once_with(purpose="nl2pln_parser")

    def test_canonical_parser_binds_its_lm_to_the_loaded_module(self):
        from parsers.canonical_pln_parser import CanonicalPLNParser

        fake_module = self._fake_nl2pln_module()
        settings = SimpleNamespace(
            canonical_pln_nl2pln_module_path="data/canonical-test.json"
        )

        with (
            patch.dict(sys.modules, {"nl2pln": fake_module}),
            patch("parsers.canonical_pln_parser.get_settings", return_value=settings),
            patch(
                "parsers.canonical_pln_parser.create_lm",
                return_value="canonical-lm",
            ) as create_lm_mock,
        ):
            parser = CanonicalPLNParser()

        self.assertEqual(parser._module.loaded_path, "data/canonical-test.json")
        self.assertEqual(parser._module.lm, "canonical-lm")
        create_lm_mock.assert_called_once_with(purpose="canonical_pln_parser")

    def test_answer_generator_binds_its_predictor_lm(self):
        from core.answer_generator import AnswerGenerator

        predictor = MagicMock()
        with (
            patch("core.answer_generator.dspy.Predict", return_value=predictor),
            patch(
                "core.answer_generator.create_lm", return_value="answer-lm"
            ) as create_lm_mock,
        ):
            generator = AnswerGenerator()

        self.assertIs(generator._predict, predictor)
        self.assertEqual(generator._predict.lm, "answer-lm")
        create_lm_mock.assert_called_once_with(purpose="answer_generator")

    @staticmethod
    def _fake_nl2pln_module():
        class FakeNL2PLNModule:
            def __init__(self):
                self.nl2pln = object()
                self.loaded_path = None
                self.lm = None

            def load(self, path):
                self.loaded_path = path

            def set_lm(self, lm):
                self.lm = lm

        return types.SimpleNamespace(
            NL2PLNModule=FakeNL2PLNModule,
            pln_spec="pln-spec",
        )


if __name__ == "__main__":
    unittest.main()
