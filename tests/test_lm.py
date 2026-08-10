import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.lm import create_lm


class CreateLMTests(unittest.TestCase):
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
            patch("parsers.nl2pln_parser.create_lm", return_value="parser-lm"),
        ):
            parser = NL2PLNParser()

        self.assertEqual(parser._module.loaded_path, "data/test.json")
        self.assertEqual(parser._module.lm, "parser-lm")

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
            ),
        ):
            parser = CanonicalPLNParser()

        self.assertEqual(parser._module.loaded_path, "data/canonical-test.json")
        self.assertEqual(parser._module.lm, "canonical-lm")

    def test_answer_generator_binds_its_predictor_lm(self):
        from core.answer_generator import AnswerGenerator

        predictor = MagicMock()
        with (
            patch("core.answer_generator.dspy.Predict", return_value=predictor),
            patch("core.answer_generator.create_lm", return_value="answer-lm"),
        ):
            generator = AnswerGenerator()

        self.assertIs(generator._predict, predictor)
        self.assertEqual(generator._predict.lm, "answer-lm")

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
