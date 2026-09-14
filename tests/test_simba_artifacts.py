import hashlib
import importlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT / "data" / "simba_canonical_pln.json",
    ROOT / "data" / "simba_all.json",
    ROOT / "simba_all.json",
)
FORBIDDEN_CHEMISTRY = (
    "methane",
    "oxygen",
    "combustion",
    "carbon dioxide",
    "chemical reaction",
    "reactswith",
    "releasesheat",
)
DEMO_VARIANTS = ("neutral", "final-removed")
PROMPT_VARIANTS = ("current", "hardened")


def parse_expression(text):
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()

    def parse_at(index):
        if tokens[index] != "(":
            return tokens[index], index + 1
        result = []
        index += 1
        while tokens[index] != ")":
            value, index = parse_at(index)
            result.append(value)
        return result, index + 1

    expression, end = parse_at(0)
    if end != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}")
    return expression


def semantic_constants(text):
    expression = parse_expression(text)
    constants = set()

    def visit(node):
        if not isinstance(node, list) or not node:
            return
        for argument in node[1:]:
            if isinstance(argument, list):
                visit(argument)
            elif (
                not argument.startswith("$")
                and argument != "->"
                and not argument.replace(".", "", 1).isdigit()
            ):
                constants.add(argument)

    # Skip the serializer's proof identifier and truth-value wrapper.
    visit(expression[2])
    return constants


class SimbaArtifactTests(unittest.TestCase):
    def test_safe_demo_output_constants_are_grounded_in_source_sentences(self):
        fixture = json.loads(
            (ROOT / "scripts" / "simba_synthetic_demos.json").read_text(
                encoding="utf-8"
            )
        )
        for index, demo in enumerate(fixture["safe_demos"]):
            source_tokens = set(
                re.findall(r"[A-Za-z0-9]+", " ".join(demo["sentences"]).lower())
            )
            emitted = {
                constant
                for output in (*demo["statements"], *demo["queries"])
                for constant in semantic_constants(output)
            }
            ungrounded = [
                constant
                for constant in emitted
                if not all(part.lower() in source_tokens for part in constant.split("_"))
            ]
            with self.subTest(demo=index):
                self.assertEqual([], sorted(ungrounded))

    def test_safe_demos_exclude_known_unsupported_semantic_shortcuts(self):
        content = (ROOT / "scripts" / "simba_synthetic_demos.json").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "I will create a rule that transfers the property",
            "Temperature is a measure of heat energy.",
            "(IsA $child child)",
            "A farmer plants corn in a field every year.",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, content)

    def test_artifacts_restore_safe_demos_with_expected_counts(self):
        fixture = json.loads(
            (ROOT / "scripts" / "simba_synthetic_demos.json").read_text(
                encoding="utf-8"
            )
        )
        expected_safe = [
            {
                "augmented": demo["augmented"],
                "sentences": demo["sentences"],
                "context": demo["context"],
                "pln_spec": fixture["pln_spec"],
                "reasoning": demo["reasoning"],
                "statements": demo["statements"],
                "queries": demo["queries"],
            }
            for demo in fixture["safe_demos"]
        ]
        canonical = json.loads(ARTIFACTS[0].read_text(encoding="utf-8"))[
            "nl2pln.predict"
        ]["demos"]
        data_baseline = json.loads(ARTIFACTS[1].read_text(encoding="utf-8"))[
            "nl2pln.predict"
        ]["demos"]
        root_baseline = json.loads(ARTIFACTS[2].read_text(encoding="utf-8"))[
            "nl2pln.predict"
        ]["demos"]

        self.assertEqual(12, len(canonical))
        self.assertEqual(7, len(data_baseline))
        self.assertEqual(7, len(root_baseline))
        self.assertEqual(expected_safe, canonical[:11])
        self.assertEqual(expected_safe[:6], data_baseline[:6])
        self.assertEqual(canonical[-1], data_baseline[-1])
        self.assertEqual(data_baseline, root_baseline)

    def test_artifacts_contain_no_forbidden_concrete_chemistry(self):
        for path in ARTIFACTS:
            content = path.read_text(encoding="utf-8").lower()
            for term in FORBIDDEN_CHEMISTRY:
                with self.subTest(path=path, term=term):
                    self.assertNotIn(term, content)

    def test_neutral_demo_output_constants_are_grounded_in_demo_input(self):
        for path in ARTIFACTS:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            demo = artifact["nl2pln.predict"]["demos"][-1]
            input_text = "\n".join(
                [*demo["sentences"], *demo["context"], demo["pln_spec"]]
            )
            input_symbols = {
                symbol.lower()
                for symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", input_text)
            }
            emitted = set()
            for output in (*demo["statements"], *demo["queries"]):
                emitted.update(semantic_constants(output))
            with self.subTest(path=path):
                self.assertEqual(
                    [],
                    sorted(
                        constant
                        for constant in emitted
                        if constant.lower() not in input_symbols
                    ),
                )

    def test_final_demo_has_relation_classification_output_shape(self):
        expected_expressions = [
            ["RelK", "token_y", "token_z"],
            [
                "Implication",
                ["Premises", ["RelK", "$item", "$other"]],
                ["Conclusions", ["ClassA", "$item"]],
            ],
            [
                "Implication",
                ["Premises", ["ClassA", "$item"]],
                ["Conclusions", ["OutputA", "$item"]],
            ],
        ]
        expected_query = ["OutputA", "token_y"]

        for path in ARTIFACTS:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            demo = artifact["nl2pln.predict"]["demos"][-1]
            expressions = [parse_expression(statement)[2] for statement in demo["statements"]]
            query = parse_expression(demo["queries"][0])[2]
            with self.subTest(path=path):
                self.assertEqual(expected_expressions, expressions)
                self.assertEqual(expected_query, query)

    def test_baseline_artifacts_are_byte_identical(self):
        data_digest = hashlib.sha256(ARTIFACTS[1].read_bytes()).digest()
        root_digest = hashlib.sha256(ARTIFACTS[2].read_bytes()).digest()
        self.assertEqual(data_digest, root_digest)

    def test_builder_rebuilds_missing_outputs_and_detects_signature_tampering(self):
        script = ROOT / "scripts" / "build_simba_artifacts.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            missing_check = subprocess.run(
                [sys.executable, str(script), "--check", "--output-dir", first],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, missing_check.returncode)
            self.assertIn("missing or stale artifacts", missing_check.stderr)

            subprocess.run(
                [sys.executable, str(script), "--output-dir", first], check=True
            )
            subprocess.run(
                [sys.executable, str(script), "--check", "--output-dir", first],
                check=True,
            )

            canonical = Path(first) / "data" / "simba_canonical_pln.json"
            tampered = json.loads(canonical.read_text(encoding="utf-8"))
            tampered["nl2pln.predict"]["signature"]["instructions"] = "tampered"
            canonical.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_check = subprocess.run(
                [sys.executable, str(script), "--check", "--output-dir", first],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, tampered_check.returncode)
            self.assertIn("data/simba_canonical_pln.json", tampered_check.stderr)

            subprocess.run(
                [sys.executable, str(script), "--output-dir", first], check=True
            )
            subprocess.run(
                [sys.executable, str(script), "--output-dir", second], check=True
            )
            for relative in (
                Path("data/simba_canonical_pln.json"),
                Path("data/simba_all.json"),
                Path("simba_all.json"),
            ):
                self.assertEqual(
                    (Path(first) / relative).read_bytes(),
                    (Path(second) / relative).read_bytes(),
                )

    def test_builder_generates_controlled_demo_and_prompt_matrix(self):
        script = ROOT / "scripts" / "build_simba_artifacts.py"
        fixture = json.loads(
            (ROOT / "scripts" / "simba_synthetic_demos.json").read_text(
                encoding="utf-8"
            )
        )

        def expand(demo):
            return {
                "augmented": demo["augmented"],
                "sentences": demo["sentences"],
                "context": demo["context"],
                "pln_spec": fixture["pln_spec"],
                "reasoning": demo["reasoning"],
                "statements": demo["statements"],
                "queries": demo["queries"],
            }

        safe_demos = [expand(demo) for demo in fixture["safe_demos"]]
        neutral_demo = expand(fixture["neutral_demo"])
        expected_demos = {
            "neutral": ([*safe_demos, neutral_demo], [*safe_demos[:6], neutral_demo]),
            "final-removed": (safe_demos, safe_demos[:6]),
        }
        contract = (ROOT / "scripts" / "simba_grounding_contract.txt").read_text(
            encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as directory:
            builds = {}
            for demo_variant in DEMO_VARIANTS:
                for prompt_variant in PROMPT_VARIANTS:
                    output = Path(directory) / f"{demo_variant}-{prompt_variant}"
                    command = [
                        sys.executable,
                        str(script),
                        "--demo-variant",
                        demo_variant,
                        "--prompt-variant",
                        prompt_variant,
                        "--output-dir",
                        str(output),
                    ]
                    subprocess.run(command, check=True)
                    subprocess.run([*command, "--check"], check=True)

                    canonical = json.loads(
                        (output / "data" / "simba_canonical_pln.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    data_baseline = output / "data" / "simba_all.json"
                    root_baseline = output / "simba_all.json"
                    self.assertEqual(
                        data_baseline.read_bytes(), root_baseline.read_bytes()
                    )
                    baseline = json.loads(data_baseline.read_text(encoding="utf-8"))
                    builds[(demo_variant, prompt_variant)] = (canonical, baseline)

                    expected_canonical, expected_baseline = expected_demos[demo_variant]
                    self.assertEqual(
                        expected_canonical, canonical["nl2pln.predict"]["demos"]
                    )
                    self.assertEqual(
                        expected_baseline, baseline["nl2pln.predict"]["demos"]
                    )

            for demo_variant in DEMO_VARIANTS:
                current = builds[(demo_variant, "current")]
                hardened = builds[(demo_variant, "hardened")]
                for current_artifact, hardened_artifact in zip(current, hardened):
                    current_predictor = current_artifact["nl2pln.predict"]
                    hardened_predictor = hardened_artifact["nl2pln.predict"]
                    current_instructions = current_predictor["signature"]["instructions"]
                    hardened_instructions = hardened_predictor["signature"]["instructions"]
                    self.assertEqual(
                        current_instructions + contract, hardened_instructions
                    )
                    self.assertEqual(
                        current_predictor["demos"], hardened_predictor["demos"]
                    )

            for prompt_variant in PROMPT_VARIANTS:
                neutral = builds[("neutral", prompt_variant)]
                final_removed = builds[("final-removed", prompt_variant)]
                for neutral_artifact, removed_artifact in zip(neutral, final_removed):
                    self.assertEqual(
                        neutral_artifact["nl2pln.predict"]["signature"],
                        removed_artifact["nl2pln.predict"]["signature"],
                    )

    def test_artifacts_load_with_nl2pln_when_available(self):
        try:
            nl2pln = importlib.import_module("nl2pln")
        except ModuleNotFoundError as error:
            if error.name == "nl2pln":
                self.skipTest("nl2pln is not installed in this environment")
            raise

        for path in ARTIFACTS:
            module = nl2pln.NL2PLNModule()
            with self.subTest(path=path):
                module.load(str(path))


if __name__ == "__main__":
    unittest.main()
