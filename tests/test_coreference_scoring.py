import json
import unittest
from pathlib import Path

from scripts.evaluate_coreference_benchmark import evaluate_report, score_row


LABEL_PATH = Path("data/benchmarks/stress7_coreference_labels_v1.json")


def _labels():
    payload = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    return payload, {case["case_id"]: case for case in payload["cases"]}


def _row(case, proof, grounded=True):
    return {
        "case_id": case["case_id"],
        "case_hash": case["case_hash"],
        "end_to_end": {
            "query": {
                "proof": proof,
                "proof_provenance": {"current_case_grounded": grounded},
            }
        },
    }


class CoreferenceProofScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels, cls.by_id = _labels()

    def test_relevant_grounded_proof_passes(self):
        row = _row(
            self.by_id["A09"],
            "(: bridge (EnhancesHeterointerfaceBonding sodium_heparin perovskite-solar-cells) "
            "(MitigatesDefects SnO2_perovskite_interface) (ImprovesEfficiency sodium_heparin PCE) "
            "(ImprovesStability sodium_heparin operational_stability))",
        )

        result = score_row(row, self.by_id["A09"])

        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_reasons"], [])
        self.assertEqual(result["coverage"]["entity"], 1.0)

    def test_generic_compared_proof_fails_a06(self):
        result = score_row(
            _row(self.by_id["A06"], "(: generic (Compared material other_material))"),
            self.by_id["A06"],
        )

        self.assertFalse(result["passed"])
        self.assertIn("insufficient_entity_groups", result["failure_reasons"])
        self.assertIn("insufficient_semantic_groups", result["failure_reasons"])

    def test_incomplete_biomarker_proof_fails_a10(self):
        result = score_row(
            _row(self.by_id["A10"], "(: marker (PredictsCOVID19Severity D-dimer))"),
            self.by_id["A10"],
        )

        self.assertFalse(result["passed"])
        self.assertIn("insufficient_entity_groups", result["failure_reasons"])
        self.assertIn("crp", result["missing_entity_groups"])

    def test_patient_191_foreign_only_proof_fails_a11(self):
        result = score_row(
            _row(
                self.by_id["A11"],
                "(: foreign (PredictCOVID19Severity patient_191 D-dimer CRP LDH ALT neutrophil_count))",
            ),
            self.by_id["A11"],
        )

        self.assertFalse(result["passed"])
        self.assertIn("forbidden_group_hit", result["failure_reasons"])
        self.assertEqual(
            {group["id"] for group in result["matched_forbidden_groups"]},
            {"foreign_patient_191", "foreign_a10_d_dimer", "foreign_a10_alt_sgpt"},
        )

    def test_foreign_semantics_are_not_credited_by_unrelated_current_proof(self):
        case = self.by_id["A01"]
        row = {
            "case_id": case["case_id"],
            "case_hash": case["case_hash"],
            "end_to_end": {
                "query": {
                    "proof": str([
                        "(: old (ImprovesGlycemicControl digital_diabetes_logbook))",
                        "(: current (Unrelated current_case_fact))",
                    ]),
                    "proof_provenance": [
                        {"current_case_grounded": False, "foreign_only": True},
                        {"current_case_grounded": True, "foreign_only": False},
                    ],
                }
            },
        }

        result = score_row(row, case)

        self.assertFalse(result["passed"])
        self.assertEqual(result["grounded_proof_count"], 1)
        self.assertIn("insufficient_entity_groups", result["failure_reasons"])

    def test_report_rejects_non_isolated_mode_and_hash_mismatch(self):
        case = self.by_id["A01"]
        report = {
            "mode": "cumulative",
            "suite": "stress25_v1",
            "parsers": {"parser": [_row(case, "proof")]},
        }
        with self.assertRaisesRegex(ValueError, "isolated"):
            evaluate_report(report, self.labels)

        report["mode"] = "isolated"
        report["parsers"]["parser"][0]["case_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate_report(report, self.labels)


if __name__ == "__main__":
    unittest.main()
