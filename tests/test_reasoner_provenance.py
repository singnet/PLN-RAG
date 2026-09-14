import unittest

from core.reasoner import AtomProvenance, Reasoner


class ReasonerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.reasoner = Reasoner.__new__(Reasoner)
        self.reasoner._provenance_by_name = {}
        self.reasoner._statement_hashes_by_name = {}

    def register(self, name, provenance):
        statement = f"(: {name} (Example value) (STV 1.0 1.0))"
        self.reasoner._register_provenance(statement, provenance)

    def test_derived_proof_reports_current_and_foreign_sources(self):
        self.register("current_fact", AtomProvenance("document", "A11", 0))
        self.register("foreign_rule", AtomProvenance("document", "A10", 0))

        result = self.reasoner._proof_provenance(
            "(: (foreign_rule current_fact) (Severity patient) (STV 1 1))",
            "A11",
        )

        self.assertTrue(result["current_case_grounded"])
        self.assertFalse(result["foreign_only"])
        self.assertEqual(result["foreign_case_ids"], ["A10"])
        self.assertEqual(result["atom_names"], ["current_fact", "foreign_rule"])

    def test_foreign_only_and_query_support_only_are_distinct(self):
        self.register("old_fact", AtomProvenance("document", "A10", 0))
        self.register("query_fact", AtomProvenance("query_support", "A11"))

        foreign = self.reasoner._proof_provenance("(old_fact)", "A11")
        support = self.reasoner._proof_provenance("(query_fact)", "A11")

        self.assertTrue(foreign["foreign_only"])
        self.assertFalse(foreign["current_case_grounded"])
        self.assertTrue(support["query_support_only"])
        self.assertFalse(support["foreign_only"])

    def test_unregistered_proof_is_marked_unknown(self):
        result = self.reasoner._proof_provenance("(: derived (Unknown x) (STV 1 1))", "A01")

        self.assertTrue(result["unknown"])
        self.assertFalse(result["current_case_grounded"])

    def test_colliding_atom_name_is_not_accepted_as_current_case_grounding(self):
        self.register("shared_fact", AtomProvenance("document", "A10", 0))
        self.register("shared_fact", AtomProvenance("document", "A11", 0))

        result = self.reasoner._proof_provenance("(shared_fact)", "A11")

        self.assertFalse(result["current_case_grounded"])
        self.assertTrue(result["foreign_only"])
        self.assertEqual(result["ambiguous_atom_names"], ["shared_fact"])

    def test_different_definitions_with_same_name_are_ambiguous(self):
        self.reasoner._register_provenance(
            "(: shared (First value) (STV 1 1))",
            AtomProvenance("document", "A11", 0),
        )
        self.reasoner._register_provenance(
            "(: shared (Second value) (STV 1 1))",
            AtomProvenance("query_support", "A11"),
        )

        result = self.reasoner._proof_provenance("(shared)", "A11")

        self.assertFalse(result["current_case_grounded"])
        self.assertEqual(result["ambiguous_atom_names"], ["shared"])


if __name__ == "__main__":
    unittest.main()
