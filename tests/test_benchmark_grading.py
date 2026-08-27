"""Tests for gold-answer grading.

Every proof shape pinned here was taken from a real artifact under `data/benchmarks/`.
"""

import json
from pathlib import Path

import pytest

import benchmark_grading as bg

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

TRACE_RULE = (
    "(: (significant_factor_rule (background_covid_study x)) "
    "(SignificantRiskFactor patient1 d_dimer) (STV 1.0 0.9996001442979974))"
)
TRACE_NOT = (
    "(: (negated (negated (iot_approach_not_reducing_hba1c materialized_utilize_fact))) "
    "(Not (ReducesHbA1c group)) (STV 1.0 0.9998000398670517))"
)
TRACE_BARE_TERM = (
    "(: materialized_may_achieve_fact "
    "(MayAchieve optimal_glycemic_control in_intervention_group) (STV 1.0 1.0))"
)
TRACE_ZERO_STRENGTH = "(: (sun_brightness_rule cpu) (AppearsBrighter sun) (STV 0.0 1.0e-06))"
TRACE_LOOKUP = "(: june_summer (IsInSeason june summer) (STV 1.0 1.0))"


def case(proof, *, key="proof", proof_found=True, case_id="X01"):
    return {
        "case": {"case_id": case_id},
        "proof_found": proof_found,
        "end_to_end": {"query": {key: proof}},
    }


class TestSplitTopLevel:
    def test_bare_atom_then_groups(self):
        assert bg.split_top_level("name (A b) (STV 1.0 1.0)") == [
            "name",
            "(A b)",
            "(STV 1.0 1.0)",
        ]

    def test_nested_groups_are_not_split(self):
        assert bg.split_top_level("(A (B c)) (D e)") == ["(A (B c))", "(D e)"]

    def test_group_is_not_emitted_twice(self):
        assert bg.split_top_level("(A b) (C d)") == ["(A b)", "(C d)"]


class TestParseTrace:
    def test_rule_proof(self):
        conclusion = bg.parse_trace(TRACE_RULE)
        assert conclusion.atom == "(SignificantRiskFactor patient1 d_dimer)"
        assert conclusion.negated is False
        assert conclusion.strength == 1.0
        assert "d_dimer" in conclusion.symbols

    def test_bare_proof_term_does_not_yield_the_truth_value(self):
        conclusion = bg.parse_trace(TRACE_BARE_TERM)
        assert conclusion.atom.startswith("(MayAchieve")
        assert "optimal_glycemic_control" in conclusion.symbols

    def test_not_wrapped_conclusion_is_unwrapped_and_flagged(self):
        conclusion = bg.parse_trace(TRACE_NOT)
        assert conclusion.negated is True
        assert conclusion.atom == "(ReducesHbA1c group)"

    def test_zero_strength_is_not_positive(self):
        conclusion = bg.parse_trace(TRACE_ZERO_STRENGTH)
        assert conclusion.strength == 0.0
        assert conclusion.is_positive is False

    def test_variables_are_not_symbols(self):
        conclusion = bg.parse_trace("(: r (AppearsBrighter sun $_190274) (STV 1.0 1.0))")
        assert conclusion.symbols == ["appear_brighter", "sun"]

    def test_predicate_head_is_a_symbol(self):
        conclusion = bg.parse_trace("(: r (Closer sun other_star) (STV 1.0 1.0))")
        assert "closer" in conclusion.symbols

    def test_non_trace_returns_none(self):
        assert bg.parse_trace("(Eats kebede fish)") is None


class TestExtractProofTraces:
    def test_python_repr_list_is_decoded(self):
        assert bg.extract_proof_traces(case(repr([TRACE_LOOKUP]))) == [TRACE_LOOKUP]

    def test_empty_list_string(self):
        assert bg.extract_proof_traces(case("[]", proof_found=False)) == []

    def test_raw_proof_fallback_for_older_artifacts(self):
        result = case(repr([TRACE_LOOKUP]), key="raw_proof")
        assert bg.extract_proof_traces(result) == [TRACE_LOOKUP]

    def test_unknown_schema_with_a_proof_raises(self):
        with pytest.raises(bg.ArtifactSchemaError):
            bg.extract_proof_traces(case(None, key="mystery", proof_found=True))

    def test_unknown_schema_without_a_proof_is_silent(self):
        assert bg.extract_proof_traces(case(None, key="mystery", proof_found=False)) == []


class TestGradeCase:
    def test_matched_entity(self):
        gold = {"entities": [["june"]], "match": "all", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_LOOKUP])))
        assert result["answer_correct"] is True
        assert result["answer_score"] == 1.0

    def test_no_proof_is_incorrect(self):
        gold = {"entities": [["june"]], "match": "all", "verdict": None}
        result = bg.grade_case(gold, case("[]", proof_found=False))
        assert result["answer_correct"] is False
        assert result["answer_reason"] == "no proof"

    def test_partial_coverage_scores_between_zero_and_one(self):
        gold = {
            "entities": [["d_dimer"], ["crp"], ["ldh"], ["blood_cell_count"]],
            "match": "any",
            "verdict": None,
        }
        result = bg.grade_case(gold, case(repr([TRACE_RULE])))
        assert result["answer_correct"] is True
        assert result["answer_score"] == 0.25

    def test_match_all_rejects_partial_coverage(self):
        gold = {"entities": [["d_dimer"], ["crp"]], "match": "all", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_RULE])))
        assert result["answer_correct"] is False
        assert result["answer_score"] == 0.5

    def test_substring_does_not_earn_credit(self):
        gold = {"entities": [["contact_angle"]], "match": "any", "verdict": None}
        trace = "(: r (DistinguishesSurfaceHydrophilicityHydrophobicity angle) (STV 1.0 1.0))"
        result = bg.grade_case(gold, case(repr([trace])))
        assert result["answer_correct"] is False
        assert result["answer_score"] == 0.0

    def test_alias_earns_credit(self):
        gold = {"entities": [["crp", "c_reactive_protein"]], "match": "any", "verdict": None}
        trace = "(: r (Marker c_reactive_protein) (STV 1.0 1.0))"
        assert bg.grade_case(gold, case(repr([trace])))["answer_correct"] is True

    def test_zero_strength_symbols_do_not_satisfy_a_slot(self):
        gold = {"entities": [["sun"]], "match": "any", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_ZERO_STRENGTH])))
        assert result["answer_correct"] is False
        assert "strength 0.0" in result["answer_reason"]

    def test_topic_only_answer_fails_a_reason_gold(self):
        """The discrimination the grader exists for: restating the question is not an answer."""
        gold = {"entities": [["closer"], ["distance"]], "match": "any", "verdict": None}
        trace = "(: r (AppearsBrighter sun) (STV 1.0 1.0))"
        result = bg.grade_case(gold, case(repr([trace])))
        assert result["answer_correct"] is False

    def test_multiple_traces_union_their_entities(self):
        gold = {"entities": [["june"], ["d_dimer"]], "match": "all", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_LOOKUP, TRACE_RULE])))
        assert result["answer_correct"] is True
        assert result["answer_score"] == 1.0


class TestVerdicts:
    def test_negative_verdict_met_by_negated_conclusion(self):
        # Parsers fuse the measure into the head, so `(Not (ReducesHbA1c group))` yields
        # `reduce_hb_a1c` with no standalone `hb_a1c` for gold to match.
        gold = {
            "entities": [["hb_a1c", "hba1c", "reduce_hb_a1c"]],
            "match": "any",
            "verdict": "negative",
        }
        result = bg.grade_case(gold, case(repr([TRACE_NOT])))
        assert result["answer_correct"] is True
        assert result["verdict_gradable"] is True

    def test_positive_verdict_unmet_by_negated_conclusion(self):
        gold = {"entities": [["reduce_hb_a1c"]], "match": "any", "verdict": "positive"}
        result = bg.grade_case(gold, case(repr([TRACE_NOT])))
        assert result["answer_correct"] is False
        assert "verdict positive: unmet" in result["answer_reason"]

    def test_negated_conclusion_does_not_answer_a_wh_question(self):
        gold = {"entities": [["reduce_hb_a1c"]], "match": "any", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_NOT])))
        assert result["answer_correct"] is False

    def test_null_verdict_is_not_graded(self):
        gold = {"entities": [["june"]], "match": "any", "verdict": None}
        result = bg.grade_case(gold, case(repr([TRACE_LOOKUP])))
        assert result["verdict_gradable"] is False
        assert "verdict" not in result["answer_reason"]


class TestGradeResultsAndSummary:
    def test_missing_gold_entry_is_ungraded_not_wrong(self):
        rows = bg.grade_results({}, [case(repr([TRACE_LOOKUP]), case_id="Z99")])
        assert rows[0]["answer_correct"] is None
        assert rows[0]["answer_reason"] == "no gold entry"

    def test_summary_excludes_ungraded_cases_from_the_denominator(self):
        gold = {"E02": {"entities": [["june"]], "match": "all", "verdict": None}}
        rows = bg.grade_results(
            gold,
            [case(repr([TRACE_LOOKUP]), case_id="E02"), case("[]", proof_found=False, case_id="Z99")],
        )
        stats = bg.summarize(rows)
        assert stats["cases"] == 2
        assert stats["answer_graded"] == 1
        assert stats["answer_correct"] == 1

    def test_grading_does_not_mutate_the_input(self):
        gold = {"E02": {"entities": [["june"]], "match": "all", "verdict": None}}
        result = case(repr([TRACE_LOOKUP]), case_id="E02")
        before = repr(result)
        bg.grade_results(gold, [result])
        assert repr(result) == before


class TestStress25Gold:
    """Gold lives in a separate file from the suite; a drifted case id grades as "no gold
    entry" and silently shrinks the denominator rather than failing."""

    @staticmethod
    def _load(name):
        return json.loads((BENCHMARK_DIR / name).read_text(encoding="utf-8"))

    def test_gold_covers_exactly_the_suite(self):
        suite = self._load("stress25_v1.json")
        cases = suite["cases"] if isinstance(suite, dict) else suite
        suite_ids = {c.get("case_id") or c.get("id") or c.get("name") for c in cases}
        gold_ids = set(self._load("stress25_v1_gold.json")["cases"])
        assert gold_ids == suite_ids

    def test_every_entry_is_gradable_and_supported(self):
        gold = self._load("stress25_v1_gold.json")["cases"]
        for case_id, entry in gold.items():
            assert entry["match"] in (bg.MATCH_ANY, bg.MATCH_ALL), case_id
            assert entry["verdict"] in (None, *bg.VERDICTS), case_id
            assert entry["entities"], case_id
            assert all(slot for slot in entry["entities"]), case_id
            # No support quote means the gold entity was invented, not licensed by the corpus.
            assert entry["support"].strip(), case_id

    def test_slots_canonicalize_without_collapsing(self):
        gold = self._load("stress25_v1_gold.json")["cases"]
        for case_id, entry in gold.items():
            for slot in bg._canonical_slots(entry["entities"]):
                assert "" not in slot, case_id
                assert slot, case_id
