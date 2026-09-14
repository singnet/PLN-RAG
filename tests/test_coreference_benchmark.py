import argparse
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_parsers
from scripts.compare_coreference_benchmarks import compare_reports
from scripts.run_paired_benchmark import _benchmark_command, _output_from_stdout


def _coref(status="resolved", changed=True, duration=0.25):
    return SimpleNamespace(
        backend="fcoref",
        model="model",
        status=status,
        changed=changed,
        replacement_count=2 if changed else 0,
        duration_seconds=duration,
        score_available=True,
        error=None,
        diagnostics=["diagnostic"],
        resolved_text="Alice said Alice won.",
    )


def _ingest(status="success", coreference=None):
    return SimpleNamespace(
        text="Alice said she won.",
        status=status,
        atoms=["atom"],
        error=None if status == "success" else "parser failed",
        chunk_count=2,
        batch_count=1,
        batch_sizes=[2],
        parser_calls=1,
        rejected_count=0,
        rejected_samples=[],
        coreference=coreference,
    )


def _row(case_id, case_hash, proof, query="query", total=1.0):
    return {
        "case_id": case_id,
        "case_hash": case_hash,
        "case": {"name": case_id},
        "proof_found": proof,
        "timing": {"total_seconds": total},
        "end_to_end": {
            "ingest": [],
            "query": {"original_query": query, "executed_query": query, "pln_query": query},
        },
    }


def _report(backend, rows, *, valid=True, mode="cumulative"):
    return {
        "valid": valid,
        "pair_id": "pair",
        "coref": {"backend": backend},
        "config": {"mode": mode},
        "suite_metadata": {"suite_id": "suite"},
        "git": {"commit": "abc", "dirty": True},
        "parsers": {"parser": rows},
    }


class BenchmarkHelpersTest(unittest.TestCase):
    def test_coreference_backend_sets_enabled_backend_and_clears_cache(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            benchmark_parsers, "_clear_settings_cache"
        ) as clear_cache:
            benchmark_parsers._configure_coreference(None, "lingmess")

            self.assertEqual(os.environ["COREFERENCE_ENABLED"], "true")
            self.assertEqual(os.environ["COREFERENCE_BACKEND"], "lingmess")
            clear_cache.assert_called_once_with()

    def test_conflicting_coreference_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            benchmark_parsers._configure_coreference(False, "fcoref")

    def test_compact_ingest_retains_diagnostics_and_internal_resolved_text(self):
        result = benchmark_parsers._compact_ingest_results([_ingest(coreference=_coref())])[0]

        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["batch_sizes"], [2])
        self.assertEqual(result["coreference"]["resolved_text"], "Alice said Alice won.")
        self.assertEqual(result["coreference"]["replacement_count"], 2)

    def test_summary_does_not_turn_missing_latency_into_zero(self):
        result = {
            "error": "boom",
            "timing": {"total_seconds": None},
            "end_to_end": {"ingest": [], "query": {"query_status": "error"}},
        }

        summary = benchmark_parsers._summarize_parser([result])

        self.assertEqual(summary["execution_errors"], 1)
        self.assertEqual(summary["latency_samples"], 0)
        self.assertIsNone(summary["avg_latency_seconds"])

    def test_coreference_summary_and_validity(self):
        compact = benchmark_parsers._compact_ingest_results(
            [_ingest(coreference=_coref()), _ingest(coreference=_coref("failed_open", False, 0.5))]
        )
        summary = benchmark_parsers._summarize_parser(
            [{"timing": {"total_seconds": 1.0}, "end_to_end": {"ingest": compact, "query": {}}}]
        )

        self.assertEqual(summary["coreference"]["processed"], 2)
        self.assertEqual(summary["coreference"]["successful_invocations"], 1)
        self.assertEqual(summary["coreference"]["failures"], 1)
        self.assertIn(
            "coreference_failed_open",
            benchmark_parsers._invalid_reasons(1, {"parser": summary}, True),
        )

    def test_zero_cases_and_no_coreference_success_are_invalid(self):
        summary = {"parser": {"coreference": {"successful_invocations": 0}}}
        self.assertEqual(
            benchmark_parsers._invalid_reasons(0, summary, True),
            ["zero_cases", "coreference_no_successful_invocations"],
        )


class PairedSupportTest(unittest.TestCase):
    def test_runner_command_changes_only_backend_for_same_args(self):
        args = argparse.Namespace(
            mode="cumulative",
            parsers=["nl2pln"],
            output_dir="out",
            suite_file="suite.json",
            suite="combined",
            quick=True,
            progress=False,
            limit=2,
            case_ids=["A01"],
        )
        none = _benchmark_command(args, "none", "pair")
        fcoref = _benchmark_command(args, "fcoref", "pair")
        backend_index = none.index("--coreference-backend") + 1

        self.assertEqual(none[:backend_index], fcoref[:backend_index])
        self.assertEqual(none[backend_index + 1 :], fcoref[backend_index + 1 :])
        self.assertEqual(_output_from_stdout('{"output": "report.json"}\nsummary'), "report.json")
        self.assertEqual(
            _output_from_stdout('runtime noise\n[()]\n{"output": "report.json"}\nsummary'),
            "report.json",
        )

    def test_runner_passes_backend_specific_cassette_scope(self):
        args = argparse.Namespace(
            mode="isolated",
            parsers=["canonical_pln"],
            output_dir="out",
            suite_file="suite.json",
            suite="combined",
            quick=False,
            progress=False,
            limit=0,
            case_ids=[],
            llm_cassette_mode="capture",
            llm_cassette_path="cassette.json",
        )

        command = _benchmark_command(args, "lingmess", "pair")

        self.assertIn("--llm-cassette-path", command)
        scope_index = command.index("--llm-cassette-scope") + 1
        self.assertEqual(command[scope_index], "lingmess")

    def test_comparison_reports_transitions_query_changes_and_deltas(self):
        left = _report("none", [_row("A", "hash", False, total=1.0)])
        right = _report("fcoref", [_row("A", "hash", True, query="changed", total=1.5)])

        result = compare_reports(left, right)

        self.assertEqual(result["proof_transitions"]["false_to_true"], 1)
        self.assertEqual(result["query_changes"], 1)
        self.assertEqual(result["timing_deltas_seconds"]["total_seconds"]["mean_delta"], 0.5)

    def test_comparison_rejects_hash_mismatch(self):
        left = _report("none", [_row("A", "one", False)])
        right = _report("fcoref", [_row("A", "two", False)])

        with self.assertRaisesRegex(ValueError, "case hash"):
            compare_reports(left, right)

    def test_comparison_rejects_invalid_or_unpaired_reports(self):
        with self.assertRaisesRegex(ValueError, "must be valid"):
            compare_reports(
                _report("none", [_row("A", "hash", False)], valid=False),
                _report("fcoref", [_row("A", "hash", False)]),
            )
        right = _report("fcoref", [_row("A", "hash", False)])
        right["pair_id"] = "other"
        with self.assertRaisesRegex(ValueError, "pair ID"):
            compare_reports(_report("none", [_row("A", "hash", False)]), right)

    def test_disabled_coreference_contributes_zero_overhead(self):
        left_row = _row("A", "hash", False)
        right_row = _row("A", "hash", False)
        left_row["end_to_end"]["ingest"] = [
            {"coreference": {"status": "disabled", "duration_seconds": 0.0}}
        ]
        right_row["end_to_end"]["ingest"] = [
            {"coreference": {"status": "resolved", "duration_seconds": 0.25}}
        ]

        result = compare_reports(
            _report("none", [left_row]), _report("fcoref", [right_row])
        )

        self.assertEqual(
            result["timing_deltas_seconds"]["coreference_seconds"]["mean_delta"],
            0.25,
        )

    def test_isolated_query_change_without_rewrite_is_flagged(self):
        left_row = _row("A", "hash", True, query="before")
        right_row = _row("A", "hash", True, query="after")
        right_row["end_to_end"]["ingest"] = [
            {
                "coreference": {
                    "status": "unchanged",
                    "changed": False,
                    "replacement_count": 0,
                    "duration_seconds": 0.25,
                }
            }
        ]

        result = compare_reports(
            _report("none", [left_row], mode="isolated"),
            _report("lingmess", [right_row], mode="isolated"),
        )

        self.assertFalse(result["causal_interpretation_valid"])
        self.assertEqual(result["query_changes_without_rewrite"], 1)
        self.assertIn("query_changed_without_coreference_rewrite", result["warnings"])

    def test_isolated_capture_without_unexplained_query_change_is_controlled(self):
        left = _report("none", [_row("A", "hash", True)], mode="isolated")
        right = _report("fcoref", [_row("A", "hash", True)], mode="isolated")
        for report, scope in ((left, "none"), (right, "fcoref")):
            report["lm_cassette"] = {
                "mode": "capture",
                "path": "/tmp/cassette.json",
                "scope": scope,
                "replay_valid": True,
                "misses": 0,
                "unconsumed": 0,
            }

        result = compare_reports(left, right)

        self.assertTrue(result["causal_interpretation_valid"])
        self.assertEqual(result["warnings"], [])

    def test_capture_with_live_calls_marks_provider_timing_invalid(self):
        left = _report("none", [_row("A", "hash", True)], mode="isolated")
        right = _report("fcoref", [_row("A", "hash", True)], mode="isolated")
        for report, scope, live_calls in (
            (left, "none", 4),
            (right, "fcoref", 1),
        ):
            report["lm_cassette"] = {
                "mode": "capture",
                "path": "/tmp/cassette.json",
                "scope": scope,
                "replay_valid": True,
                "live_calls": live_calls,
                "misses": live_calls,
                "unconsumed": 0,
            }

        result = compare_reports(left, right)

        self.assertTrue(result["causal_interpretation_valid"])
        self.assertFalse(result["provider_inclusive_timing_valid"])
        self.assertEqual(
            result["timing_warnings"],
            ["capture_timing_mixes_live_and_replayed_lm_calls"],
        )


if __name__ == "__main__":
    unittest.main()
