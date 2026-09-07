import json
from types import SimpleNamespace

import pytest

from benchmark_replay import GenerationTape, GenerationTapeError, content_hash
from parsers.canonical_pln_parser import CanonicalPLNParser


def result(statements=None, queries=None):
    return SimpleNamespace(statements=statements or [], queries=queries or [])


def capture_one(tmp_path, *, metadata=None):
    path = tmp_path / "generation-tape.json"
    tape = GenerationTape("capture", path, metadata=metadata or {"suite": "test"})
    backend = tape.backend("canonical_pln", "A01")
    backend.set_phase("e2e_query")
    returned = backend.generate(
        sentences=["Is the camera expensive?"],
        context=["(: a (HasProperty camera expensive) (STV 1.0 1.0))"],
        pln_spec="spec",
        live=lambda **_: result(queries=["(: $prf (HasProperty camera expensive) $tv)"]),
    )
    backend.finish()
    return path, returned


class TestGenerationTape:
    def test_capture_and_replay_round_trip_without_live_call(self, tmp_path):
        path, captured = capture_one(tmp_path)
        tape = GenerationTape("replay", path, metadata={"suite": "test"})
        backend = tape.backend("canonical_pln", "A01")
        backend.set_phase("e2e_query")

        replayed = backend.generate(
            sentences=["Is the camera expensive?"],
            context=["(: a (HasProperty camera expensive) (STV 1.0 1.0))"],
            pln_spec="spec",
            live=lambda **_: pytest.fail("replay called the live LLM"),
        )
        backend.finish()

        assert replayed.queries == captured.queries
        assert tape.report()["parser_runs"][0]["context_mismatches"] == 0

    def test_tape_stores_raw_inputs_outputs_and_hashes(self, tmp_path):
        path, _ = capture_one(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        call = payload["calls"][0]

        assert call["case_id"] == "A01"
        assert call["phase"] == "e2e_query"
        assert call["context_sha256"] == content_hash(call["context"])
        assert call["queries"] == ["(: $prf (HasProperty camera expensive) $tv)"]

    def test_metadata_mismatch_fails_before_replay(self, tmp_path):
        path, _ = capture_one(tmp_path, metadata={"suite": "stress25", "cases": "abc"})
        with pytest.raises(GenerationTapeError, match="cases"):
            GenerationTape(
                "replay", path, metadata={"suite": "stress25", "cases": "changed"}
            )

    def test_missing_scope_fails(self, tmp_path):
        path, _ = capture_one(tmp_path)
        backend = GenerationTape("replay", path).backend("canonical_pln", "A02")
        backend.set_phase("e2e_query")
        with pytest.raises(GenerationTapeError, match="No generation calls"):
            backend.generate(
                sentences=["question"], context=[], pln_spec="spec", live=lambda **_: None
            )

    def test_canonical_context_mismatch_fails(self, tmp_path):
        path, _ = capture_one(tmp_path)
        backend = GenerationTape("replay", path).backend("canonical_pln", "A01")
        backend.set_phase("e2e_query")
        with pytest.raises(GenerationTapeError, match="Context mismatch"):
            backend.generate(
                sentences=["Is the camera expensive?"],
                context=["different"],
                pln_spec="spec",
                live=lambda **_: None,
            )

    def test_senf_context_drift_is_reported_not_rejected(self, tmp_path):
        path, _ = capture_one(tmp_path)
        tape = GenerationTape("replay", path)
        backend = tape.backend("canonical_senf_pln", "A01")
        backend.set_phase("e2e_query")
        replayed = backend.generate(
            sentences=["Is the camera expensive?"],
            context=["senf-changed-context"],
            pln_spec="spec",
            live=lambda **_: pytest.fail("replay called the live LLM"),
        )
        backend.finish()

        assert replayed.queries
        assert tape.report()["parser_runs"][0]["context_mismatches"] == 1

    def test_extra_canonical_call_fails(self, tmp_path):
        path, _ = capture_one(tmp_path)
        backend = GenerationTape("replay", path).backend("canonical_pln", "A01")
        backend.set_phase("e2e_query")
        arguments = {
            "sentences": ["Is the camera expensive?"],
            "context": ["(: a (HasProperty camera expensive) (STV 1.0 1.0))"],
            "pln_spec": "spec",
            "live": lambda **_: None,
        }
        backend.generate(**arguments)
        with pytest.raises(GenerationTapeError, match="extra generation call"):
            backend.generate(**arguments)

    def test_unconsumed_canonical_call_fails(self, tmp_path):
        path, _ = capture_one(tmp_path)
        backend = GenerationTape("replay", path).backend("canonical_pln", "A01")
        with pytest.raises(GenerationTapeError, match="Unconsumed"):
            backend.finish()


class TestCanonicalGenerationSeam:
    def test_default_generation_calls_live_module(self):
        parser = CanonicalPLNParser.__new__(CanonicalPLNParser)
        parser._generation_backend = None
        parser._pln_spec = "spec"
        calls = []
        parser._nl2pln = lambda **kwargs: calls.append(kwargs) or result(statements=["fact"])

        generated = parser._generate(["sentence"], ["context"])

        assert generated.statements == ["fact"]
        assert calls == [
            {"sentences": ["sentence"], "context": ["context"], "pln_spec": "spec"}
        ]

    def test_installed_backend_owns_generation(self):
        parser = CanonicalPLNParser.__new__(CanonicalPLNParser)
        parser._pln_spec = "spec"
        parser._nl2pln = lambda **_: pytest.fail("backend should replace the live module")

        class Backend:
            def generate(self, **kwargs):
                assert kwargs["sentences"] == ["sentence"]
                return result(queries=["query"])

        parser.set_generation_backend(Backend())
        assert parser._generate(["sentence"], []).queries == ["query"]
