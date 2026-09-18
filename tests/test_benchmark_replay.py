import json
from types import SimpleNamespace

import pytest

from benchmark_replay import GenerationTape, GenerationTapeError, content_hash
from parsers.canonical_pln_parser import CanonicalPLNParser


def result(statements=None, queries=None):
    return SimpleNamespace(statements=statements or [], queries=queries or [])


def extraction():
    return SimpleNamespace(
        extraction_class="senf_feature", extraction_text="She",
        attributes={"name": "mention_type", "value": "pronoun"},
        char_interval=SimpleNamespace(start_pos=0, end_pos=3),
        alignment_status=SimpleNamespace(name="MATCH_EXACT"),
    )


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

        assert call["parser"] == "canonical_pln"
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

    def test_feature_configuration_drift_fails_before_replay(self, tmp_path):
        path, _ = capture_one(
            tmp_path,
            metadata={"suite": "test", "feature_configs": {
                "canonical_senf_pln": {"config_sha256": "old"},
            }},
        )
        with pytest.raises(GenerationTapeError, match="feature_configs"):
            GenerationTape(
                "replay",
                path,
                metadata={"suite": "test", "feature_configs": {
                    "canonical_senf_pln": {"config_sha256": "new"},
                }},
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        payload.pop("feature_calls")
        path.write_text(json.dumps(payload), encoding="utf-8")
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

    def test_schema2_senf_generation_replay_is_strict(self, tmp_path):
        path = tmp_path / "senf-generation.json"
        capture = GenerationTape("capture", path)
        backend = capture.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        arguments = {
            "sentences": ["question"], "context": ["context"], "pln_spec": "spec",
            "live": lambda **_: result(queries=["query"]),
        }
        backend.generate(**arguments)
        backend.finish()

        replay = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        replay.set_phase("query")
        with pytest.raises(GenerationTapeError, match="Context mismatch"):
            replay.generate(**{**arguments, "context": ["changed"]})

        replay = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        with pytest.raises(GenerationTapeError, match="Unconsumed generation calls"):
            replay.finish()

        replay = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        replay.set_phase("query")
        replay.generate(**arguments)
        with pytest.raises(GenerationTapeError, match="extra generation call"):
            replay.generate(**arguments)

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


class TestFeatureTape:
    def test_schema2_feature_capture_replay_is_strict(self, tmp_path):
        path = tmp_path / "features.json"
        capture = GenerationTape("capture", path)
        backend = capture.backend("canonical_senf_pln", "A01")
        backend.set_phase("e2e_query")
        backend.feature_call(
            request={"text": "She"}, config={"model": "m", "max_features": 4},
            live=lambda: [extraction()],
        )
        backend.finish()

        replay = GenerationTape("replay", path)
        backend = replay.backend("canonical_senf_pln", "A01")
        backend.set_phase("e2e_query")
        restored = backend.feature_call(
            request={"text": "She"}, config={"model": "m", "max_features": 4},
            live=lambda: pytest.fail("feature replay called live"),
        )
        backend.finish()
        assert restored[0].extraction_text == "She"

    def test_schema2_feature_calls_are_owned_by_parser(self, tmp_path):
        path = tmp_path / "features.json"
        capture = GenerationTape("capture", path)
        backend = capture.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        backend.feature_call(
            request={"text": "She"}, config={"model": "m", "max_features": 4}, live=lambda: [extraction()],
        )
        backend.finish()

        other = GenerationTape("replay", path).backend("canonical_pln", "A01")
        other.finish()

    def test_schema2_rejects_malformed_feature_payload(self, tmp_path):
        path = tmp_path / "features.json"
        capture = GenerationTape("capture", path)
        backend = capture.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        backend.feature_call(
            request={"text": "She"}, config={"model": "m", "max_features": 4}, live=lambda: [extraction()],
        )
        backend.finish()
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["feature_calls"][0]["extractions"][0]["alignment"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        backend = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        with pytest.raises(GenerationTapeError, match="invalid schema"):
            backend.feature_call(
                request={"text": "She"}, config={"model": "m", "max_features": 4}, live=lambda: [],
            )

    def test_feature_replay_rejects_extra_hash_mismatch_and_unconsumed(self, tmp_path):
        path = tmp_path / "features.json"
        capture = GenerationTape("capture", path)
        backend = capture.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        backend.feature_call(
            request={"text": "She"}, config={"model": "m", "max_features": 4},
            live=lambda: [extraction()],
        )
        backend.finish()

        backend = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        with pytest.raises(GenerationTapeError, match="Feature (text|request) mismatch"):
            backend.feature_call(
                request={"text": "He"}, config={"model": "m", "max_features": 4}, live=lambda: [],
            )

        backend = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        with pytest.raises(GenerationTapeError, match="Unconsumed feature calls"):
            backend.finish()

        backend = GenerationTape("replay", path).backend("canonical_senf_pln", "A01")
        backend.set_phase("query")
        arguments = {
            "request": {"text": "She"}, "config": {"model": "m", "max_features": 4},
            "live": lambda: [],
        }
        backend.feature_call(**arguments)
        with pytest.raises(GenerationTapeError, match="extra feature call"):
            backend.feature_call(**arguments)

    def test_feature_capture_failure_aborts_without_incomplete_entry(self, tmp_path):
        tape = GenerationTape("capture", tmp_path / "tape.json")
        backend = tape.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")

        with pytest.raises(GenerationTapeError, match="Feature capture failed"):
            backend.feature_call(
                request={"text": "She"},
                config={"model": "m", "max_features": 4},
                live=lambda: (_ for _ in ()).throw(TimeoutError("offline")),
            )

        assert tape.feature_calls == []

    def test_feature_capture_serializes_only_cap_and_overflow_sentinel(self, tmp_path):
        path = tmp_path / "bounded.json"
        tape = GenerationTape("capture", path)
        backend = tape.backend("canonical_senf_pln", "A01")
        backend.set_phase("query")

        def extractions():
            yield extraction()
            overflow = extraction()
            overflow.extraction_text = "rejected raw secret"
            yield overflow
            pytest.fail("feature capture consumed beyond the overflow sentinel")

        returned = backend.feature_call(
            request={"text": "She"},
            config={"model": "m", "max_features": 1},
            live=extractions,
        )
        backend.finish()

        assert len(returned) == 2
        serialized = path.read_text(encoding="utf-8")
        assert "rejected raw secret" not in serialized
        assert json.loads(serialized)["feature_calls"][0]["extractions"][1][
            "extraction_class"
        ] == "__senf_feature_overflow__"

    def test_schema1_tape_remains_readable(self, tmp_path):
        path, _ = capture_one(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        payload.pop("feature_calls")
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert GenerationTape("replay", path).feature_calls == []
