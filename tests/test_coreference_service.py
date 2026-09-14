import unittest
from types import SimpleNamespace

from core.coreference import CoreferenceStatus, ResolvedDocument
from core.parser import ParseResult, SemanticParser
from core.service import PLNRAGService


class Parser(SemanticParser):
    def __init__(self):
        self.seen = []

    def parse(self, text, context):
        self.seen.append(text)
        return ParseResult(
            statements=["(: fact (Evaluation (Predicate P) (Concept A)) (STV 1 1))"]
        )


class CoreferenceServiceTests(unittest.TestCase):
    def test_ingest_resolves_once_before_chunking(self):
        resolved = ResolvedDocument(
            original="Alice arrived. She smiled.",
            resolved="Alice arrived. Alice smiled.",
            backend="fixture",
            model="fixture/model",
            status="resolved",
            duration_seconds=0.25,
        )
        parser = Parser()
        service = PLNRAGService.__new__(PLNRAGService)
        service._coreference = SimpleNamespace(resolve=lambda text: resolved)
        service._parser = parser
        service._chunker = SimpleNamespace(chunk=lambda text: [text])
        service._vector_store = SimpleNamespace(
            retrieve_context=lambda text, top_k: ([], [0.0]),
            store=lambda text, atoms, vector: None,
        )
        service._reasoner = SimpleNamespace(
            add_statements_report=lambda statements, provenance=None: (statements, [])
        )
        service._context_top_k = 1
        service._enrich_context = lambda context: context

        result = service._ingest_single(resolved.original)

        self.assertEqual(parser.seen, [resolved.resolved])
        self.assertEqual(result.text, resolved.original)
        self.assertEqual(result.coreference.status, "resolved")
        self.assertEqual(result.coreference.resolved_text, resolved.resolved)

    def test_later_ingest_failure_retains_coreference_diagnostics(self):
        resolved = ResolvedDocument(
            original="Alice arrived. She smiled.",
            resolved="Alice arrived. Alice smiled.",
            backend="fixture",
            status="resolved",
        )
        parser = Parser()
        parser.parse = lambda text, context: (_ for _ in ()).throw(RuntimeError("parse failed"))
        service = PLNRAGService.__new__(PLNRAGService)
        service._coreference = SimpleNamespace(resolve=lambda text: resolved)
        service._parser = parser
        service._chunker = SimpleNamespace(chunk=lambda text: [text])
        service._vector_store = SimpleNamespace(
            retrieve_context=lambda text, top_k: ([], [0.0])
        )
        service._context_top_k = 1
        service._enrich_context = lambda context: context

        result = service._ingest_single(resolved.original)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.coreference.status, "resolved")
        self.assertEqual(result.coreference.resolved_text, resolved.resolved)

    def test_health_and_readiness_report_degraded_fail_open_backend(self):
        status = CoreferenceStatus(
            enabled=True,
            fail_open=True,
            backend="lingmess",
            model="model",
            state="degraded",
            score_available=False,
            documents_processed=1,
            documents_changed=0,
            replacements=0,
            failures=1,
            total_duration_seconds=1.0,
            last_duration_seconds=1.0,
            last_error="load failed",
        )
        service = PLNRAGService.__new__(PLNRAGService)
        service._coreference = SimpleNamespace(status=lambda: status)
        service._conceptnet = SimpleNamespace(
            status=lambda: {
                "enabled": False,
                "indexing": False,
                "indexed_count": 0,
                "expected_count": 0,
                "last_error": "",
            }
        )
        service._reasoner = SimpleNamespace(size=0, background_size=0)
        service._vector_store = SimpleNamespace(
            count=0,
            is_qdrant_available=lambda: (True, "ok"),
            is_ollama_available=lambda: (True, "ok"),
        )
        service._parser = Parser()

        self.assertEqual(service.health()["status"], "degraded")
        readiness = service.ready()
        self.assertEqual(readiness["status"], "degraded")
        self.assertEqual(readiness["coreference"]["failures"], 1)

    def test_fail_closed_not_loaded_backend_is_not_ready(self):
        status = CoreferenceStatus(
            enabled=True,
            fail_open=False,
            backend="fcoref",
            model="model",
            state="not_loaded",
            score_available=False,
            documents_processed=0,
            documents_changed=0,
            replacements=0,
            failures=0,
            total_duration_seconds=0.0,
            last_duration_seconds=0.0,
            last_error=None,
        )
        service = PLNRAGService.__new__(PLNRAGService)
        service._coreference = SimpleNamespace(status=lambda: status)
        service._conceptnet = SimpleNamespace(
            status=lambda: {
                "enabled": False,
                "indexing": False,
                "indexed_count": 0,
                "expected_count": 0,
                "last_error": "",
            }
        )
        service._reasoner = SimpleNamespace(size=0, background_size=0)
        service._vector_store = SimpleNamespace(
            count=0,
            is_qdrant_available=lambda: (True, "ok"),
            is_ollama_available=lambda: (True, "ok"),
        )
        service._parser = Parser()

        self.assertEqual(service.ready()["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
