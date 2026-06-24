import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.models import (
    IngestRequest, IngestResponse,
    ReasonRequest, ReasonResponse,
    ResetRequest, ResetResponse,
    HealthResponse, ReadyResponse,
)
from core.service import PLNRAGService
from parsers import get_parser
from config import get_settings

_start_time = time.time()
_service: PLNRAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    cfg = get_settings()
    print(f"[Startup] Loading parser: {cfg.parser}")
    parser = get_parser()
    _service = PLNRAGService(parser)
    print("[Startup] Service ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title="PLN-RAG API",
    description="Probabilistic Logic Network RAG service",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service() -> PLNRAGService:
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "service_not_ready", "message": "Service not ready"},
        )
    return _service


def ensure_dependencies_ready(svc: PLNRAGService):
    info = svc.ready()
    if info["status"] == "unavailable":
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dependencies_unavailable",
                "message": "Required dependencies are unavailable",
                "details": info["details"],
            },
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Ingest a batch of texts into the knowledge base.
    Texts are chunked, parsed into PLN atoms, added to the
    atomspace, and indexed in the vector store.
    Processing is sequential — each text sees all previous atoms.
    """
    svc = get_service()
    ensure_dependencies_ready(svc)
    results = await svc.ingest_batch(req.texts)
    return IngestResponse(
        processed_count=len(results),
        results=results
    )


@app.post("/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest):
    """
    Run a reasoning request against the knowledge base.
    The query is parsed into a PLN query, reasoned over via
    PeTTaChainer, and the proof trace is translated to natural language.
    """
    svc = get_service()
    ensure_dependencies_ready(svc)
    return await svc.reason(req.query)


@app.delete("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = ResetRequest()):
    """
    Clear the knowledge base.
    scope='all': clears atomspace + vector DB
    scope='atomspace': clears only the PLN atomspace
    scope='vectordb': clears only Qdrant
    """
    svc = get_service()
    svc.reset(req.scope)
    return ResetResponse(status="ok", scope=req.scope)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — returns process/component status and sizes."""
    svc = get_service()
    info = svc.health()
    return HealthResponse(
        status=info["status"],
        parser=info["parser"],
        atomspace_size=info["atomspace_size"],
        background_atomspace_size=info["background_atomspace_size"],
        vectordb_count=info["vectordb_count"],
        conceptnet_enabled=info["conceptnet_enabled"],
        conceptnet_indexing=info["conceptnet_indexing"],
        conceptnet_vectors_indexed=info["conceptnet_vectors_indexed"],
        conceptnet_vectors_expected=info["conceptnet_vectors_expected"],
        conceptnet_last_error=info["conceptnet_last_error"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness check — returns dependency availability and degraded state."""
    svc = get_service()
    info = svc.ready()
    return ReadyResponse(**info)
