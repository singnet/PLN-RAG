from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Any


#  Ingest 

class IngestRequest(BaseModel):
    texts: List[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("texts must contain at least one item")
        if not any(str(item).strip() for item in value):
            raise ValueError("texts must contain at least one non-empty item")
        return value


class CoreferenceSummary(BaseModel):
    backend: str
    model: Optional[str] = None
    status: Literal["disabled", "unchanged", "resolved", "failed_open"]
    changed: bool
    replacement_count: int
    duration_seconds: float
    score_available: bool
    error: Optional[str] = None
    diagnostics: List[str] = Field(default_factory=list)
    resolved_text: Optional[str] = Field(default=None, exclude=True)


class IngestItemResult(BaseModel):
    text: str
    atoms: List[str] = []
    status: Literal["success", "failed"]
    error: Optional[str] = None
    chunk_count: int = 0
    batch_count: int = 0
    batch_sizes: List[int] = []
    parser_calls: int = 0

    # Parser/reasoner contract diagnostics
    rejected_count: int = 0
    rejected_samples: List[str] = []
    coreference: Optional[CoreferenceSummary] = None


class IngestResponse(BaseModel):
    processed_count: int
    results: List[IngestItemResult]


#  Reason 

class ReasonRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("query must not be empty")
        return value


class ReasonResponse(BaseModel):
    query: str
    pln_query: str
    original_query: str
    executed_query: str
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "no_query"]
    proof: str
    sources: List[str]       # NL sentences that contributed to the proof
    answer: str
    # Internal evaluation diagnostics; omitted from public API serialization.
    proof_provenance: List[dict[str, Any]] = Field(default_factory=list, exclude=True)
    query_support_atoms: List[str] = Field(default_factory=list, exclude=True)

    # Candidate execution diagnostics
    candidate_count: Optional[int] = None  # total candidates available
    candidate_count_tried: Optional[int] = None
    executed_candidate_index: Optional[int] = None
    retry_used: Optional[bool] = None

    # Optional query path timings (seconds)
    context_retrieval_seconds: Optional[float] = None
    parse_query_seconds: Optional[float] = None
    reasoning_seconds: Optional[float] = None
    source_lookup_seconds: Optional[float] = None
    answer_generation_seconds: Optional[float] = None


#  Reset 

class ResetRequest(BaseModel):
    scope: Literal["all", "vectordb", "atomspace"] = "all"


class ResetResponse(BaseModel):
    status: Literal["ok"]
    scope: str


#  Health 

class CoreferenceHealth(BaseModel):
    enabled: bool
    fail_open: bool
    backend: str
    model: Optional[str] = None
    state: Literal["disabled", "not_loaded", "ready", "degraded"]
    score_available: bool
    documents_processed: int
    documents_changed: int
    replacements: int
    failures: int
    total_duration_seconds: float
    last_duration_seconds: float
    last_error: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    parser: str
    atomspace_size: int
    background_atomspace_size: int
    vectordb_count: int
    conceptnet_enabled: bool
    conceptnet_indexing: bool
    conceptnet_vectors_indexed: int
    conceptnet_vectors_expected: int
    conceptnet_last_error: str
    coreference: CoreferenceHealth
    uptime_seconds: float


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    parser: str
    reasoner_ready: bool
    qdrant_ready: bool
    ollama_ready: bool
    conceptnet_enabled: bool
    conceptnet_status: str
    conceptnet_last_error: str
    coreference: CoreferenceHealth
    details: dict[str, Any] = Field(default_factory=dict)
