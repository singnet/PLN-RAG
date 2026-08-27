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

    # Parser-specific diagnostics, absent unless the active parser reports any.
    senf: Optional[dict] = None


#  Reset 

class ResetRequest(BaseModel):
    scope: Literal["all", "vectordb", "atomspace"] = "all"


class ResetResponse(BaseModel):
    status: Literal["ok"]
    scope: str


#  Health 

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
    details: dict[str, Any] = Field(default_factory=dict)
