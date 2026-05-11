from pydantic import BaseModel
from typing import List, Optional, Literal


#  Ingest 

class IngestRequest(BaseModel):
    texts: List[str]


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


#  Query 

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    pln_query: str
    original_query: str
    executed_query: str
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    raw_proof: str
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
