from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_model: str = "openai/gpt-4o-mini"
    openai_base_url: Optional[str] = None

    # Options: "nl2pln" | "canonical_pln" | "manhin" | "langextract" | "canonical_langextract"
    parser: str = "canonical_pln"
    nl2pln_module_path: str = "data/simba_all.json"
    canonical_pln_nl2pln_module_path: str = "data/simba_canonical_pln.json"

    # LangExtract parser
    langextract_api_key: Optional[str] = None
    langextract_model_id: str = "gpt-4o-mini"
    langextract_model_url: Optional[str] = None
    langextract_examples_path: str = "data/langextract_examples.json"
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 1
    langextract_skip_fuzzy: bool = True
    langextract_chunk_size: Optional[int] = None

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pln_rag"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "nomic-embed-text"

    # Atomspace persistence
    atomspace_path: str = "data/atomspace/kb.metta"

    # FAISS predicate store (used by Manhin parser)
    faiss_path: str = "data/faiss"

    # Processing
    chunk_size: int = 512  # chars per chunk
    chunk_overlap: int = 64  # overlap between chunks
    context_top_k: int = 10  # atoms to retrieve as parser context
    parser_batch_sentences: int = 4
    parser_batch_max_chars: int = 2000

    # Reasoning
    chaining_timeout: int = 30  # seconds before proof search is killed
    chaining_max_steps: int = 100

    # Query execution
    query_fallback_enabled: bool = True

    # Maximum number of query candidates to try before giving up.
    # Applies to all parsers when query_fallback_enabled is true.
    # Set to 0 to disable the cap.
    query_candidate_max_tries: int = 5

    # Query performance knobs
    answer_generation_enabled: bool = True
    # Default off: source lookup is expensive and not required
    # for proof search or benchmarking.
    source_lookup_max_atoms: int = 0

    # Hybrid query behavior (canonical_langextract)
    # Options: "langextract_first" | "canonical_first" | "canonical_only"
    hybrid_query_mode: str = "langextract_first"

    # ConceptNet background knowledge
    conceptnet_enabled: bool = False
    conceptnet_autoload: bool = True
    conceptnet_input_file: str = "data/conceptnet/conceptnet-assertions-5.7.0.csv.gz"
    conceptnet_atomspace_path: str = "data/conceptnet/conceptnet_background.metta"
    conceptnet_vector_payload_path: str = (
        "data/conceptnet/conceptnet_background.jsonl"
    )
    conceptnet_manifest_path: str = "data/conceptnet/conceptnet_manifest.json"
    conceptnet_index_on_startup: bool = True
    conceptnet_min_weight: float = 2.0
    conceptnet_coverage_percent: float = 100.0
    conceptnet_sample_seed: int = 42
    conceptnet_auto_rebuild_on_change: bool = False
    conceptnet_reindex_on_reset: bool = True
    conceptnet_startup_fail_open: bool = True

    # SENF extension (canonical_senf_pln)
    senf_identity_threshold: float = 0.75
    senf_context_top_k: int = 10
    senf_session_max_frames: int = 200
    senf_use_vector_context: bool = True
    senf_exemplar_enabled: bool = True
    senf_emit_bridge_atoms: bool = False
    senf_transport_truth_values: bool = True
    senf_weave_top_k: int = 3
    # Weave-derived query scoring (C7). Zero reproduces pre-SENF ranking exactly.
    senf_source_grounding_weight: int = 3
    senf_role_compat_weight: int = 2
    senf_distortion_weight: int = 0
    senf_identity_support_weight: int = 2
    senf_exemplar_coherence_weight: int = 2
    senf_conflict_weight: int = 3
    senf_transport_cost_weight: int = 2

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
