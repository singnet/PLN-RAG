from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict, Field
from typing import Literal, Optional


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
    query_execution_policy: Optional[
        Literal["ranked_first_proof", "ranked_first_only", "original_only"]
    ] = None

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
    senf_weave_top_k: int = Field(default=3, gt=0, le=16)
    # Global query-planning limits, before the lower-level weave bounds apply.
    senf_query_max_priors: int = 16
    senf_query_max_source_frames: int = 128
    senf_query_max_mentions: int = 256
    senf_query_max_candidate_work: int = 32
    # Deterministic weave search resource limits.
    senf_weave_per_source_k: int = Field(default=3, gt=0, le=16)
    senf_weave_beam_width: int = Field(default=32, gt=0, le=256)
    senf_weave_max_frames: int = Field(default=64, gt=0, le=128)
    senf_weave_max_pair_candidates: int = Field(default=256, gt=0, le=4_096)
    senf_weave_max_exemplar_alternatives: int = Field(default=4, gt=0, le=32)
    senf_weave_max_cost: float = 2.0
    senf_weave_engine: Literal["beam", "hierarchical"] = "beam"
    senf_weave_global_candidate_cap: int = Field(default=512, gt=0)
    senf_weave_max_cells: int = Field(default=65_536, gt=0)
    senf_weave_max_seeds: int = Field(default=32, gt=0)
    senf_weave_max_iterations: int = Field(default=200, gt=0)
    senf_weave_sinkhorn_tolerance: float = Field(
        default=1e-7, gt=0.0, allow_inf_nan=False
    )
    senf_weave_sinkhorn_regularization: float = Field(
        default=0.25, gt=0.0, allow_inf_nan=False
    )
    # Optional practical modality controls. Defaults preserve existing weaves.
    senf_weave_forget_fine_costs: bool = False
    senf_weave_coarse_identity: bool = False
    senf_weave_max_conflict_cost: float = Field(
        default=1_000_000.0, ge=0.0, allow_inf_nan=False
    )
    senf_weave_max_transport_cost: float = Field(
        default=1_000_000.0, ge=0.0, allow_inf_nan=False
    )
    senf_weave_require_context_match: bool = False
    # Weave-derived query scoring (C7). Zero reproduces pre-SENF ranking exactly.
    senf_source_grounding_weight: int = 3
    senf_role_compat_weight: int = 2
    senf_distortion_weight: int = 0
    senf_identity_support_weight: int = 2
    senf_exemplar_coherence_weight: int = 2
    senf_conflict_weight: int = 3
    senf_transport_cost_weight: int = 2
    senf_matched_soft_mass_weight: int = 0
    senf_global_residual_ratio_weight: int = 0
    senf_alignment_confidence_weight: int = 0
    # Response-only bounds. These do not change planner search or scoring.
    senf_diagnostics_max_weaves: int = Field(default=3, gt=0, le=16)
    senf_diagnostics_max_items: int = Field(default=128, gt=0, le=1_024)
    senf_diagnostics_max_evidence: int = Field(default=16, gt=0, le=64)
    # Stage 7 remains separately opt-in inside the experimental SENF parser.
    senf_counterfactual_enabled: bool = False
    senf_branch_max_nodes: int = 64
    senf_branch_max_depth: int = 16
    senf_branch_max_theory_statements: int = 128
    senf_temporal_decay_rate: float = 0.01
    # Optional feature-only enrichment. The canonical parser remains authoritative.
    senf_feature_provider: Literal["none", "langextract"] = "none"
    senf_feature_exact_only: bool = True
    senf_feature_max_features: int = Field(default=64, gt=0)
    senf_feature_timeout: float = Field(default=20.0, gt=0.0, allow_inf_nan=False)
    senf_feature_model: Optional[str] = None
    senf_feature_examples_path: str = "data/senf_feature_examples.json"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
