# stress25_v1 Benchmark Report (cumulative, d797409e)

## Artifacts

- JSON: `data/benchmarks/parser_benchmark_stress25_v1_cumulative_d797409e.json`
- Logs: `data/benchmarks/logs/parser_benchmark_stress25_v1_cumulative_d797409e.log`

## Setup

- Suite: `data/benchmarks/stress25_v1.json` (25 cases)
- Mode: `cumulative` (reset per parser)
- Parsers: `nl2pln`, `canonical_pln`, `canonical_langextract`
- Knobs: `QUERY_CANDIDATE_MAX_TRIES=5`, `ANSWER_GENERATION_ENABLED=false`, `SOURCE_LOOKUP_MAX_ATOMS=0`, `LANGEXTRACT_SKIP_FUZZY=true`, `HYBRID_QUERY_MODE=langextract_first`

## Summary

| Parser | Cases | Proof Found | No Query | Weak Align | Fallback | Avg Latency | Median Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| nl2pln | 25 | 2 | 1 | 4 | 3 | 62.0214s | 56.3767s |
| canonical_pln | 25 | 13 | 0 | 2 | 2 | 54.3642s | 54.7147s |
| canonical_langextract | 25 | 9 | 0 | 25 | 25 | 102.7891s | 106.1947s |

## Winner

- Winner: `canonical_pln`
- Why: highest `proof_found` (13/25) and best average latency (~54.36s)

## Notes

- `correct_known=0` for all parsers because `stress25_v1` does not provide `expected_proof` labels. This run compares proof discovery and diagnostics, not verified correctness.
- `canonical_langextract` shows `weakly_aligned=25/25` and `fallback_used=25/25`, indicating it relied on fallback queries for every case under `HYBRID_QUERY_MODE=langextract_first`.
