# Stress25 Hierarchical TransWeave Ablation

This is a frozen-generation, isolated-mode comparison of the validated Stage 7
beam arm and the hierarchical TransWeave arm. It is descriptive evidence, not a
claim of statistical superiority. Both arms use `ranked_first_proof`; all
settings recorded by both reports match except `SENF_WEAVE_ENGINE`, while the
hierarchical report additionally records its solver, diagnostics, and feature
settings.

## Inputs

- Suite: `data/benchmarks/stress25_v1.json` (25 cases)
- Gold: `data/benchmarks/stress25_v1_gold.json`
- Generation tape SHA-256: `9cac194f1f7534ee3b1130f26d66b4fdd76b8873458900cf8eff74cfa5920fe2`
- Beam source report SHA-256: `5f9e810caeaea6b349bb1e457cee68978ae7bae81d25dbb439063528ff33b48f`
- Hierarchical source report SHA-256: `f48132b13fc693555e822906dcc01200a60a0023b3fc1a5ee97bfc5075158ff7`
- Hierarchical arm: `canonical_senf_pln_hierarchical_ranked_first_proof`
- Feature provider: `none`
- Executable bridges: disabled

The replay consumed all 126 generation calls with zero context mismatches,
repeated calls, unconsumed calls, or live generation calls.

## Results

| Metric | Beam | Hierarchical | Delta |
|---|---:|---:|---:|
| Proofs found | 22/25 | 22/25 | 0 |
| Correct answers | 8/25 | 7/25 | -1 |
| Mean answer score | 0.15 | 0.13 | -0.02 |
| Weak alignments | 20 | 21 | +1 |
| Query fallbacks | 19 | 20 | +1 |
| Average latency | 1.8222s | 2.0624s | +0.2402s (+13.2%) |
| Median latency | 0.3249s | 0.4658s | +0.1409s (+43.4%) |
| Hierarchical solver fallbacks | n/a | 0 | n/a |

The proof set was unchanged. Candidate ranking changed for `S06` and `E03`.
`S06` retained a proof but moved to a weaker fallback query. `E03` moved from
`Closest sun earth` to `Produces sun light`, causing the one-answer regression.

All 25 hierarchical runs converged. Global residuals were between
`1.4414e-09` and `7.9577e-08`, below the configured `1e-07` tolerance. There
were no numerical failures and no hierarchical-to-beam fallbacks.

## Decision

Hierarchical TransWeave remains available behind `SENF_WEAVE_ENGINE`, but this
run fails the answer-quality and latency promotion gates. `beam` remains the
default. Executable bridges also remain disabled pending separate paired safety
and answer-quality evidence.

The raw reports are intentionally not tracked because they contain large
per-case traces. The hashes above bind this summary to the reviewed artifacts.
