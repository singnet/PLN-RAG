# Stress25 FCoref A/B

Evaluation date: 2026-09-13

## Configuration

- Suite: `stress25_v1`, all 25 cases
- Parser: `canonical_pln`
- Mode: cumulative
- Baseline: coreference disabled
- Treatment: FCoref on CPU
- Pair ID: `4374fdb30e48`
- Execution validity: both runs valid, with no parser, ingest, or coreference failures

## Results

| Metric | No Coreference | FCoref |
|---|---:|---:|
| Proofs found | 20/25 | 16/25 |
| Weakly aligned queries | 2 | 2 |
| Coreference documents changed | 0 | 7 |
| Coreference replacements | 0 | 15 |
| Mean total latency | 26.445s | 26.650s |
| Median total latency | 24.458s | 25.381s |

Proof transitions favored the baseline:

- 2 no-proof to proof
- 6 proof to no-proof
- 14 remained proof
- 3 remained no-proof

FCoref added a mean 0.723 seconds of coreference work per case, with a median
of 0.459 seconds. The first document included model initialization and took
7.037 seconds for coreference. The measured mean total latency increase was
only 0.204 seconds because unrelated parser timing variation offset some of the
coreference cost.

## Interpretation

This single run shows a four-proof reduction, but it is not a reliable causal
estimate. Stress25 has no proof-correctness labels, and the parser LM uses
`cache=False` without a reproducibility seed.

The isolated negative control (`3179716d56b9`) evaluated `E01` and `E03`, where
FCoref made no text replacements. The generated query still changed for `E03`.
The comparator consequently marked that control as unsuitable for causal
interpretation. As with the LingMess experiment, deterministic or replayed LLM
responses are required before attributing proof gains or losses to
coreference.

Compared with LingMess, FCoref is substantially faster but was less accurate on
the labeled synthetic rewrite suite (0.687 versus 1.000 replacement F1). The
available downstream runs do not support enabling either model by default.

## Artifacts

- Baseline run: `parser_benchmark_stress25_v1_cumulative_c4752c6a.json`
- FCoref run: `parser_benchmark_stress25_v1_cumulative_c850c72f.json`
- Comparison: `coreference_comparison_4374fdb30e48.json`
- Isolated negative-control manifest: `paired_benchmark_3179716d56b9.json`
- Isolated negative-control comparison: `coreference_comparison_3179716d56b9.json`
