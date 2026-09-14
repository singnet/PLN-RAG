# Stress25 Coreference A/B

Evaluation date: 2026-09-13

## Configuration

- Suite: `stress25_v1`, all 25 cases
- Parser: `canonical_pln`
- Mode: cumulative
- Baseline: coreference disabled
- Treatment: LingMess on CPU
- Pair ID: `e8517efceacf`
- Execution validity: both runs valid, with no parser, ingest, or coreference failures

## Results

| Metric | No Coreference | LingMess |
|---|---:|---:|
| Proofs found | 17/25 | 17/25 |
| Weakly aligned queries | 2 | 1 |
| Coreference documents changed | 0 | 7 |
| Coreference replacements | 0 | 15 |
| Mean total latency | 20.342s | 24.281s |
| Median total latency | 18.558s | 22.717s |

Proof transitions were balanced:

- 5 no-proof to proof
- 5 proof to no-proof
- 12 remained proof
- 3 remained no-proof

LingMess added a mean 2.444 seconds of coreference work and 3.939 seconds of
total latency per case. Its first document included model initialization and
took 14.926 seconds for coreference; the median was 1.778 seconds.

## Interpretation

This run shows no net proof-count improvement. Stress25 does not contain
correctness labels, so a proof transition is not necessarily a correctness
transition.

The result is observational rather than a clean causal estimate. The parser LM
is configured with `cache=False` and no reproducibility seed. An isolated
negative control (`fc87d29cab6c`) ran `E01` and `E03`, where LingMess made no
text replacements. Both cases still generated different PLN queries between
the baseline and treatment runs, although both retained proofs. This confirms
that parser nondeterminism can produce query differences independently of
coreference.

The current evidence supports keeping coreference disabled by default. A
stronger downstream evaluation requires deterministic or recorded/replayed LLM
responses and correctness labels for the expected proofs.

## Artifacts

- Baseline run: `parser_benchmark_stress25_v1_cumulative_25f2fc92.json`
- LingMess run: `parser_benchmark_stress25_v1_cumulative_804e0de5.json`
- Comparison: `coreference_comparison_e8517efceacf.json`
- Isolated negative-control manifest: `paired_benchmark_fc87d29cab6c.json`
