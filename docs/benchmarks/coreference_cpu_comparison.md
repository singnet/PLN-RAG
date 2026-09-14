# CPU Coreference Comparison

Evaluation date: 2026-09-10

## Setup

- CPU: AMD Ryzen AI 7 350, 8 cores / 16 threads
- Device: CPU only
- fastcoref: 2.1.6
- Transformers: 5.17.0
- Dataset: `data/benchmarks/coreference_pronouns_v1.json`, 36 synthetic cases
- Policy: rewrite only supported unambiguous pronouns, abstain on `her`, and do not filter raw pair logits

The fixture backend achieved 36/36 exact matches and 1.000 replacement F1,
which validates the deterministic rewrite and evaluator path independently of a
model.

## Results

| Backend | Parameters | Exact | Change Accuracy | Replacement Precision | Replacement Recall | Replacement F1 | Median CPU Latency | p95 CPU Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCoref | 90.5M | 24/36 (0.667) | 0.556 | 0.917 | 0.550 | 0.687 | 0.396s | 0.443s |
| LingMess | 590.0M | 36/36 (1.000) | 1.000 | 1.000 | 1.000 | 1.000 | 1.420s | 1.474s |

LingMess was about 3.6 times slower per short document, but substantially more
accurate on this suite. FCoref primarily missed neuter, plural, multi-cluster,
and biomedical links. These synthetic results support LingMess when rewrite
quality is the priority and FCoref when latency and memory are constrained.
They do not by themselves justify changing the service default.

## Stress25 Audit

The models were also run over the document text in all 25 Stress25 cases,
without invoking an LLM or parser:

| Backend | Documents Changed | Replacements | Failed Open |
|---|---:|---:|---:|
| FCoref | 7/25 | 15 | 0 |
| LingMess | 7/25 | 15 | 0 |

Thirteen replacements were effectively shared. The models differed in cases
`A06` and `A09`, where long biomedical noun phrases made antecedent selection
less reliable. This audit has no correctness labels and therefore measures
rewrite behavior, not accuracy. Downstream proof gains and losses must be
measured with the paired benchmark after the parser LLM endpoint is available.

## Commands

```bash
python3 scripts/evaluate_coreference.py --backend fixture --require-effective --output /tmp/coref-fixture.json
python3 scripts/evaluate_coreference.py --backend fcoref --require-effective --output /tmp/coref-fcoref.json
python3 scripts/evaluate_coreference.py --backend lingmess --require-effective --output /tmp/coref-lingmess.json
```

The first LingMess load required compatibility handling for Transformers 5:
eager Longformer attention and the current tied-weight API. Both are applied by
`FastCorefBackend` before model construction.
