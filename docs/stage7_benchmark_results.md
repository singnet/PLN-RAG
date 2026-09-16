# Stage 7 Benchmark Results

These results were produced from base commit
`c94ce1d9ceeb01ec782f77644a4ef87e633e27e7` plus the benchmark changes in the
same commit as this document. The container image was
`pln-rag-senf-stage7-fresh:latest`, built without cache from the pinned
PeTTaChainer revision `02a85c63be6a735f50f50e1084a717b3256b8406`.

## Counterfactual Suite

The `stage7_counterfactual_v1` suite fixes LLM generation to reviewed MeTTa
fixtures. It measures production parser post-processing, persistence, query
planning, service execution, and PeTTaChainer reasoning. It does not measure
language-generation quality.

| Arm | Correct | Authorized proofs | Safety negatives | False positives | Invalid executions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `canonical_pln` | 7/12 | 3/5 | 4/7 | 3 | 0 |
| `canonical_senf_pln_stage6` | 7/12 | 3/5 | 4/7 | 3 | 0 |
| `canonical_senf_pln_stage7` | 12/12 | 5/5 | 7/7 | 0 | 0 |

Stage 7 also passed all 38 branch, interval, probability, temporal-decay,
entity-persistence, and structured-rejection checks. Every arm ingested the
complete fixture atom set with zero rejected atoms.

Raw report: `stage7_comparison_a2e42820.json`

SHA-256: `02aab9faba57e85bc403651cf27dcfa61111091a0b765061df5e76f8c94d72dd`

## Ordinary-World Regression

The three-arm `stress25_v1` replay used the same frozen canonical generation
tape for every arm. Stage 6 and Stage 7 had exactly equivalent stable per-case
fields, with zero contextual cases and zero errors.

| Arm | Proofs | Correct answers | Errors |
| --- | ---: | ---: | ---: |
| `canonical_pln` | 22/25 | 3/25 | 0 |
| `canonical_senf_pln_stage6` | 22/25 | 8/25 | 0 |
| `canonical_senf_pln_stage7` | 22/25 | 8/25 | 0 |

Raw report: `parser_benchmark_stress25_v1_isolated_eb705aff.json`

SHA-256: `16676c03244fb08802d57ebc73806cb9d3a630ab12df4536b1f7f46c5da19cc3`

Generation tape SHA-256:
`9cac194f1f7534ee3b1130f26d66b4fdd76b8873458900cf8eff74cfa5920fe2`

Generated reports and tapes are intentionally not committed because they are
large run artifacts. The tracked suite and harness reproduce the
counterfactual comparison. Reproducing the stress25 replay additionally
requires the frozen tape identified by its hash above; the hashes identify the
exact local artifacts used for these tables.
