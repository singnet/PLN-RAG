# Coreference Experiment Evidence

Evaluation period: 2026-09-10 through 2026-09-14

This directory preserves the available machine-readable evidence for the CPU
coreference experiment described in
[`docs/benchmarks/coreference_downstream_evaluation.md`](../../../../docs/benchmarks/coreference_downstream_evaluation.md).

## Publication Boundary

The repository is public. This bundle includes successful benchmark reports,
paired-run manifests, comparisons, labeled evaluations, and deterministic
replay reports. It deliberately excludes:

- LM cassette response payloads.
- Failed replay attempts caused by an intentionally mismatched LM
  configuration.
- Redundant smoke-test reports.
- Unrelated historical files under `data/benchmarks/`.

The cassette remains local at `/tmp/opencode/stress7_structural_v1_cassette.json`
on the machine that ran the experiment. A repository clone can audit the
captured and replayed results, including replay counters and logical hashes, but
cannot independently reproduce the exact captured LM responses without that
local cassette.

The source suite contains abstracts marked `requires_manual_copy=true` because
their redistribution rights were not verified. The reports preserve the inputs
used by the experiment and therefore inherit the copyright caution documented
in `data/benchmarks/stress25_v1.json`.

## Directory Contents

### `stress25_cumulative/`

Exploratory cumulative runs performed before deterministic LM control and exact
proof-quality scoring were available.

- Pair `e8517efceacf`: no coreference versus LingMess.
- Pair `4374fdb30e48`: no coreference versus FCoref.
- Four raw benchmark reports.
- Two paired-run manifests.
- Two raw proof-presence comparisons.

These files are historical evidence, not causal estimates. The parser LM was
nondeterministic, the runs used cumulative atomspace state, and raw proof
presence did not establish correctness.

### `negative_controls/`

Isolated two-case controls where the treatment made no document rewrite but the
parser still changed a query:

- Pair `fc87d29cab6c`: no coreference versus LingMess.
- Pair `3179716d56b9`: no coreference versus FCoref.

The available paired manifests and comparisons are preserved. Their raw parser
reports were no longer present when this evidence bundle was assembled.

### `stress7/`

The authoritative deterministic experiment for cases A01, A04, A06, A08, A09,
A10, and A11.

Capture reports:

| Backend | Run ID | File |
|---|---|---|
| None | `ae5fb7d3` | `parser_benchmark_stress25_v1_isolated_ae5fb7d3.json` |
| FCoref | `3bcc8505` | `parser_benchmark_stress25_v1_isolated_3bcc8505.json` |
| LingMess | `6de65f2a` | `parser_benchmark_stress25_v1_isolated_6de65f2a.json` |

Successful replay reports:

| Backend | Replay 1 | Replay 2 |
|---|---|---|
| None | `72121a0b` | `77ae8245` |
| FCoref | `813e6e2f` | `07cd472a` |
| LingMess | `975dbf24` | `77f1ddc6` |

Each replay report records zero live LM calls, zero misses, and zero unconsumed
calls. `logical_equivalence.json` records the normalized SHA-256 digest shared
by capture and both replays for each backend.

The three `stress7_*_evaluation.json` files contain model-free proof-quality
scores. The two `stress7_none_vs_*.json` files contain paired proof and quality
transitions. Provider-inclusive capture timing is explicitly marked invalid
because shared response-pool reuse caused asymmetric live LM calls.

## Authoritative Result

| Configuration | Passed | Average score |
|---|---:|---:|
| No coreference | 0/7 | 0.148810 |
| FCoref | 0/7 | 0.059524 |
| LingMess | 0/7 | 0.059524 |

Neither model produced a fail-to-pass quality transition. Both lost three raw
proofs and gained one current-case-grounded but semantically incomplete proof.
The evidence supports keeping coreference disabled by default.

## Integrity

`SHA256SUMS` contains raw file digests for this evidence directory, excluding
`SHA256SUMS` itself. Verify it from this directory with:

```bash
sha256sum --check SHA256SUMS
```

Raw checksums establish file integrity. `stress7/logical_equivalence.json`
separately establishes logical equality after removing only documented volatile
run, storage, cassette-counter, and timing fields.
