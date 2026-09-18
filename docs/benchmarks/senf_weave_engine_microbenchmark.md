# SENF Weave Engine Microbenchmark

This is a deterministic synthetic, service-free comparison. No LLM calls are made.
Wall times are descriptive for the recorded environment and are not regression thresholds.

## Reproduce

```bash
python scripts/benchmark_weave_engines.py --sizes 2,4 --sparsities 0.0,0.25,1.0 --sources 2 --repeats 3 --warmups 1 --top-k 3 --json-output docs/benchmarks/senf_weave_engine_microbenchmark.json --markdown-output docs/benchmarks/senf_weave_engine_microbenchmark.md
```

## Environment

- `python`: 3.10.12
- `platform`: Linux-7.1.3-arch1-3-x86_64-with-glibc2.35
- `processor`: x86_64
- `cpu_count`: 16

## Results

| Case | Engine | Hash | Order stable | Pairs | Candidates | Max residual | Fallback | Median ms | Work / cap | Matrix / cap |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| n2-s0.000-sources2 | beam | `84aa5e821c35` | true | 5 | 0 | n/a | none | 0.408914 | 8 / 8192 | 36 / 65536 |
| n2-s0.000-sources2 | hierarchical | `c730d52bce27` | true | 6 | 4 | 7.056085715717586e-08 | none | 11.957499 | 8 / 8192 | 36 / 65536 |
| n2-s0.250-sources2 | beam | `fba15882e13b` | true | 4 | 0 | n/a | none | 0.446132 | 8 / 8192 | 25 / 65536 |
| n2-s0.250-sources2 | hierarchical | `05fb94d5665e` | true | 5 | 3 | 9.678907963994732e-08 | none | 10.416147 | 8 / 8192 | 25 / 65536 |
| n2-s1.000-sources2 | beam | `b88ab533cee5` | true | 0 | 0 | n/a | none | 0.143795 | 8 / 8192 | 4 / 65536 |
| n2-s1.000-sources2 | hierarchical | `b6acc066768b` | true | 0 | 0 | n/a | no_hard_compatible_candidates | 0.093382 | 8 / 8192 | 4 / 65536 |
| n4-s0.000-sources2 | beam | `f1076b0a5f43` | true | 11 | 0 | n/a | none | 1.957439 | 32 / 8192 | 144 / 65536 |
| n4-s0.000-sources2 | hierarchical | `c599929fd77f` | true | 12 | 8 | 8.569811416059281e-08 | none | 24.928196 | 32 / 8192 | 144 / 65536 |
| n4-s0.250-sources2 | beam | `08ada730058f` | true | 8 | 0 | n/a | none | 0.991168 | 32 / 8192 | 100 / 65536 |
| n4-s0.250-sources2 | hierarchical | `bb221dcd7a9d` | true | 12 | 6 | 9.678907963994732e-08 | none | 25.535657 | 32 / 8192 | 100 / 65536 |
| n4-s1.000-sources2 | beam | `b88ab533cee5` | true | 0 | 0 | n/a | none | 0.304932 | 32 / 8192 | 16 / 65536 |
| n4-s1.000-sources2 | hierarchical | `b6acc066768b` | true | 0 | 0 | n/a | no_hard_compatible_candidates | 0.244841 | 32 / 8192 | 16 / 65536 |

`Pairs` is the sum across returned top-k results. `Candidates` is hierarchical polish diagnostics; the beam engine does not expose an equivalent counter. Matrix values are conservative dense-cell upper bounds for the hierarchical Sinkhorn and assignment stages.

## Reproducibility

- Dataset SHA-256: `f88f5c18bb7aee900a802acf795c1449877d9da64257ede641b61af025627fe4`
- Generator SHA-256: `ed3fd1d2abd07308f0766a24a9b2da7644f90d7c27cf841bf17c3d5efb4a163e`
- Git revision: `4244dc69782dde7031ac11fe6398436959eb0a78`
- Determinism scope: same-process repeats plus one fresh-process hash test in pytest

```json
{
  "dependencies": {
    "numpy": "2.2.6",
    "pydantic": "2.13.5",
    "pydantic-settings": "2.15.0"
  },
  "settings": {
    "senf_query_max_priors": 16,
    "senf_query_max_source_frames": 128,
    "senf_weave_beam_width": 32,
    "senf_weave_coarse_identity": false,
    "senf_weave_engine": [
      "beam",
      "hierarchical"
    ],
    "senf_weave_forget_fine_costs": false,
    "senf_weave_global_candidate_cap": 512,
    "senf_weave_max_cells": 65536,
    "senf_weave_max_conflict_cost": 1000000.0,
    "senf_weave_max_cost": 2.0,
    "senf_weave_max_exemplar_alternatives": 4,
    "senf_weave_max_frames": 64,
    "senf_weave_max_iterations": 200,
    "senf_weave_max_pair_candidates": 256,
    "senf_weave_max_seeds": 32,
    "senf_weave_max_transport_cost": 1000000.0,
    "senf_weave_per_source_k": 3,
    "senf_weave_require_context_match": false,
    "senf_weave_sinkhorn_regularization": 0.25,
    "senf_weave_sinkhorn_tolerance": 1e-07
  }
}
```
