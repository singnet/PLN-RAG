# SENF Component Benchmark v1

Cases: **14**  
Schema: `senf-components-report/v2`

## Classification And Mapping Metrics

| Metric | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| bridge_adapter | 1.000000 | 1.000000 | 1.000000 | 1 | 0 | 0 |
| exemplar | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 |
| id_minus | 0.230769 | 1.000000 | 0.375000 | 3 | 10 | 0 |
| id_plus | 0.833333 | 1.000000 | 0.909091 | 10 | 2 | 0 |
| identity_merge | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 |
| mapping_entity | 1.000000 | 1.000000 | 1.000000 | 6 | 0 | 0 |
| mapping_exemplar | 1.000000 | 1.000000 | 1.000000 | 2 | 0 | 0 |
| mapping_frame | 1.000000 | 1.000000 | 1.000000 | 5 | 0 | 0 |
| mapping_location | 1.000000 | 1.000000 | 1.000000 | 1 | 0 | 0 |
| mapping_role | 1.000000 | 1.000000 | 1.000000 | 7 | 0 | 0 |
| mapping_time | 1.000000 | 1.000000 | 1.000000 | 1 | 0 | 0 |
| spans | 1.000000 | 1.000000 | 1.000000 | 6 | 0 | 0 |
| typed_mappings_micro | 1.000000 | 1.000000 | 1.000000 | 22 | 0 | 0 |

## Safety And Numerical Metrics

- Hit@3: **4/4** (1.000000)
- Wrong guards: **1**; extras alongside a hit: **1**
- False-merge rate: **0.000000**
- False-bridge rate: **0.000000**
- Exact adapter F1: **1.000000**
- Residual convergence: **3/3** (max `9.51e-08`, tolerance `1e-07`)
- Determinism: **14/14**

## Cases

| Case | Component | Deterministic | Result |
|---|---|---:|---|
| span_lending | spans | yes | exact |
| span_repeated_occurrences | spans | yes | exact |
| identity_pronoun_positive | identity | yes | no false merge |
| identity_definite_positive | identity | yes | no false merge |
| identity_pronoun_abstention | identity | yes | no false merge |
| identity_definite_abstention | identity | yes | no false merge |
| identity_camera_contradiction | identity | yes | no false merge |
| identity_lens_conflict | identity | yes | no false merge |
| identity_bridge_deny | identity | yes | no false merge |
| exemplar_games | exemplar | yes | exact |
| weave_typed_location_bridge_allow | weave | yes | top-k hit |
| weave_typed_time_exemplar | weave | yes | top-k hit |
| weave_bridge_deny_inverse | weave | yes | top-k hit |
| weave_multi_source | weave | yes | top-k hit |

## Reproducibility

- Command: `python scripts/evaluate_senf_components.py --gold data/benchmarks/senf_components_v1.json --format markdown --output docs/benchmarks/senf_components_v1.md`
- Dataset SHA-256: `94307583375462b9f680345a60cdf8026724c2232c5d8d83d82fc058e568df0d`
- Generator SHA-256: `8424aa3d6421a51c632b652072c7f557f6df060e6eef32791671d0ba1a731f55`
- Git revision: `4244dc69782dde7031ac11fe6398436959eb0a78`
- Determinism scope: two sequential runs in the same process

```json
{
  "environment": {
    "cpu_count": 16,
    "dependencies": {
      "numpy": "2.5.3",
      "pydantic": "2.13.5",
      "pydantic-settings": "2.15.0"
    },
    "platform": "Linux-7.1.3-arch1-3-x86_64-with-glibc2.43",
    "processor": "unknown",
    "python": "3.14.6"
  },
  "settings": {
    "senf_exemplar_enabled": true,
    "senf_identity_threshold": 0.75,
    "senf_query_max_candidate_work": 64,
    "senf_query_max_mentions": 256,
    "senf_query_max_priors": 16,
    "senf_query_max_source_frames": 128,
    "senf_weave_beam_width": 32,
    "senf_weave_coarse_identity": false,
    "senf_weave_engine": "hierarchical",
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
    "senf_weave_sinkhorn_tolerance": 1e-07,
    "senf_weave_top_k": 3
  }
}
```
