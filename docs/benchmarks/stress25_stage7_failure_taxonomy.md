# Proof-Bearing Failure Taxonomy

Suite: `stress25_v1`  
Run: `eb705aff`  
Parser: `canonical_senf_pln_stage7`  
Source report SHA-256: `16676c03244fb08802d57ebc73806cb9d3a630ab12df4536b1f7f46c5da19cc3`

This is a deterministic symptom taxonomy, not a root-cause annotation. Proof evidence uses the benchmark grader. Query answer-symbol misses are neutral diagnostics: a valid query need not contain the answer it is intended to derive.

Proof-bearing incorrect cases: **14**  
Incorrect cases without proofs: **3**

| Case | Primary symptom | Status | Candidate | Evidence | Executed query | Proof conclusion |
| --- | --- | --- | ---: | --- | --- | --- |
| A02 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss, verdict_mismatch | (: $prf (ExaminedEffect trial iot_effect) $tv) | (ExaminedEffect trial iot_effect) |
| A03 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Increased chimerism) $tv) | (Increased chimerism) |
| A05 | malformed_executed_query | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, malformed_executed_query, proof_answer_evidence_miss, nonpositive_proof | (: $prf (ReducesMACE tirzepatide}) $tv) | (ReducesMACE tirzepatide}) |
| A07 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Role gut_microbiome) $tv) | (Role gut_microbiome) |
| A08 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Understood contribution microbiota physiology) $tv) | (Understood contribution microbiota physiology) |
| A10 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (DeterminedRiskFactors study) $tv) | (DeterminedRiskFactors study) |
| A11 | proof_answer_evidence_miss | well_aligned | 1 | original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (HasCondition patient severe_covid) $tv) | (HasCondition patient severe_covid) |
| A12 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (ControlSurfaceWettabilityAndRoughness study surface) $tv) | (ControlSurfaceWettabilityAndRoughness study surface) |
| A13 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (EnablesComparison device) $tv) | (EnablesComparison device) |
| A14 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Analyze mppt_data) $tv) | (Analyze mppt_data) |
| A15 | proof_answer_evidence_miss | well_aligned | 1 | original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (GatherInsights diverse_range_of_stakeholder) $tv) | (GatherInsights diverse_range_of_stakeholder) |
| S01 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Reported sodium_heparin additive_strategy) $tv) | (Reported sodium_heparin additive_strategy) |
| S03 | proof_answer_evidence_miss | weakly_aligned | 1 | weak_alignment, fallback_selected, executed_query_changed, original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Used semaglutide type_diabetes) $tv) | (Used semaglutide type_diabetes) |
| S06 | proof_answer_evidence_miss | well_aligned | 1 | original_query_answer_symbol_miss, executed_query_answer_symbol_miss, proof_answer_evidence_miss | (: $prf (Improves diabetes_control logbook) $tv) | (Improves diabetes_control logbook) |

## Counts

- `malformed_executed_query`: 1
- `proof_answer_evidence_miss`: 13

## Evidence Flags

- `executed_query_answer_symbol_miss`: 13
- `executed_query_changed`: 11
- `fallback_selected`: 11
- `malformed_executed_query`: 1
- `nonpositive_proof`: 1
- `original_query_answer_symbol_miss`: 13
- `proof_answer_evidence_miss`: 14
- `verdict_mismatch`: 1
- `weak_alignment`: 11
