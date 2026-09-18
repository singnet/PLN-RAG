# Query Execution Policy Comparison

Source report SHA-256: `5f9e810caeaea6b349bb1e457cee68978ae7bae81d25dbb439063528ff33b48f`  
Generation tape SHA-256: `9cac194f1f7534ee3b1130f26d66b4fdd76b8873458900cf8eff74cfa5920fe2`

| Policy | Proofs | Automatic correct | Avg candidates tried |
| --- | ---: | ---: | ---: |
| `ranked_first_proof` | 22/25 | 8/25 | 1.1600 |
| `ranked_first_only` | 22/25 | 8/25 | 1.0000 |
| `original_only` | 15/25 | 1/25 | 1.0000 |

## Manual Adjudication

Status: `first_blind_pass`
Scope: 20 changed cases from the hash-bound blind packet. The JSON also records five unchanged inherited labels, which are excluded here.

| Policy | Adequate | Qualified | Insufficient | Incorrect | Strict | Usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ranked_first_proof` | 0 | 7 | 10 | 3 | 0/20 | 7/20 |
| `ranked_first_only` | 0 | 7 | 10 | 3 | 0/20 | 7/20 |
| `original_only` | 3 | 3 | 5 | 9 | 3/20 | 6/20 |

| Policy | Grader TP | Grader FP | Grader FN | Grader TN |
| --- | ---: | ---: | ---: | ---: |
| `ranked_first_proof` | 6 | 2 | 1 | 11 |
| `ranked_first_only` | 6 | 2 | 1 | 11 |
| `original_only` | 1 | 0 | 5 | 14 |

Status: `second_blind_pass`
Scope: 20 changed cases from the hash-bound blind packet. The JSON also records five unchanged inherited labels, which are excluded here.

| Policy | Adequate | Qualified | Insufficient | Incorrect | Strict | Usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ranked_first_proof` | 4 | 3 | 11 | 2 | 4/20 | 7/20 |
| `ranked_first_only` | 4 | 3 | 11 | 2 | 4/20 | 7/20 |
| `original_only` | 4 | 2 | 4 | 10 | 4/20 | 6/20 |

| Policy | Grader TP | Grader FP | Grader FN | Grader TN |
| --- | ---: | ---: | ---: | ---: |
| `ranked_first_proof` | 6 | 2 | 1 | 11 |
| `ranked_first_only` | 6 | 2 | 1 | 11 |
| `original_only` | 1 | 0 | 5 | 14 |

| Policy | Exact label agreement | Usable/not-usable agreement |
| --- | ---: | ---: |
| `ranked_first_proof` | 15/20 | 18/20 |
| `ranked_first_only` | 15/20 | 18/20 |
| `original_only` | 18/20 | 20/20 |

Changed outputs requiring blind adjudication: **20**

| Case | Changed | Current query | Original-only query |
| --- | --- | --- | --- |
| A01 | yes | (: $prf (Achieve optimal_glycemic_control) $tv) | (: $prf (Improve glycemic_control) $tv) |
| A02 | yes | (: $prf (ExaminedEffect trial iot_effect) $tv) | (: $prf (HbA1CChange trial over_52_week) $tv) |
| A03 | yes | (: $prf (Increased chimerism) $tv) | (: $prf (StableHemoglobinProduction $cells) $tv) |
| A04 | yes | (: $prf (Decreases hba1c) $tv) | (: $prf (CompareEffectiveness semaglutide other_glp1_receptor_agonist) $tv) |
| A05 | yes | (: $prf (ReducesMACE tirzepatide}) $tv) | (: $prf (ReducesMACE {liraglutide,) $tv) |
| A06 | yes | (: $prf (Measured coefficient friction) $tv) | (: $prf (EvaluateWettabilityFriction serafilcon_a) $tv) |
| A07 | yes | (: $prf (Role gut_microbiome) $tv) | (: $prf (Unclear gut_microbiome crc) $tv) |
| A08 | yes | (: $prf (Understood contribution microbiota physiology) $tv) | (: $prf (Effects inflammation microbiota) $tv) |
| A09 | yes | (: $prf (BridgeLayer heparin_sodium) $tv) | (: $prf (EnhanceBonding heparin_sodium) $tv) |
| A10 | yes | (: $prf (DeterminedRiskFactors study) $tv) | (: $prf (PredictSeverity $patient) $tv) |
| A12 | yes | (: $prf (ControlSurfaceWettabilityAndRoughness study surface) $tv) | (: $prf (HydrophilicHydrophobicDistinction $contact_angle) $tv) |
| A13 | yes | (: $prf (EnablesComparison device) $tv) | (: $prf (DifficultiesInComparingStabilityData $device) $tv) |
| A14 | yes | (: $prf (Analyze mppt_data) $tv) | (: $prf (HigherEfficiencyAlsoHigherStability $efficiency) $tv) |
| E01 | yes | (: $prf (Covered venus highly_reflective_cloud) $tv) | (: $prf (Brighter venus) $tv) |
| E02 | yes | (: $prf (InSeason june summer) $tv) | (: $prf (Greatest daylight $month) $tv) |
| E03 | yes | (: $prf (Closest sun earth) $tv) | (: $prf (WillAppearBrighter sun) $tv) |
| S01 | yes | (: $prf (Reported sodium_heparin additive_strategy) $tv) | (: $prf (Links sodium_heparin $mechanism) $tv) |
| S03 | yes | (: $prf (Used semaglutide type_diabetes) $tv) | (: $prf (ProducesStrongGlycemicControl semaglutide relative_to $other_agent) $tv) |
| S04 | yes | (: $prf (MeasurementOf contact_angle surface_wettability) $tv) | (: $prf (MeasurementOf contact_angle surface_wettability) $tv) |
| S05 | yes | (: $prf (EvidenceBase evidence mixed) $tv) | (: $prf (Links many_study gut_microbiome_composition colorectal_cancer_risk) $tv) |
