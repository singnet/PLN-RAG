# Coreference Manual Audit

Evaluation date: 2026-09-13

## Scope

This audit covers the seven Stress25 documents changed by both FCoref and
LingMess: `A01`, `A04`, `A06`, `A08`, `A09`, `A10`, and `A11`. It compares
each resolved document, generated atoms, generated query, and proof against the
source text and its `expected_focus` metadata.

The judgments are manual rather than formal ground truth. Both A/B runs were
cumulative and used a non-deterministic parser LM, so a downstream difference
is attributed to coreference only when the evidence is unusually direct.

## Replacement Quality

| Case | LingMess | FCoref | Main observation |
|---|---|---|---|
| A01 | Correct referent | Correct referent | Both expand `their characteristics` to the 2,861 participants, but create a cumbersome possessive. The edit is unrelated to the queried intervention result. |
| A04 | Correct | Correct | `it` becomes `semaglutide at usual doses`. This is the cleanest and most useful edit. |
| A06 | 3 correct, 1 incorrect | 2 acceptable, 2 incorrect or malformed | Both over-expand possessives. The blister-pack `their` is incorrectly assigned to all four materials. FCoref also produces `serafilcon A gradually's reserve`. |
| A08 | 2 correct | 2 correct | The community members and microbiota-derived molecules are appropriate antecedents. |
| A09 | 2 correct, 1 questionable, 1 incorrect | 3 acceptable, 1 incorrect | Both handle heparin sodium references, but confuse at least one device/result reference. Appositive replacements remain awkward. |
| A10 | 2 correct | 2 correct | Patient outcomes and parameter cut-off values receive appropriate antecedents. |
| A11 | Correct referent | Correct referent | Both insert the full phrase `114 patients with COVID-19 at the Jinyintan Hospital, Wuhan`, producing an excessively long possessive. |

LingMess produced approximately 12 correct, 2 incorrect, and 1 questionable
replacement. FCoref produced approximately 12 semantically acceptable and 3
materially incorrect or malformed replacements. Surface quality was weaker
than semantic accuracy for both models.

Twelve of the fifteen edits were possessive expansions. This is the dominant
risk: correct antecedent selection can still create text that is harder for the
parser to interpret.

## Downstream Audit

| Case | LingMess result | FCoref result | Manual conclusion |
|---|---|---|---|
| A01 | Proof remained; treatment proof better represented the logbook-outcome relation. | Proof remained with a different but relevant query. | Output quality varied, but the only rewrite occurs in the propensity-matching methods section. Coreference attribution is weak. |
| A04 | Gained a direct proof of superior HbA1c and weight reductions. | Lost a broad comparison proof. | LingMess is the strongest plausible gain. FCoref's loss is less meaningful because the baseline proof was generic and cumulative. |
| A06 | Gained a generic `Compared` proof. | Retained a generic comparison proof. | Neither proof establishes the expected wettability or friction comparison. These are proof-presence false positives. |
| A08 | Lost its proof despite two correct rewrites. | Lost its proof despite two correct rewrites. | Generated atoms and queries use incompatible predicate shapes. This is primarily parser alignment failure. |
| A09 | Lost a proof because a zero-argument conclusion was queried as unary. | Gained a source-grounded bridge-to-bonding proof. | FCoref `A09` is the strongest plausible gain. LingMess demonstrates an atom/query arity failure and includes an incorrect narrowing of device scope. |
| A10 | Retained a proof but no longer enumerated the requested biomarkers. | Lost its proof because a multi-argument atom could not satisfy a unary query. | Both treatments regress answer quality or schema alignment even though their coreference edits are correct. |
| A11 | Retained a generic severity proof using inherited state. | Retained a proof about `patient_191` inherited from A10, although A11 studies 114 patients. | Both results are contaminated by cumulative state. The FCoref proof is a clear cross-case false positive. |

Within the seven rewritten cases, LingMess exchanged two gains for two losses;
FCoref produced one gain and three losses. Proof counts alone obscure important
quality failures: A06 finds an irrelevant generic proof, A10 fails to answer
"which biomarkers", and A11 uses evidence from a different case.

## Patterns

1. Coreference accuracy does not imply parser improvement. Correct rewrites in
   `A08` and `A10` still lead to atom/query incompatibility.
2. Long possessive expansions are the primary rewrite weakness. They preserve
   references but often damage sentence fluency and occasionally broaden or
   narrow entity scope incorrectly.
3. Predicate and arity instability dominate downstream behavior. Examples
   include `Affects` versus `ImpactsInflammation`, `Comparison` versus
   `ComparisonResult`, and zero-, unary-, and multi-argument forms of the same
   concept.
4. Cumulative evaluation permits cross-case leakage. `A11` demonstrates a
   proof based on the previous case's 191-patient cohort.
5. The query itself is not coreference-resolved. Query changes come from the
   parser and its retrieved context, not direct query rewriting.
6. The isolated controls changed queries without any document rewrite, so the
   observed A/B transitions also contain ordinary LLM variation.

## Recommendation

Keep coreference disabled by default. LingMess remains the stronger resolver,
but neither model has demonstrated a reliable downstream benefit.

Before another broad A/B run:

1. Add an abstention rule for long or syntactically noisy possessive
   antecedents rather than inserting an entire noun phrase.
2. Build a small isolated suite from these seven cases with manually labeled
   expected answers, required entities, and acceptable proof predicates.
3. Score proof relevance and entity coverage, not only whether a proof exists.
4. Reject proofs whose provenance comes exclusively from earlier benchmark
   cases when evaluating current-case correctness.
5. Use deterministic recorded/replayed parser responses or repeated randomized
   trials before estimating a causal effect.
