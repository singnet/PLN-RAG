# Coreference Impact on PLN-RAG

Evaluation period: 2026-09-10 through 2026-09-14

## Executive Summary

We evaluated whether adding document-level coreference resolution improves the
quality of PLN-RAG's generated PLN, queries, and proof results. The comparison
covered two CPU-compatible models:

- FCoref, a 90.5M-parameter model.
- LingMess, a 590.0M-parameter model.

The work progressed from synthetic rewrite testing to full downstream runs, a
manual audit, and finally an isolated deterministic benchmark with explicit
proof-quality labels and exact source provenance.

The main result is that neither model has demonstrated a downstream quality
benefit for the current PLN-RAG pipeline. On the final deterministic Stress7
evaluation, the baseline and both coreference treatments passed 0 of 7 cases.
The baseline nevertheless had the highest average proof-quality score:

| Configuration | Cases passed | Average score |
|---|---:|---:|
| No coreference | 0/7 | 0.148810 |
| FCoref | 0/7 | 0.059524 |
| LingMess | 0/7 | 0.059524 |

FCoref and LingMess produced the same proof transitions: one raw proof gain,
three raw proof losses, and no correctness gain. Their only gained proof, in
case A10, was grounded in the current document but did not express the required
relationship between the listed biomarkers and COVID-19 severity. It therefore
failed the semantic evaluation.

The models can resolve pronouns accurately, especially LingMess, but resolver
accuracy does not automatically improve the downstream system. The larger
problem is currently the stability and alignment of the generated PLN schema:
predicate names, argument counts, generated facts, generated rules, and queries
frequently do not line up well enough for the reasoner to prove the requested
claim. Rewriting a document can cause the parser to choose a different schema,
even when the rewrite is linguistically correct.

Coreference should remain disabled by default. LingMess is more accurate on the
synthetic rewrite benchmark but is slower and did not outperform FCoref or the
baseline downstream. FCoref is faster but likewise showed no downstream gain.

## Research Question

The experiment was designed to answer a narrower question than whether a model
can resolve pronouns:

> Does resolving references in a document before parsing improve PLN-RAG's
> ability to produce a relevant, current-document-grounded proof for the user's
> question?

This distinction is important. A coreference edit can be linguistically
correct and still be neutral or harmful downstream. For example, replacing a
pronoun with a long noun phrase may preserve its referent but make the sentence
harder to parse. A rewrite can also lead the LLM parser to invent a different
predicate or argument structure that no longer matches its generated query.

The evaluation therefore considered four separate properties:

1. Rewrite correctness: did the resolver identify the intended antecedent?
2. Rewrite safety: did the replacement preserve readable and structurally safe
   text?
3. Downstream quality: did the resulting proof contain the required entities
   and meaning?
4. Evidence validity: did the proof actually depend on atoms from the current
   benchmark case rather than query scaffolding, persisted state, background
   knowledge, or another case?

## Integration Design

Coreference was added as optional document preprocessing. The resolver runs
once on the complete input document before chunking. The resolved document is
then passed through the existing chunker, parser, vector store, and reasoner.
The user query is not coreference-resolved or rewritten.

The integration has the following constraints:

- CPU is the only supported device for this experiment.
- Coreference remains disabled by default.
- Supported backends are `none`, `fcoref`, and `lingmess`.
- The original text is retained when a model fails and fail-open behavior is
  enabled.
- Resolution status, model name, elapsed time, replacements, diagnostics, and
  raw pair logits are available for benchmark inspection.
- Raw pair logits are reported for observability. They are not treated as
  calibrated probabilities and are not filtered by
  `COREFERENCE_MIN_CONFIDENCE`.
- Ambiguous `her` references are never rewritten.

The main implementation and evaluation surfaces are:

- `core/coreference.py`: model adapters, rewrite logic, abstention, and
  diagnostics.
- `core/service.py`: document-level preprocessing before chunking.
- `config.py`: disabled-by-default CPU configuration.
- `scripts/evaluate_coreference.py`: deterministic rewrite evaluation.
- `benchmark_parsers.py`: downstream parser and reasoner benchmark reporting.
- `scripts/run_paired_benchmark.py`: sequential multi-backend runs.
- `scripts/evaluate_coreference_benchmark.py`: labeled proof-quality scoring.
- `scripts/compare_coreference_benchmarks.py`: paired transitions and timing
  warnings.

## Phase 1: Synthetic Rewrite Evaluation

The first evaluation isolated the resolver and deterministic rewrite layer from
the LLM parser and reasoner. It used 36 synthetic cases from
`data/benchmarks/coreference_pronouns_v1.json`.

The cases covered singular, plural, neuter, multi-cluster, possessive, and
biomedical references. The policy rewrote only supported unambiguous pronouns,
abstained on ambiguous `her`, and evaluated exact resolved text and individual
replacement spans.

The fixture backend received gold clusters and achieved 36/36 exact matches and
replacement F1 of 1.000. This validated that the deterministic rewriter and
evaluator behaved correctly independently of model predictions.

### Synthetic Results

| Backend | Parameters | Exact matches | Change accuracy | Replacement precision | Replacement recall | Replacement F1 | Median CPU latency | p95 CPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixture | N/A | 36/36 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | N/A |
| FCoref | 90.5M | 24/36 | 0.556 | 0.917 | 0.550 | 0.687 | 0.396s | 0.443s |
| LingMess | 590.0M | 36/36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.420s | 1.474s |

LingMess was the stronger resolver on this suite. FCoref primarily missed
neuter, plural, multi-cluster, and biomedical links. LingMess was about 3.6
times slower per short synthetic document.

These results established that useful coreference was technically possible,
but they did not answer the downstream research question. The synthetic inputs
were short, the expected replacements were directly labeled, and no LLM parser
or proof search was involved.

## Phase 2: Stress25 Rewrite Audit

Both models were next run over the 25 Stress25 documents without invoking the
LLM parser. This measured how often real benchmark documents would be changed.

| Backend | Documents changed | Replacements | Failed open |
|---|---:|---:|---:|
| FCoref | 7/25 | 15 | 0 |
| LingMess | 7/25 | 15 | 0 |

Both models changed the same seven cases: A01, A04, A06, A08, A09, A10, and
A11. Thirteen of the fifteen replacements were effectively shared. The main
differences occurred in A06 and A09, which contain long biomedical or technical
noun phrases.

Twelve of the fifteen edits were possessive expansions. This became the main
safety concern. A model could identify a semantically plausible antecedent but
produce constructions such as a long cohort description followed by `'s`, or
attach a possessive to a list, appositive, comparative group, or clause. Such
text is often less suitable for the parser than the original pronoun.

This stage had no downstream correctness labels. It measured model activity and
surface behavior, not whether PLN-RAG answers improved.

## Phase 3: Initial Cumulative Downstream Runs

The next experiments ran the complete canonical PLN parser, atom ingestion,
query generation, and proof search over all 25 Stress25 cases. The runs used
cumulative mode, so atoms from earlier cases remained available to later cases.

### LingMess Cumulative Run

Pair ID: `e8517efceacf`

| Metric | No coreference | LingMess |
|---|---:|---:|
| Proofs found | 17/25 | 17/25 |
| Weakly aligned queries | 2 | 1 |
| Documents changed | 0 | 7 |
| Replacements | 0 | 15 |
| Mean total latency | 20.342s | 24.281s |
| Median total latency | 18.558s | 22.717s |

Raw proof transitions were balanced:

- 5 cases changed from no proof to proof.
- 5 cases changed from proof to no proof.
- 12 cases retained a proof.
- 3 cases retained no proof.

### FCoref Cumulative Run

Pair ID: `4374fdb30e48`

| Metric | No coreference | FCoref |
|---|---:|---:|
| Proofs found | 20/25 | 16/25 |
| Weakly aligned queries | 2 | 2 |
| Documents changed | 0 | 7 |
| Replacements | 0 | 15 |
| Mean total latency | 26.445s | 26.650s |
| Median total latency | 24.458s | 25.381s |

Raw proof transitions favored the baseline:

- 2 cases changed from no proof to proof.
- 6 cases changed from proof to no proof.
- 14 cases retained a proof.
- 3 cases retained no proof.

### Why These Results Were Not Conclusive

The two independent baseline runs found different proof totals, 17 and 20,
despite both having coreference disabled. This was direct evidence that raw
proof-count differences could not be attributed to coreference alone.

The parser LM had caching disabled and no reproducibility seed. In isolated
negative controls, queries changed between baseline and treatment even when the
coreference resolver made no document changes. The cumulative atomspace also
allowed later cases to prove claims using atoms introduced by earlier cases.

Stress25 had metadata describing the expected focus, but it did not yet have
formal proof-quality labels. A syntactically successful proof could therefore
be counted even if it was generic, incomplete, or unrelated to the requested
answer.

The initial runs were useful exploratory evidence, but not clean causal tests.

## Phase 4: Manual Audit

We manually examined the seven documents changed by both models. The audit
compared resolved text, generated atoms, generated queries, proofs, and each
case's expected focus.

### Rewrite Findings

| Case | LingMess assessment | FCoref assessment | Main observation |
|---|---|---|---|
| A01 | Correct referent | Correct referent | Both expanded `their characteristics` to the 2,861 participants. The result was cumbersome and unrelated to the queried intervention outcome. |
| A04 | Correct | Correct | Replacing `it` with `semaglutide at usual doses` was the cleanest potentially useful edit. |
| A06 | 3 correct, 1 incorrect | 2 acceptable, 2 incorrect or malformed | Both over-expanded possessives. FCoref also produced a malformed `serafilcon A gradually's reserve` phrase. |
| A08 | 2 correct | 2 correct | Community-member and microbiota-derived-molecule references were assigned plausible antecedents. |
| A09 | 2 correct, 1 questionable, 1 incorrect | 3 acceptable, 1 incorrect | Both handled some heparin sodium references but confused at least one device or result reference. |
| A10 | 2 correct | 2 correct | Patient outcomes and parameter cut-off references received plausible antecedents. |
| A11 | Correct referent | Correct referent | Both inserted a full 114-patient cohort phrase, creating an excessively long possessive. |

The audit estimated about 12 correct, 2 incorrect, and 1 questionable LingMess
replacement. FCoref produced about 12 semantically acceptable replacements and
3 materially incorrect or malformed replacements. Surface quality was weaker
than antecedent accuracy for both models.

### Downstream Findings

The manual review showed why proof presence was an inadequate metric:

- A04 produced the strongest plausible LingMess gain in the cumulative run, but
  the result could not be isolated from parser variation.
- A06 found generic comparison proofs that did not establish the requested
  wettability and friction findings.
- A08 contained correct rewrites but lost proofs because generated predicates
  and query shapes were incompatible.
- A09 exposed zero-argument versus unary predicate mismatches and incorrect
  narrowing of device scope.
- A10 either stopped enumerating the requested biomarkers or generated an atom
  whose argument structure did not match the query.
- A11 demonstrated cumulative contamination. One proof referred to the
  191-patient cohort from A10 even though A11 described 114 patients.

The recurring failures were predicate instability, arity mismatch, incomplete
entity coverage, generic proof targets, and cross-case state leakage. Correct
coreference did not solve these issues and sometimes triggered a less compatible
parse.

## Phase 5: Safety and Evaluation Hardening

The manual audit led to changes in both runtime policy and experimental design.

### Structural Possessive Abstention

The rewriter now rejects possessive substitutions whose antecedents have risky
structure. There is deliberately no fixed word or character limit. Decisions
are based on syntax-like surface risks rather than phrase length alone.

The current policy abstains for:

- Internal punctuation or list structure.
- Quantified phrases such as `a total of`.
- Comparative groups containing `other`.
- Relative clauses.
- Multiple prepositional attachments.
- Terminal discourse adverbs such as `gradually` or `respectively`.

Abstentions preserve the original pronoun and emit reason-specific
`unsafe_possessive_antecedent` diagnostics. Plain non-possessive pronouns remain
eligible for rewriting.

### Isolated Execution

The authoritative downstream evaluation uses isolated mode. Each case begins
with reset benchmark state, preventing an A11 proof from silently depending on
A10 document atoms. Cumulative mode remains useful for diagnosing leakage and
long-running behavior but is not accepted by the labeled scorer.

### Exact Proof Provenance

Proof provenance is recorded when atoms are inserted into the reasoner. Sources
are classified as:

- Current or foreign benchmark documents, with case ID and document index.
- Query-support atoms.
- Background atoms.
- Persisted atoms loaded from disk.

The reasoner extracts named atoms from each returned proof and maps them to the
registered source records. It reports current-case grounding, foreign case IDs,
query-support-only evidence, unknown evidence, and ambiguous atom names.

Statement hashes prevent two different definitions that share an atom name from
being accepted as unambiguous current-case evidence. This avoids granting
semantic credit based only on a reused label.

### Stress7 Labels and Scoring

The seven affected Stress25 cases received a labels-only sidecar at
`data/benchmarks/stress7_coreference_labels_v1.json`. Each label is bound to the
source case with a SHA-256 case hash and defines:

- Required entity groups and accepted aliases.
- Required semantic groups and accepted aliases.
- Minimum entity and semantic coverage.
- Forbidden evidence from another case where applicable.

Only individually current-case-grounded proof traces contribute entity or
semantic credit. Proof metadata, foreign traces, and query-support-only traces
cannot make a result pass.

For each case, the score is:

```text
(entity coverage + semantic coverage) / 2
```

The default threshold is 0.8. Passing additionally requires a proof, the
specified minimum number of entity and semantic groups, current-case grounding,
and no forbidden-group match. The threshold alone is therefore not enough to
pass.

### Deterministic LM Cassettes

The DSPy LM was wrapped with capture and strict replay support. Requests are
represented in the cassette by SHA-256 hashes rather than raw prompt bodies.
Responses are stored in occurrence pools, while each backend receives a stable
ordered scope: `none`, `fcoref`, or `lingmess`.

Replay mode enforces:

- Exact request hashes.
- Exact call order.
- Exact purpose labels for parser and answer-generation calls.
- Exact repeated-request occurrence indexes.
- No provider fallback.
- No missing or unconsumed calls.

This separates coreference-induced prompt changes from ordinary provider
variation. Identical parser requests receive captured identical responses,
while a document rewrite that changes the request receives its own recorded
response.

## Phase 6: Final Deterministic Stress7 Experiment

The final experiment compared three isolated arms:

- Baseline with coreference disabled.
- FCoref on CPU with structural abstention.
- LingMess on CPU with structural abstention.

Configuration:

| Setting | Value |
|---|---|
| Suite | `stress25_v1` |
| Selected cases | A01, A04, A06, A08, A09, A10, A11 |
| Parser | `canonical_pln` |
| Mode | Isolated |
| Pair ID | `stress7-structural-v1` |
| Score threshold | 0.8 |
| Coreference device | CPU |
| Query rewriting | Disabled |
| Answer generation | Disabled |

### Rewrite Activity After Safeguards

| Case | FCoref replacements | LingMess replacements | Safety result |
|---|---:|---:|---|
| A01 | 0 | 0 | Quantified possessive antecedent rejected. |
| A04 | 1 | 1 | Rewrite allowed. |
| A06 | 0 | 0 | Punctuation, comparative-group, and terminal-adverb risks rejected. |
| A08 | 2 | 2 | Rewrites allowed. |
| A09 | 3 | 2 | Some rewrites allowed; unsafe possessive or clause candidates rejected. |
| A10 | 2 | 2 | Rewrites allowed. |
| A11 | 0 | 0 | Punctuated long cohort antecedent rejected. |
| **Total** | **8** | **7** | **4 of 7 documents changed by each model.** |

The safeguards eliminated the most problematic A01, A06, and A11 possessive
expansions. They intentionally did not suppress all A09 activity because safe
plain-pronoun replacements remained eligible.

### Proof and Quality Results

| Metric | No coreference | FCoref | LingMess |
|---|---:|---:|---:|
| Raw proofs found | 3/7 | 1/7 | 1/7 |
| Quality passes | 0/7 | 0/7 | 0/7 |
| Average quality score | 0.148810 | 0.059524 | 0.059524 |
| Documents rewritten | 0/7 | 4/7 | 4/7 |
| Query changes versus baseline | N/A | 4 | 4 |
| Query changes without a rewrite | N/A | 0 | 0 |

Both model comparisons had the same raw proof transitions:

| Transition | FCoref | LingMess |
|---|---:|---:|
| No proof to proof | 1 | 1 |
| Proof to no proof | 3 | 3 |
| Proof remained | 0 | 0 |
| No proof remained | 3 | 3 |

Both quality comparisons had seven `unchanged_fail` transitions. Neither model
turned a failing baseline result into a passing result.

### Case-Level Results

| Case | Baseline | FCoref | LingMess | Interpretation |
|---|---|---|---|---|
| A01 | No proof, score 0 | No proof, score 0 | No proof, score 0 | Unsafe quantified possessive was correctly left unchanged, but the parser still queried `diabete` rather than the logbook-to-control result. |
| A04 | Grounded proof, score 0.25 | No proof, score 0 | No proof, score 0 | The baseline mentioned semaglutide but omitted comparator and outcome semantics. Both rewrites changed the parse and lost even that incomplete proof. |
| A06 | No proof, score 0 | No proof, score 0 | No proof, score 0 | Structural abstention prevented known malformed rewrites. Parser output still did not support both wettability and friction findings. |
| A08 | Grounded proof, score 0.666667 | No proof, score 0 | No proof, score 0 | The baseline captured inflammation semantics but omitted microbiota and mechanism entities. Both treatments changed predicate/query structure and lost the proof. |
| A09 | Grounded proof, score 0.125 | No proof, score 0 | No proof, score 0 | The baseline proof was generic and incomplete. Both treatments generated more specific queries but no matching proof. |
| A10 | No proof, score 0 | Grounded proof, score 0.416667 | Grounded proof, score 0.416667 | Both treatments recovered five biomarker groups but omitted COVID severity and the predictive relationship, so the apparent proof gain was not a correct answer. |
| A11 | No proof, score 0 | No proof, score 0 | No proof, score 0 | Unsafe cohort possessive was rejected and isolated mode prevented the earlier A10 evidence leakage, but no valid severity-prediction proof was produced. |

The strongest baseline result was A08 at 0.666667. It matched the required
inflammation semantics but only one of three required entity groups. The
strongest treatment result was A10 at 0.416667. It matched five of six entity
groups but none of the required prediction semantics.

These examples demonstrate why a raw proof count can be misleading. The only
treatment proof gain looked substantial by count but failed to answer the
question represented by the labels.

## Determinism Verification

The captured Stress7 cassette was replayed twice with an intentionally invalid
OpenAI API key. This ensures an accidental provider call could not silently
succeed.

### Cassette Consumption

| Backend scope | Capture calls | Replay 1 | Replay 2 |
|---|---:|---:|---:|
| `none` | 41 | 41 hits, 0 live, 0 misses, 0 unconsumed | 41 hits, 0 live, 0 misses, 0 unconsumed |
| `fcoref` | 42 | 42 hits, 0 live, 0 misses, 0 unconsumed | 42 hits, 0 live, 0 misses, 0 unconsumed |
| `lingmess` | 42 | 42 hits, 0 live, 0 misses, 0 unconsumed | 42 hits, 0 live, 0 misses, 0 unconsumed |

All six replayed reports were valid. Capture, replay 1, and replay 2 were
byte-equivalent after normalizing only expected volatile fields:

- Run and pair IDs.
- Atomspace and Qdrant collection names.
- Cassette mode and counters.
- Parser, retrieval, ingestion, reasoner, and coreference timing values.

Statements, generated queries, proof values, provenance, coreference decisions,
replacement counts, diagnostics, and benchmark summaries were identical.

An initial replay attempt correctly failed at the first request because it was
started with a different model and endpoint configuration. Replaying with the
original `.env` model and endpoint values, while overriding only the API key,
matched the captured hashes. This confirmed that the request hash protects the
complete LM configuration as well as prompt content.

The replay exercise also exposed and fixed a canonicalization edge case where
DSPy could pass a Pydantic model class during structured-output fallback. Model
classes are now represented as callables rather than incorrectly invoking an
unbound `model_dump()` method.

## Performance Results

### Resolver-Only Synthetic Latency

| Backend | Median | p95 |
|---|---:|---:|
| FCoref | 0.396s | 0.443s |
| LingMess | 1.420s | 1.474s |

### Cached Isolated Stress7 Coreference Duration

| Backend | Mean per document | Median per document | Total for 7 documents |
|---|---:|---:|---:|
| FCoref | 2.927s | 2.575s | 20.491s |
| LingMess | 5.323s | 4.295s | 37.259s |

LingMess was approximately 1.7 times slower by Stress7 median and has about 6.5
times as many parameters. FCoref is the more practical CPU option when latency
and memory matter, but its lower cost did not correspond to a quality gain.

These Stress7 values include isolated benchmark setup effects and should not be
treated as warm long-running API latency. Fresh replay containers also incurred
large first-case model retrieval and initialization costs. Those cold download
times were excluded from the table because they measure cache state and network
availability rather than normal inference.

Provider-inclusive end-to-end timing from the capture run is not comparable
between arms. The shared response pool caused asymmetric live LM use:

- Baseline: 41 live calls.
- FCoref: 14 live calls and 28 pool hits.
- LingMess: 3 live calls and 39 pool hits.

This explains why treatment parse and total times sometimes appeared lower even
after adding CPU coreference work. The comparison artifacts now expose
`provider_inclusive_timing_valid: false` and the warning
`capture_timing_mixes_live_and_replayed_lm_calls`. Semantic comparisons remain
valid because matching requests receive matching captured responses, but the
capture's total latency deltas are not production latency estimates.

## Engineering Verification

The experiment required more than adding model calls. The supporting work
included:

- CPU-only Torch packaging and successful application-image rebuild.
- Compatibility handling for LingMess, Longformer, and Transformers 5.
- Optional disabled-by-default service configuration.
- Document-level resolution before chunking.
- Fail-open behavior and structured diagnostics.
- Raw pair-logit reporting without confidence filtering.
- Structural possessive abstention.
- Deterministic synthetic rewrite fixtures.
- Paired benchmark orchestration for all three backends.
- Isolated benchmark execution.
- Stress7 correctness labels and hash validation.
- Model-free proof scoring.
- Exact atom insertion and proof provenance.
- Statement-hash collision detection.
- Strict LM capture and replay with atomic cassette writes.
- Logical comparison and separate timing-validity reporting.

The final test run passed 76 tests. It included resolver, service integration,
benchmark, scorer, comparator, cassette, and provenance tests. `git diff
--check` also passed.

## What We Learned

### 1. Resolver Accuracy Is Not Downstream Accuracy

LingMess achieved perfect exact matching on the synthetic resolver suite, yet it
did not improve any Stress7 case. Correct antecedents can still destabilize the
LLM parser or produce awkward text.

### 2. Possessive Expansion Is Disproportionately Risky

Most real-document replacements were possessive. Long or structurally complex
noun phrases followed by `'s` often reduced fluency and sometimes changed
scope. Structural abstention removed the clearest failures without imposing an
arbitrary phrase-length limit.

### 3. PLN Schema Alignment Is the Dominant Bottleneck

The parser often generates statements and queries that differ in predicate name
or arity. Coreference can change which schema the parser chooses but cannot make
an internally incompatible schema provable. A08, A09, and A10 are representative
examples.

### 4. Proof Presence Is Too Weak a Success Metric

A proof can mention only one requested entity, prove a generic relation, omit
the requested outcome, or come from another case. Entity coverage, semantic
coverage, and source provenance materially changed the interpretation of the
results.

### 5. Isolation and Provenance Are Both Necessary

Resetting the atomspace prevents intentional cumulative state, but provenance
is still needed to verify each proof trace. Exact insertion tracking also makes
persisted, background, query-support, foreign, and ambiguous evidence visible.

### 6. Deterministic LM Control Is Essential for Causal Attribution

The earlier negative controls changed queries without any coreference rewrite.
The final cassette-controlled run had four query changes for each model, all in
the four cases whose documents were actually rewritten. There were zero query
changes without a rewrite. This makes the final logical transitions much more
credible than the earlier cumulative transitions.

### 7. Deterministic Correctness and Live Timing Need Different Controls

Replay is suitable for proving logical repeatability and preventing provider
fallback. Capture with shared response reuse is suitable for constructing a
controlled logical response pool. Neither should be confused with a balanced
provider-inclusive latency benchmark.

## Limitations

- Stress7 contains only the seven Stress25 documents changed by both original
  model runs. It is targeted and useful for regression analysis but is not a
  representative sample of all PLN-RAG inputs.
- All three configurations passed 0/7, showing that the current parser and
  reasoner struggled with the strict labels. The lower treatment averages are
  evidence against enabling coreference, but this small result does not prove
  that coreference can never help.
- The label scorer uses explicit alias matching after token normalization. It is
  deterministic and auditable but does not recognize every possible semantic
  paraphrase.
- The experiment evaluates proof content rather than a separately generated
  natural-language answer because answer generation was disabled.
- The final deterministic run used one captured response pool. Two replays
  establish repeatability of that pool, not statistical robustness across
  independent provider samples.
- Isolated benchmark latency includes repeated service and model initialization
  behavior and differs from a warm production process.
- The original Stress25 material includes cases whose source text must be
  handled according to its copyright and manual-copy metadata. Real cassettes
  remain ignored because generated responses can contain source-derived text.

## Recommendation

Keep coreference disabled by default.

The evidence supports the following conclusions:

- LingMess is the more accurate standalone resolver on the synthetic suite.
- FCoref is substantially smaller and faster.
- Neither model improved labeled downstream proof quality.
- Both models reduced the average Stress7 score relative to no coreference.
- The current parser/query schema alignment problem is more important than
  unresolved pronouns for these cases.
- Structural abstention should remain in place for any future coreference
  experiments.

Enabling either backend globally would add CPU cost and introduce rewrite risk
without demonstrated benefit.

## Recommended Next Experiments

1. Stabilize parser predicates and argument shapes before repeating broad
   coreference A/B tests.
2. Expand labels beyond Stress7 to include unaffected cases and more cases where
   pronoun resolution is genuinely necessary to answer the question.
3. Add answer-level correctness labels in addition to proof-level labels.
4. Collect several independent cassettes with randomized backend order to
   estimate sensitivity to provider samples.
5. Separate quality experiments from performance experiments. Use replay for
   logical control and balanced live runs for provider-inclusive timing.
6. Measure warm-service CPU latency with each coreference model loaded once and
   reused across documents.
7. Continue recording exact proof provenance and reject foreign-only,
   query-support-only, unknown, or ambiguous evidence in benchmark scoring.
8. Investigate whether targeted coreference activation can be limited to
   documents with unresolved non-possessive pronouns rather than preprocessing
   every document.

## Commands

Resolver-only evaluation:

```bash
python3 scripts/evaluate_coreference.py \
  --backend fixture \
  --require-effective \
  --output /tmp/coref-fixture.json

python3 scripts/evaluate_coreference.py \
  --backend fcoref \
  --require-effective \
  --output /tmp/coref-fcoref.json

python3 scripts/evaluate_coreference.py \
  --backend lingmess \
  --require-effective \
  --output /tmp/coref-lingmess.json
```

Stress7 capture:

```bash
python3 scripts/run_paired_benchmark.py \
  --mode isolated \
  --suite-file data/benchmarks/stress25_v1.json \
  --case-ids A01 --case-ids A04 --case-ids A06 --case-ids A08 \
  --case-ids A09 --case-ids A10 --case-ids A11 \
  --parsers canonical_pln \
  --backends none fcoref lingmess \
  --pair-id stress7-structural-v1 \
  --llm-cassette-mode capture \
  --llm-cassette-path data/benchmarks/cassettes/stress7.json
```

Replace `capture` with `replay` and use the same cassette, model, and endpoint
configuration for strict deterministic replay.

Labeled scoring and comparison:

```bash
python3 scripts/evaluate_coreference_benchmark.py REPORT.json \
  --labels data/benchmarks/stress7_coreference_labels_v1.json

python3 scripts/compare_coreference_benchmarks.py \
  BASELINE_REPORT.json TREATMENT_REPORT.json \
  --labels data/benchmarks/stress7_coreference_labels_v1.json
```

## Result Artifacts

Repository supporting files:

- `docs/benchmarks/coreference_cpu_comparison.md`
- `docs/benchmarks/coreference_manual_audit.md`
- `docs/benchmarks/stress25_coreference_e8517efceacf.md`
- `docs/benchmarks/stress25_fcoref_4374fdb30e48.md`
- `data/benchmarks/coreference_pronouns_v1.json`
- `data/benchmarks/stress7_coreference_labels_v1.json`

Committed experiment evidence:

- `data/benchmarks/evidence/coreference_2026-09/README.md`
- `data/benchmarks/evidence/coreference_2026-09/stress25_cumulative/`
- `data/benchmarks/evidence/coreference_2026-09/negative_controls/`
- `data/benchmarks/evidence/coreference_2026-09/stress7/`
- `data/benchmarks/evidence/coreference_2026-09/SHA256SUMS`

The final Stress7 directory contains capture reports, two complete sets of
successful replay reports, labeled evaluations, comparisons, paired manifests,
and normalized logical-equivalence hashes. The LM cassette itself is
intentionally not tracked because it contains generated provider responses and
can contain source-derived model output.
