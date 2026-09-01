# RAGForge AI Experiment Log

This file is the permanent record of RAG experiments run on this project.
Historical results are not overwritten — when a configuration is revisited,
a new dated entry is added instead of editing the old one.

Evaluation set (unless noted otherwise): `evaluation/questions_chunk100.json`,
10 hand-labeled questions over `evaluation/corpus.pdf`, plus
`evaluation/negative_questions.json` (5 out-of-document questions) for
rejection testing.

---

## Baseline — Retrieval

**Configuration**
- Chunking: 100 words, 20 word overlap (`evaluation/build_chunking_experiment.py`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized
- Vector store: FAISS `IndexFlatIP`
- Retrieval: plain top-k, no reranking, no threshold applied
- Script: `evaluation/evaluate_retrieval.py`
- Result file: `evaluation/results.json`

**Metrics**

| Metric | Value |
|---|---|
| Recall@1 | 0.500 |
| Recall@3 | 0.900 |
| Recall@5 | 0.900 |
| MRR@1 | 0.500 |
| MRR@3 | 0.683 |
| MRR@5 | 0.683 |

**Conclusion**: solid recall at k=3+ on the current 10-question set. Recall@1
is weak, meaning the top single chunk isn't reliably the right one — this is
what motivated the threshold and reranking experiments below.

---

## Experiment 1 — Retrieval Threshold Calibration

**Configuration**: same retrieval as baseline, sweeping `min_score` over
`[0.00, 0.10, 0.20, 0.30, 0.35, 0.40]`.
Script: `evaluation/evaluate_thresholds.py` → `evaluation/threshold_results.json`.
Negative-query rejection measured separately with
`evaluation/evaluate_negative_thresholds.py` → `evaluation/negative_threshold_results.json`.

**Metrics**

| Threshold | Recall@3 | Positive queries rejected | Negative-query rejection |
|---|---|---|---|
| 0.20 | 0.900 | 0/10 | 4/5 (80%) |
| 0.25 | 0.900 | 0/10 | 5/5 (100%) |
| 0.30 | 0.900 | 0/10 | 5/5 (100%) |
| 0.35 | 0.900 | 0/10 | 5/5 (100%) |
| 0.40 | 0.700 | 2/10 | 5/5 (100%) |

**Decision**: threshold set to **0.25**.

Reasoning: it preserves Recall@3 on the current 10-question positive set,
rejects all 5 tested negative/out-of-document queries, and is less
aggressive than 0.30/0.35 (smaller chance of rejecting a legitimate
in-document question as the corpus grows). This is an empirically selected
value on the current small validation set, not a universally optimal
threshold — it should be re-checked as the evaluation set grows.

**Production decision**: kept. `MIN_RELEVANCE_SCORE = 0.25` in `src/rag.py`.

---

## Experiment 2 — Cross-Encoder Reranking

**Configuration**: FAISS retrieves top-5 candidates (threshold 0.25), then
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranks down to top-3 (and,
separately, top-5). Scripts: `evaluation/evaluate_reranking.py` (retrieval
metrics), `evaluation/evaluate_reranked_generation.py` (top-3 generation),
`evaluation/evaluate_reranked_generation_top5.py` (top-5 generation).

**Retrieval ranking metrics** (`evaluation/reranking_results.json`)

| Metric | Baseline (no rerank) | Reranked |
|---|---|---|
| Recall@1 | 0.500 | 0.800 |
| MRR@1 | 0.500 | 0.800 |
| Recall@3 | 0.900 | 0.900 |
| MRR@3 | 0.683 | 0.833 |
| Recall@5 | 0.900 | 0.900 |
| MRR@5 | 0.683 | 0.833 |

Reranking is a clear win on retrieval ranking quality — it more than fixes
the weak Recall@1 from the baseline.

**End-to-end generation metrics** (keyword coverage, 10 questions)

| Configuration | Average keyword coverage |
|---|---|
| Plain FAISS top-5, no rerank (`evaluate_generation.py`) | 0.742 |
| Reranked, final top-3 (`evaluate_reranked_generation.py`) | 0.522 |
| Reranked, final top-5 (`evaluate_reranked_generation_top5.py`) | 0.581 |

**Conclusion**: improved retrieval ranking did **not** translate into
better generated answers. Reranking reorders toward chunks that are
topically closer to the question but apparently have less literal lexical
overlap with the human-written expected answers, which hurts the
keyword-coverage metric and, on manual inspection, correlates with answers
that drift from the source text. This is treated as a real, documented
trade-off, not noise.

**Production decision**: **rejected for now**. Reranking is implemented in
`src/reranker.py` and exercised by the evaluation scripts above, but
`src/rag.py` (the production path) does not call it. Do not enable it
without a follow-up experiment that either improves generation quality
with reranking or replaces keyword coverage with a metric that better
reflects answer correctness.

---

## Experiment 3 — Prompt / Generation Variants

**Configuration**: comparing prompt wording and final top-k for the
grounded-generation stage, no reranking, threshold 0.25.

| Configuration | Average keyword coverage |
|---|---|
| Original generation baseline | **0.742** |
| Newer concise prompt, top-3 | 0.637 |
| Newer prompt, plain FAISS top-5 (current `prompt_builder.py`) | 0.721 |
| Reranked top-3 | 0.522 |
| Reranked top-5 | 0.581 |

**Conclusion**: the original generation baseline (0.742) remains the best
measured generation result so far. The current prompt + top-5 (0.721) is
close but has not surpassed it. Reranking should not be used in
production. Keyword coverage is a crude, lexical-overlap metric — useful
for relative comparison between configurations run on the same questions,
but it should eventually be supplemented with a semantic answer-quality
metric before being treated as ground truth.

**Production decision**: current prompt/top-5 configuration kept as the
active baseline pending a controlled re-run against the 0.742 configuration
(the exact original prompt used for that run was not preserved verbatim
alongside its result file, so it is recorded here as historical context
rather than something currently reproducible byte-for-byte).

---

## Experiment 4 — Generation Pipeline Warning Fix

**Date**: 2026-09-01

**Issue**: `src/generator.py` logged
`"Both max_new_tokens (=120) and max_length(=20) seem to have been set."`
on every call, despite the code only ever passing `max_new_tokens`.

**Root cause**: `transformers.pipeline("text-generation", ...)` builds its
own merged `generation_config` (a `Pipeline.generation_config` distinct
from `model.generation_config`), combining the model's own generation
config with the pipeline's internal defaults. That merge produces
`max_length=20` even though Qwen2.5-1.5B-Instruct's own
`generation_config.json` never sets `max_length` at all — confirmed by
inspecting `model.generation_config.max_length` (`None`) versus
`pipeline.generation_config.max_length` (`20`) directly. That stale
`max_length=20` on the pipeline-level object is what collides with
`max_new_tokens=120` at call time.

**Fix**: `src/generator.py` now clears the stale attribute once, right
after the pipeline is constructed:

```python
generator.generation_config.max_length = None
```

This removes the conflict at its source rather than filtering the warning.
Verified with `warnings.catch_warnings()` plus a direct call using the
exact `generate_answer` arguments — the max_length warning no longer
appears.

One separate, pipeline-internal notice remains:
`"Passing generation_config together with generation-related
arguments=({'max_new_tokens', 'do_sample'}) is deprecated..."`. This comes
from `TextGenerationPipeline._forward`, which always injects its own
`generation_config` object alongside whatever loose kwargs are passed to
the pipeline call — it fires regardless of caller code and isn't something
`generate_answer` controls without abandoning the `pipeline()` convenience
wrapper for a manual `model.generate()` call. Left as-is; not worth that
tradeoff for a cosmetic notice.

**Verification**: re-ran `evaluation/evaluate_generation.py` (top_k=5,
threshold 0.25, no reranking — same configuration as Experiment 3's
"newer prompt, plain FAISS top-5" row) after the fix.

| | Before fix | After fix |
|---|---|---|
| Average keyword coverage | 0.721 | 0.721 |
| `max_length` warning | present | absent |

`evaluation/generation_results.json` is byte-for-byte identical to the
pre-fix run — expected, since generation is deterministic
(`do_sample=False`) and `max_new_tokens` already took precedence over the
stale `max_length` per the warning's own text. The fix is a correctness/
cleanliness fix for the generation config, not a quality change.

**Production decision**: kept. No metric regression, warning eliminated.

---

## Experiment 5 — Semantic Similarity Metric

**Date**: 2026-09-01

**Motivation**: keyword coverage is a token-intersection metric — it
rewards exact word overlap and has no notion of paraphrase or partial
correctness. It's cheap and useful for relative comparisons between
configs run on the same questions, but it's a weak signal for whether an
individual answer is actually correct, and the task list flagged Q10 as
one of the worst answers based on its low coverage score alone.

**Change**: `evaluation/evaluate_generation.py` now also computes
`semantic_similarity` per question — cosine similarity between the MiniLM
(`all-MiniLM-L6-v2`) embeddings of the expected answer and the generated
answer, via `src.embedding.create_embeddings` (already L2-normalized, so
a plain dot product is the cosine score). No new dependency, no new model
download — it reuses the embedding model already loaded for retrieval.
`keyword_coverage` is unchanged and still reported; semantic similarity is
additive, not a replacement.

**Result** (`evaluation/generation_results.json`, same config as
Experiment 3's "newer prompt, plain FAISS top-5" row: top_k=5, threshold
0.25, no reranking):

| Metric | Value |
|---|---|
| Average keyword coverage | 0.721 (unchanged) |
| Average semantic similarity | 0.751 |

**Per-question comparison** (keyword coverage / semantic similarity):

| Q | Question | Keyword coverage | Semantic similarity |
|---|---|---|---|
| 1 | Main objective | 0.684 | 0.710 |
| 2 | Water-quality variables | 0.750 | 0.734 |
| 3 | Dataset quality risks | 0.846 | 0.615 |
| 4 | Anomaly detection methods | 0.684 | 0.793 |
| 5 | Forecasting methods | 0.933 | 0.853 |
| 6 | Forecasting evaluation metrics | 0.625 | 0.763 |
| 7 | Temporal train/val/test split | 0.786 | **0.588** |
| 8 | Robustness/ablation study | 0.840 | 0.771 |
| 9 | Final model/pipeline freeze | 0.731 | 0.916 |
| 10 | Mandatory deliverables | **0.326** | 0.764 |

**Finding**: the two metrics disagree in an informative way. Q10 has the
worst keyword coverage (0.326) but a solidly average semantic similarity
(0.764) — inspecting the generated answer shows it's essentially a correct
restatement of the deliverables list, just tokenized differently from the
expected answer (e.g. slash-joined terms like "Preprocessing/EDA" split
across tokens). That looks like a keyword-coverage measurement artifact,
not a real generation defect. Q7, by contrast, has above-average keyword
coverage (0.786) but the *lowest* semantic similarity in the set (0.588) —
its generated answer is a long, meandering paragraph that never states the
one fact the question asks for (that the split must be chronological), so
it accumulates matching keywords without answering the question. This
matches the concern already flagged for Task 2. This is exactly why a
second, independent metric is useful even without abandoning keyword
coverage.

**Production decision**: kept as a second evaluation signal alongside
keyword coverage. Neither metric replaces the other; both are recorded
per question going forward in `evaluate_generation.py`. Q7 is flagged as
the priority case for the Task 2 answer-quality investigation.

---

## Experiment 6 — Q7 Answer-Quality Investigation and Prompt Rule (Rejected)

**Date**: 2026-09-01

**Investigation**: before changing anything, traced Q7 ("What is required
for the temporal train, validation, and test split?") through the actual
retrieved chunks in `evaluation/generation_results.json` and
`evaluation/artifacts/chunk100_chunks.pkl`:

- Q7's generated answer was truncated mid-sentence at the 120-token
  generation limit, and never states "chronological."
- Two of the three human-labeled relevant chunks for Q7
  (`questions_chunk100.json`: page 3 chunk 1, page 3 chunk 2 — which
  literally say *"Define chronological train/validation/test splits"* and
  *"Implement cleaning, chronological splits and scaling"*) were **not
  retrieved** in the top-5 at all. This is a genuine retrieval gap, out of
  scope for this task (reserved for Task 3).
- The highest-scoring retrieved chunk (page 2, chunk 1) is a task
  instruction from the source planning document — *"Define what a correct
  train/validation/test split should preserve"* — not the actual
  requirement. The model echoed this meta-instruction almost verbatim
  instead of locating the concrete fact elsewhere in its context (the word
  "chronological" does appear once in the retrieved set, in an unrelated
  ablation-study chunk, but the model never surfaced it).

Conclusion of the investigation: the problem is a mix of (a) a retrieval
gap that this task isn't allowed to touch, and (b) the model paraphrasing
a task-description sentence instead of hunting for the concrete fact in
weaker supporting context — a prompt/generation-behavior issue on top of
the retrieval gap.

**Change tried**: added one rule to `src/prompt_builder.py` instructing
the model to open with a direct factual sentence rather than describing
what needs to be "defined" or "decided":

```
3. Start with a direct sentence that states the answer itself.
   Do not open by describing what needs to be decided, defined, or
   determined - state the actual requirement or fact.
```

This was a single, general instruction (not Q7-specific), inserted early
in the numbered rule list.

**Result** (same config: top_k=5, threshold 0.25, no reranking):

| Metric | Before | After |
|---|---|---|
| Average keyword coverage | 0.721 | 0.667 |
| Average semantic similarity | 0.751 | 0.733 |
| Q7 keyword coverage | 0.786 | 0.786 (unchanged) |
| Q7 semantic similarity | 0.588 | 0.593 (unchanged, within noise) |

Q7's generated answer was **essentially unchanged** — the model still
opened with "The best practice is to define what a correct
train/validation/test split should preserve..." almost word-for-word,
ignoring the new rule. Meanwhile the same change **regressed several
unrelated questions**: Q1 (keyword coverage 0.684 → 0.421, semantic
similarity 0.710 → 0.642) lost detail and became overly terse; Q5
(keyword coverage 0.933 → 0.667, semantic similarity 0.853 → 0.618) got
noticeably worse and started blending in an unrelated anomaly-detection
method ("isolation forest") into an answer about forecasting methods.

**Why it failed**: Qwen2.5-1.5B-Instruct under greedy decoding does not
reliably follow an additional directive buried in a growing numbered rule
list — it kept reproducing the same opening sentence from its
highest-scored context chunk regardless of the new instruction, while the
instruction's side effects (pushing toward terser, more "direct-sounding"
phrasing) trimmed genuinely relevant detail from otherwise-good answers.
This is evidence that prompt-only nudges have limited leverage over this
model's tendency to anchor on and paraphrase its top-ranked context chunk,
and that Q7 specifically is bottlenecked more by what's retrieved than by
how the prompt asks for it to be used.

**Production decision**: **REJECTED and reverted**. `src/prompt_builder.py`
and `evaluation/generation_results.json` are back to their pre-experiment
state (confirmed via `git diff` — no changes). No net improvement, and a
clear regression across the benchmark. The Q7 failure mode looks primarily
retrieval-driven (missing ground-truth chunks) rather than
prompt-driven, which should inform Task 3's retrieval-quality work rather
than further prompt patches in isolation.

---

## Experiment 7 — Retrieval Candidate Depth (top_k 5 → 10)

**Date**: 2026-09-01

**Hypothesis**: Experiment 6's investigation showed Q7's missing evidence
chunk (page 3, chunk 1 — *"Define chronological train/validation/test
splits"*) ranks **#10** by FAISS similarity across the full 28-chunk
corpus, with a score of 0.2555 — above the 0.25 threshold, just outside
the top-5 cutoff. Re-checking every other question the same way found the
same pattern on Q1: its single relevant chunk (page 1, chunk 0, the
project title) ranks **#8**, score 0.286 — also above threshold, also
just outside top-5. Both are legitimate, above-threshold matches starved
by too shallow a candidate window, not an embedding-quality or
chunk-boundary problem. Hypothesis: increasing retrieval top_k from 5 to
10 should surface both without needing reranking or a different embedding
model.

**Configuration**: only `top_k` changed, from 5 to 10. Chunking (100/20),
embedding model (`all-MiniLM-L6-v2`), threshold (0.25), generation model,
and prompt (`src/prompt_builder.py`, unchanged from before Experiment 6)
all held fixed.

- `evaluation/evaluate_retrieval.py`: `TOP_K_VALUES` extended from
  `[1, 3, 5]` to `[1, 3, 5, 8, 10]` (additive — existing k=1/3/5 values
  are computed exactly as before and don't move).
- `evaluation/evaluate_generation.py`: `TOP_K` changed from 5 to 10.

**Retrieval metrics** (`evaluation/results.json`)

| Metric | k=5 (before) | k=8 | k=10 |
|---|---|---|---|
| Recall | 0.900 | 1.000 | 1.000 |
| MRR | 0.683 | 0.696 | 0.696 |

Recall@1/3/5 and MRR@1/3/5 are unchanged (0.500/0.900/0.900 and
0.500/0.683/0.683) — expected, since they're separate cutoffs and were
already "hit" for every question except Q1. Recall reaches **1.000** at
k=8 and k=10: Q1's chunk (rank 8) is now captured, and no question is a
total miss anymore.

**Generation metrics** (`evaluation/generation_results.json`, top_k=10,
threshold 0.25, no reranking, prompt unchanged)

| Metric | Before (top_k=5) | After (top_k=10) |
|---|---|---|
| Average keyword coverage | 0.721 | **0.757** |
| Average semantic similarity | 0.751 | **0.787** |

Both metrics improved, and average keyword coverage now **exceeds the
historical best of 0.742** for the first time.

**Q7 before/after**:

- Before: *"The best practice is to define what a correct
  train/validation/test split should preserve..."* (echoes a task
  instruction, never states the requirement, gets cut off mid-sentence).
  keyword coverage 0.786, semantic similarity 0.588.
- After: *"...determining the chronological train, validation, and test
  splits. These steps ensure that the data used for training, validating,
  and testing is clean, consistent, and appropriately structured..."* —
  now explicitly states the chronological requirement. keyword coverage
  0.643 (down slightly — the answer restructures rather than repeating
  the question's exact wording as densely), semantic similarity **0.732**
  (up from 0.588). Retrieved chunks now include (3, 1), the chunk
  containing the actual requirement, confirming the mechanism.

**Q1 before/after** (same root cause, different question): before, a
long rambling paragraph, keyword coverage 0.684 / semantic similarity
0.710. After: *"To detect and predict underwater environmental changes
using water quality parameters"* — short, precise, matches the project
title almost verbatim. keyword coverage 0.579 (down — the terse answer
drops some padding words that happened to overlap with the expected
answer), semantic similarity **0.834** (up from 0.710).

**Other questions**: no catastrophic regression. Q5 and Q10 lost some
keyword coverage (0.933→0.667, 0.326→0.500 respectively — note Q10 is
still net *positive*) from answer rewording, but semantic similarity held
roughly flat on both (0.853→0.808, 0.764→0.758). No sign of irrelevant
context leakage or hallucination from the larger context window on manual
inspection of the full result set.

**Conclusion**: candidate depth (top_k) was the actual bottleneck behind
both the Q7 and Q1 weaknesses — the correct evidence already scores above
the relevance threshold, it just needed a wider retrieval window to reach
the prompt. This improves retrieval recall, fixes the specific Q7 failure
mode with a general (non-Q7-specific) change, and improves both generation
metrics with no material regression elsewhere.

**Production decision**: **KEEP**. `src/rag.py`'s default `top_k` raised
from 3 to 10 to match the validated configuration (threshold 0.25 is
unchanged and still does the job of rejecting out-of-document queries;
see Experiment 1 — a larger top_k does not weaken that gate, since
`retrieve()` still drops anything under 0.25 regardless of k).
