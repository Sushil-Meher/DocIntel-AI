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
