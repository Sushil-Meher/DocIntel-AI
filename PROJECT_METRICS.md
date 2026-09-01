# Project Metrics

Resume-worthy, validated metrics only. Full methodology and raw result
files live in `evaluation/experiments.md` — this file is a curated summary.

Evaluation set: 10 hand-labeled questions over `evaluation/corpus.pdf`
(`evaluation/questions_chunk100.json`), plus 5 out-of-document negative
questions (`evaluation/negative_questions.json`).

## Retrieval

- Recall@1: 0.50 baseline → 0.80 in the cross-encoder reranking experiment
  (**experimental only** — reranking is not in production)
- Recall@3: 0.90 (baseline, unchanged with reranking)
- MRR@3: 0.683 baseline → 0.833 in the reranking experiment (**experimental only**)
- Recall@5: 0.90 → **1.00** at top_k=8/10, no reranking (**production**) —
  the one previously-missed question's relevant chunk ranked 8th by
  similarity, just outside the old top-5 window; see
  `evaluation/experiments.md` Experiment 7

## Retrieval Gating (production)

- Selected similarity threshold: **0.25**
- Positive Recall@3 at this threshold: 0.900 (0/10 valid queries rejected)
- Negative-query rejection at this threshold: 100% (5/5 out-of-document
  questions correctly rejected)

## Generation

- Current production configuration (top_k=10, threshold 0.25, no
  reranking): **0.757 average keyword coverage, 0.787 average semantic
  similarity** — the best measured generation result so far, surpassing
  the previous historical best of 0.742 (see `evaluation/experiments.md`
  Experiment 7)
- Cross-encoder reranking reduced generation keyword coverage to 0.522
  (top-3) / 0.581 (top-5) — **reranking is not used in production** for
  this reason, despite its retrieval-ranking gains above
- Evaluation set: 10 questions
- Metric caveat: keyword coverage is a crude lexical-overlap metric,
  useful for relative comparison between configurations run on the same
  question set, not an absolute correctness score
- Second, independent metric: average semantic similarity (cosine
  similarity between MiniLM embeddings of expected vs. generated answer).
  It disagrees with keyword coverage in informative ways per-question (see
  Experiment 5) and was the signal that drove the top_k fix in
  Experiment 7 — e.g. it correctly rated Q7's old answer as weak despite
  above-average keyword coverage, because that answer echoed a task
  description instead of stating the actual requirement

## Production Configuration (current)

FAISS `IndexFlatIP` retrieval (top_k=10, no reranking) → 0.25 similarity
threshold → grounded prompt → `Qwen/Qwen2.5-1.5B-Instruct` → deterministic
source/page citations. See `evaluation/experiments.md` for why reranking,
the alternate prompt variants, and the Q7 prompt-rule attempt were not
promoted, and why top_k=10 was.

## Expanded Evaluation Benchmark (Task 10)

A larger, harder benchmark (28 positive + 12 negative questions, vs. the
original 10 + 5) run against the **same production configuration**
(chunk100 index, threshold 0.25, top_k 10, no reranking, same embeddings
and generator). This is a different, harder question population than the
numbers above, not a re-measurement of the same benchmark - do not read
the deltas below as regressions.

- Recall@1/3/5/8/10: 0.321 / 0.643 / 0.750 / 0.821 / 0.893
- Generation: 0.708 average keyword coverage, 0.619 average semantic
  similarity
- Negative-query rejection at threshold 0.25: **0/12 (0%)** — down from
  5/5 on the historical negative set

That last number is the most important finding from this benchmark
expansion, not a weakness to hide: the historical 5 negatives were
trivially unrelated to the corpus (e.g. "capital of France",
similarity ~0.02) and were always going to be rejected. The 12 expanded
negatives are realistic, topically-plausible questions (dataset size,
sensor location, mentor's name) that share vocabulary with the corpus
without being answered by it, and they score 0.32-0.54 - all above the
0.25 gate. **The current similarity threshold is a topic-relevance
filter, not a fact-coverage filter.** This is a known, documented
limitation (see `evaluation/experiments.md`, Task 10), not fixed in this
task per its explicit scope (benchmark expansion, not threshold tuning).

See `evaluation/experiments.md` (Task 10) for the full methodology,
category breakdown, and per-question detail. Full results:
`evaluation/expanded_results.json`, `evaluation/
expanded_generation_results.json`, `evaluation/expanded_negative_results.json`.

## Labeling Convention

- **Production**: reflects `src/rag.py` as currently deployed.
- **Experimental**: measured in `evaluation/*.py` scripts but not wired
  into the production pipeline. Never presented as production performance.
- **Expanded benchmark**: measured on a larger, harder question set than
  the original 10+5 - a different population, not a direct before/after
  comparison with the numbers elsewhere in this file.
