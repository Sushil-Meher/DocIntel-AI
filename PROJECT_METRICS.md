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

## Retrieval Gating (production)

- Selected similarity threshold: **0.25**
- Positive Recall@3 at this threshold: 0.900 (0/10 valid queries rejected)
- Negative-query rejection at this threshold: 100% (5/5 out-of-document
  questions correctly rejected)

## Generation

- Best measured generation keyword coverage: **0.742** (original generation
  baseline, plain FAISS retrieval, no reranking)
- Current production prompt/top-5 configuration: 0.721 keyword coverage,
  confirmed stable across two runs (before and after the generation-config
  warning fix on 2026-09-01)
- Cross-encoder reranking reduced generation keyword coverage to 0.522
  (top-3) / 0.581 (top-5) — **reranking is not used in production** for
  this reason, despite its retrieval-ranking gains above
- Evaluation set: 10 questions
- Metric caveat: keyword coverage is a crude lexical-overlap metric,
  useful for relative comparison between configurations run on the same
  question set, not an absolute correctness score
- Added a second, independent metric: average semantic similarity (cosine
  similarity between MiniLM embeddings of expected vs. generated answer)
  is **0.751** on the current production prompt/top-5 configuration. It
  disagrees with keyword coverage in informative ways per-question (see
  `evaluation/experiments.md`, Experiment 5) — e.g. it correctly rates a
  paraphrased-but-correct answer highly where keyword coverage scores it
  low, and flags a keyword-rich but off-target answer that keyword
  coverage alone missed

## Production Configuration (current)

FAISS `IndexFlatIP` retrieval (top-k, no reranking) → 0.25 similarity
threshold → grounded prompt → `Qwen/Qwen2.5-1.5B-Instruct` → deterministic
source/page citations. See `evaluation/experiments.md` for why reranking
and the alternate prompt variants were not promoted.

## Labeling Convention

- **Production**: reflects `src/rag.py` as currently deployed.
- **Experimental**: measured in `evaluation/*.py` scripts but not wired
  into the production pipeline. Never presented as production performance.
