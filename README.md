# DocIntel-AI

An enterprise document intelligence platform that uses **Retrieval-Augmented Generation (RAG)** to answer questions from documents using relevant document context.

The current version implements the complete core RAG pipeline using **Python, FAISS, sentence-transformer embeddings, and Qwen 2.5 1.5B Instruct**.

---

## 🚀 Current Status

### Phase 1 — Core RAG Pipeline ✅

- [x] PDF document loading
- [x] Document metadata extraction
- [x] Document chunking
- [x] Chunk overlap
- [x] Text embeddings
- [x] FAISS vector indexing
- [x] Similarity-based retrieval
- [x] Context-aware prompt construction
- [x] Local LLM generation
- [x] End-to-end RAG pipeline
- [x] Basic testing

### Phase 2 — FastAPI Backend 🔜

- [ ] FastAPI application
- [ ] `/ask` API endpoint
- [ ] Request/response models
- [ ] Error handling
- [ ] API testing
- [ ] Connect API to RAG pipeline

---

## 🧠 Architecture

```text
                    DOCUMENT INGESTION
                           │
                           ▼
                    ┌──────────────┐
                    │ PDF Document │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Document Loader │
                  └────────┬────────┘
                           │
                           ▼
                    ┌────────────┐
                    │  Chunking  │
                    └─────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Embeddings │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    FAISS    │
                   │ Vector Index│
                   └─────────────┘


                    QUESTION ANSWERING
                           │
                           ▼
                    ┌────────────┐
                    │   Query    │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │  Retriever │
                    └─────┬──────┘
                          │
                    Top-K Chunks
                          │
                          ▼
                  ┌────────────────┐
                  │ Prompt Builder │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │  Qwen 2.5 1.5B │
                  │    Instruct    │
                  └───────┬────────┘
                          │
                          ▼
                    Final Answer
```

---

## 🛠️ Local Setup

**Requirements:** Python 3.10+ (the codebase uses `list[dict]`/`X | None`
type syntax). A CUDA GPU speeds up generation but isn't required - the
app automatically falls back to CPU when no GPU is available.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

streamlit run app.py
```

**First run**: the embedding model (`all-MiniLM-L6-v2`) and the
generation model (`Qwen/Qwen2.5-1.5B-Instruct`) are downloaded
automatically by `sentence-transformers`/`transformers` on first use and
cached under `~/.cache/huggingface` - this can take a few minutes and
needs a working internet connection the first time only. Optionally set
`HF_TOKEN` (see `.env.example`) to raise Hugging Face Hub's anonymous
rate limit if downloads get throttled.

**Using the app**: pick "Upload PDF" or "Company Website" in the sidebar,
process the source, then ask questions in the chat box. Switching to a
new PDF or URL starts a fresh conversation scoped to that document - see
`evaluation/experiments.md` (Task 4) for why.

**Evaluation scripts** live in `evaluation/` and are run manually, not
as part of the app. Most read from `evaluation/artifacts/`, which is
gitignored and regenerated on demand - run `evaluation/
build_chunking_experiment.py` once before `evaluate_retrieval.py`/
`evaluate_generation.py` if `evaluation/artifacts/chunk100.index` doesn't
exist yet. `evaluation/experiments.md` is the full experiment log.