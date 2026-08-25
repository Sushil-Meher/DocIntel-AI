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