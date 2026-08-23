from vector_store import load_index, load_chunks
from retriever import retrieve
from prompt_builder import build_prompt

index = load_index("artifacts/faiss.index")
chunks = load_chunks("artifacts/chunks.pkl")

query = "What is artificial intelligence?"

results = retrieve(
    index,
    chunks,
    query,
    top_k=3
)

prompt = build_prompt(query, results)

