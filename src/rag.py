from vector_store import load_index, load_chunks
from retriever import retrieve
from prompt_builder import build_prompt
from generator import generate_answer

index = load_index("artifacts/faiss.index")
chunks = load_chunks("artifacts/chunks.pkl")

query = "Who coined the phrase Artificial Intelligence and when?"

results = retrieve(
    index,
    chunks,
    query,
    top_k=3
)

prompt = build_prompt(query, results)

answer = generate_answer(prompt)

print(answer)

