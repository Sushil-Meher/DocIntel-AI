from vector_store import load_index, load_chunks
from retriever import retrieve
from prompt_builder import build_prompt
from generator import generate_answer


index = load_index("artifacts/faiss.index")
chunks = load_chunks("artifacts/chunks.pkl")


def answer_question(query: str, top_k: int = 3) -> str:

    results = retrieve(
        index,
        chunks,
        query,
        top_k=top_k
    )

    prompt = build_prompt(query, results)

    answer = generate_answer(prompt)

    return answer

if __name__ == "__main__":
    query = "What is artificial intelligence?"

    answer = answer_question(query)

    print(answer)



