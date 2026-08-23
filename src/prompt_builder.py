def build_prompt(query: str, results: list[dict]) -> str:

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"[Context {i}]\n"
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an enterprise document assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context,
say that you could not find the answer in the provided documents.

Context:

{context}

Question:
{query}

Answer:
"""

    return prompt


if __name__ == "__main__":

    from vector_store import load_index, load_chunks
    from retriever import retrieve

    index = load_index(
        "artifacts/faiss.index"
    )

    chunks = load_chunks(
        "artifacts/chunks.pkl"
    )

    query = "What is artificial intelligence?"

    results = retrieve(
        index,
        chunks,
        query,
        top_k=3
    )

    prompt = build_prompt(
        query,
        results
    )

    print(prompt)