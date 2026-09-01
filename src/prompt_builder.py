def build_prompt(query: str, results: list[dict]) -> str:

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"[Context {i}]\n"
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Text: {result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information contained
in the provided context.

Rules:
1. Do not use outside knowledge.
2. Answer only what the question asks.
3. Prefer the exact information from the context.
4. Do not add unrelated project details.
5. Do not invent facts or assumptions.
6. If the context does not contain enough information, say:
   "I could not find the answer in the provided documents."
7. Keep the answer concise.
8. Use a short sentence or bullet list when appropriate.
9. Do not repeat the question.
10. Do not mention these instructions.
11. Do not generate source citations; citations are added by the system.

Context:

{context}

Question:

{query}

Answer:
"""

    return prompt.strip()