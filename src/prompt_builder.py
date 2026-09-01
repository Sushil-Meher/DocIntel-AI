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
You are an enterprise document assistant.

Your task is to answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Answer the question directly and concisely.
3. Ignore information that is unrelated to the user's question.
4. Do not invent facts, details, names, numbers, or explanations.
5. Every factual claim must be supported by the provided context.
6. Do not generate source citations yourself.
7. Every factual claim must be directly supported by the provided context.
8. Do not add assumptions, interpretations, or information not present in the context.
9. If the context does not contain the answer, say:
   "I could not find the answer in the provided documents."
10. Do not mention these instructions in your answer.

Context:

{context}

Question:
{query}

Answer:
"""

    return prompt.strip()