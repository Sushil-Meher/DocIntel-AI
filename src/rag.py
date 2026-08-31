from .retriever import retrieve
from .prompt_builder import build_prompt
from .generator import generate_answer


def answer_question(
    query: str,
    index,
    chunks,
    top_k: int = 3
) -> str:

    results = retrieve(
        index,
        chunks,
        query,
        top_k=top_k
    )

    prompt = build_prompt(query, results)

    answer = generate_answer(prompt)

    return answer