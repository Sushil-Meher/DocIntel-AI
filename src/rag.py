from .retriever import retrieve
from .prompt_builder import build_prompt
from .generator import generate_answer
from .vector_store import load_index, load_chunks


# Load the current application index and chunks
index = load_index(
    "artifacts/faiss.index"
)

chunks = load_chunks(
    "artifacts/chunks.pkl"
)


# Minimum retrieval similarity required
# for a query to proceed to the LLM.
MIN_RELEVANCE_SCORE = 0.25


def answer_question(
    query: str,
    index,
    chunks,
    top_k: int = 10,
    min_score: float = MIN_RELEVANCE_SCORE
) -> str:

    # Retrieve candidate chunks
    results = retrieve(
        index,
        chunks,
        query,
        top_k=top_k,
        min_score=min_score
    )

    # No sufficiently relevant context
    if not results:

        return (
            "I could not find the answer in the "
            "provided documents."
        )

    # Build grounded prompt
    prompt = build_prompt(
        query,
        results
    )

    # Generate answer
    answer = generate_answer(prompt).strip()

    # Refusal from the model
    if answer.lower().startswith(
        "i could not find the answer"
    ):
        return answer

    # Add deterministic source citations
    sources = []
    seen = set()

    for result in results:

        source_key = (
            result["source"],
            result["page"]
        )

        if source_key not in seen:

            seen.add(source_key)

            sources.append(
                f"- {result['source']} "
                f"(Page {result['page']})"
            )

    if sources:

        answer = (
            answer
            + "\n\nSources:\n"
            + "\n".join(sources)
        )

    return answer.strip()


if __name__ == "__main__":

    query = "What is the capital of France?"
    
    answer = answer_question(
        query,
        index,
        chunks
    )

    print(answer)