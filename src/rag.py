from .retriever import retrieve
from .prompt_builder import build_prompt, build_contextual_query_prompt
from .generator import generate_answer


# Minimum retrieval similarity required
# for a query to proceed to the LLM.
MIN_RELEVANCE_SCORE = 0.25

# Only the last few turns matter for resolving a pronoun or a follow-up;
# older turns just add noise to the rewrite prompt.
HISTORY_WINDOW = 3


def contextualize_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    prompt = build_contextual_query_prompt(query, history[-HISTORY_WINDOW:])

    rewritten = generate_answer(prompt).strip()

    return rewritten or query


def answer_question(
    query: str,
    index,
    chunks,
    history: list[dict] | None = None,
    top_k: int = 10,
    min_score: float = MIN_RELEVANCE_SCORE
) -> str:

    # History is only ever used to resolve references in the query itself -
    # the retrieved document chunks remain the only source of facts.
    retrieval_query = contextualize_query(query, history or [])

    # Retrieve candidate chunks
    results = retrieve(
        index,
        chunks,
        retrieval_query,
        top_k=top_k,
        min_score=min_score
    )

    # No sufficiently relevant context
    if not results:

        return (
            "I could not find the answer in the "
            "provided documents."
        )

    # Build grounded prompt - uses the standalone (contextualized) query
    # so the model isn't asked to answer a dangling "it"/"this" without
    # the conversation that resolves it.
    prompt = build_prompt(
        retrieval_query,
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

    from .vector_store import load_index, load_chunks

    index = load_index("artifacts/faiss.index")
    chunks = load_chunks("artifacts/chunks.pkl")

    query = "What is the capital of France?"

    answer = answer_question(
        query,
        index,
        chunks
    )

    print(answer)