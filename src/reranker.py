from sentence_transformers import CrossEncoder


# Cross-encoder reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Re-rank retrieved chunks using a cross-encoder.

    Args:
        query: User's question.
        results: Initial FAISS retrieval results.
        top_k: Number of results to return after reranking.

    Returns:
        Re-ranked retrieval results.
    """

    if not results:
        return []

    pairs = [
        [query, result["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):

        updated_result = result.copy()

        updated_result["rerank_score"] = float(
            score
        )

        reranked.append(updated_result)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]