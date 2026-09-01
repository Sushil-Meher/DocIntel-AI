import numpy as np

from .embedding import create_embeddings


def retrieve(
    index,
    chunks,
    query: str,
    top_k: int = 3,
    min_score: float | None = None
):
    query_embedding = create_embeddings([query])

    query_vector = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        # For IndexFlatIP, larger score = more similar.
        if (
            min_score is not None
            and score < min_score
        ):
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "distance": float(score)
            }
        )

    return results