from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


def create_embeddings(
    texts: list[str]
) -> list[list[float]]:

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    return embeddings.tolist()