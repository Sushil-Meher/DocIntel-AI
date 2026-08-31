import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import load_index, load_chunks
from .embedding import create_embeddings


def retrieve(index,chunks,query: str,top_k: int = 3):

    query_embedding = create_embeddings([query])

    query_vector = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        chunk = chunks[index_position]

        results.append({
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "distance": float(distance)
        })

    return results

if __name__ == "__main__":

    index = load_index(
        "artifacts/faiss.index"
    )

    chunks = load_chunks(
        "artifacts/chunks.pkl"
    )

    results = retrieve(
        index,
        chunks,
        "What is artificial intelligence?",
        top_k=3
    )

    for result in results:

        print("\n" + "=" * 60)

        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Chunk:", result["chunk_id"])
        print("Distance:", result["distance"])

        print("\nText:")
        print(result["text"])