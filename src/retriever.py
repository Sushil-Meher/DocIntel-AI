import faiss
import numpy as np

from embedding import create_embeddings


def retrieve(index, chunks, query: str, top_k: int = 3):

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

    from document_loader import load_pdf
    from chunking import chunk_document
    from vector_store import create_index

    documents = load_pdf(
        "data/Artificial-Intelligence report.pdf"
    )

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(document)
        )

    index = create_index(chunks)

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

        print("\n", result["text"])