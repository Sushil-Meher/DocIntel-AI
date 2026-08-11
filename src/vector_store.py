import faiss
import numpy as np

from embedding import create_embeddings
from chunking import chunk_document
from document_loader import load_pdf
import pickle


def create_index(chunks):

    texts = [chunk.text for chunk in chunks]

    embeddings = create_embeddings(texts)

    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def save_index(index, path: str):
    faiss.write_index(index, path)


def load_index(path: str):
    return faiss.read_index(path)

def save_chunks(chunks, path: str):
    with open(path, "wb") as file:
        pickle.dump(chunks, file)


def load_chunks(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":

    documents = load_pdf(
        "data/Artificial-Intelligence report.pdf"
    )

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(document)
        )

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    index = create_index(chunks)

    save_index(index,"artifacts/faiss.index")


    save_chunks(chunks,"artifacts/chunks.pkl")


    results = search(
        index,
        chunks,
        "What is artificial intelligence?",
        top_k=3
    )

    for result in results:

        print("\n" + "-" * 60)

        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Chunk ID:", result["chunk_id"])
        print("Distance:", result["distance"])

        print("\nText:")
        print(result["text"])
