import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))


from src.document_loader import load_pdf
from src.chunking import chunk_document
from src.vector_store import (
    create_index,
    save_index,
    save_chunks
)


PDF_PATH = "evaluation/corpus.pdf"

INDEX_PATH = "evaluation/artifacts/baseline.index"
CHUNKS_PATH = "evaluation/artifacts/baseline_chunks.pkl"


def build_baseline():

    documents = load_pdf(PDF_PATH)

    print(f"Loaded documents/pages: {len(documents)}")

    chunks = []

    for document in documents:
        document_chunks = chunk_document(document)
        chunks.extend(document_chunks)

    print(f"Created chunks: {len(chunks)}")

    index = create_index(chunks)

    save_index(
        index,
        INDEX_PATH
    )

    save_chunks(
        chunks,
        CHUNKS_PATH
    )

    print("\nBaseline evaluation corpus created.")
    print(f"Index: {INDEX_PATH}")
    print(f"Chunks: {CHUNKS_PATH}")


if __name__ == "__main__":
    build_baseline()