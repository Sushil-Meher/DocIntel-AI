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

INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"


def build_experiment():

    documents = load_pdf(PDF_PATH)

    chunks = []

    for document in documents:

        chunks.extend(
            chunk_document(
                document,
                chunk_size=100,
                overlap=20
            )
        )

    print(f"Documents/pages: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    index = create_index(chunks)

    save_index(index, INDEX_PATH)
    save_chunks(chunks, CHUNKS_PATH)

    print("\nChunking experiment created.")
    print(f"Index: {INDEX_PATH}")
    print(f"Chunks: {CHUNKS_PATH}")


if __name__ == "__main__":
    build_experiment()