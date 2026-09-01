from .document_loader import load_pdf, Document
from .web_loader import load_webpage
from .chunking import chunk_document
from .vector_store import create_index, save_index, save_chunks


def ingest_pdf(file_path: str):
    documents = load_pdf(file_path)

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=100,
                overlap=20
            )
        )

    index = create_index(chunks)

    save_index(index, "artifacts/faiss.index")
    save_chunks(chunks, "artifacts/chunks.pkl")

    return index, chunks


def ingest_url(url: str):
    webpage = load_webpage(url)

    chunks = chunk_document(
            webpage,
            chunk_size=100,
            overlap=20
        )

    index = create_index(chunks)

    save_index(index, "artifacts/faiss.index")
    save_chunks(chunks, "artifacts/chunks.pkl")

    return index, chunks