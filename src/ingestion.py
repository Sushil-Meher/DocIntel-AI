from .document_loader import load_pdf, Document
from .web_loader import load_webpage
from .chunking import chunk_document
from .vector_store import create_index


# ingest_pdf/ingest_url build a fresh index/chunks pair per call and
# return it - the caller (the Streamlit session) owns that document's
# retrieval context from here. They no longer persist to a shared
# artifacts/ path: that used to mean any new document silently
# overwrote the previous one on disk, which is exactly the kind of
# cross-document leak this app needs to avoid.

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

    return index, chunks


def ingest_url(url: str):
    webpage = load_webpage(url)

    chunks = chunk_document(
            webpage,
            chunk_size=100,
            overlap=20
        )

    index = create_index(chunks)

    return index, chunks