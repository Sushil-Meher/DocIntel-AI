from dataclasses import dataclass

from document_loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def chunk_document(
    document: Document,
    chunk_size: int = 50,
    overlap: int = 10
) -> list[Chunk]:

    words = document.text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)

        chunks.append(
            Chunk(
                text=chunk_text,
                source=document.source,
                page=document.page,
                chunk_id=chunk_id
            )
        )

        chunk_id += 1

        start = end - overlap

    return chunks

