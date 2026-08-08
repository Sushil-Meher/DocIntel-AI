from dataclasses import dataclass
import pymupdf


@dataclass
class Document:
    text: str
    source: str
    page: int


def load_pdf(file_path: str) -> list[Document]:
    document = pymupdf.open(file_path)

    documents = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            documents.append(
                Document(
                    text=text,
                    source=file_path,
                    page=page_number
                )
            )

    document.close()

    return documents


