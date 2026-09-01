import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.document_loader import Document
from src.retriever import retrieve
from src import ingestion


DOC_A_TEXT = (
    "Project Zephyr uses a quantum flux capacitor to stabilize the "
    "graviton lattice. The zephyr calibration constant is 42."
)

DOC_B_TEXT = (
    "Project Orion relies on a cryogenic plasma injector to cool the "
    "fusion core. The orion pressure threshold is 917 kPa."
)

WEBSITE_TEXT = (
    "Acme Corp offers cloud storage plans starting at nineteen dollars "
    "per month with unlimited bandwidth."
)


def ingest_pdf_text(text, filename="doc.pdf"):
    document = Document(text=text, source=filename, page=1)
    with patch("src.ingestion.load_pdf", return_value=[document]):
        return ingestion.ingest_pdf(filename)


def ingest_url_text(text, url="https://example.com"):
    document = Document(text=text, source=url, page=1)
    with patch("src.ingestion.load_webpage", return_value=document):
        return ingestion.ingest_url(url)


class DocumentIsolationTests(unittest.TestCase):

    def test_pdf_a_retrieves_own_content(self):
        index, chunks = ingest_pdf_text(DOC_A_TEXT, "a.pdf")

        results = retrieve(
            index, chunks, "What is the zephyr calibration constant?", top_k=1
        )

        self.assertIn("42", results[0]["text"])

    def test_pdf_b_retrieves_own_content(self):
        index, chunks = ingest_pdf_text(DOC_B_TEXT, "b.pdf")

        results = retrieve(
            index, chunks, "What is the orion pressure threshold?", top_k=1
        )

        self.assertIn("917", results[0]["text"])

    def test_switching_from_a_to_b_drops_a_content(self):
        # Simulates a user processing PDF A, then processing PDF B.
        ingest_pdf_text(DOC_A_TEXT, "a.pdf")
        index_b, chunks_b = ingest_pdf_text(DOC_B_TEXT, "b.pdf")

        for chunk in chunks_b:
            self.assertNotIn("zephyr", chunk.text.lower())

        results = retrieve(
            index_b,
            chunks_b,
            "What is the zephyr calibration constant?",
            top_k=3,
            min_score=0.25
        )

        self.assertEqual(results, [])

    def test_website_retrieves_own_content(self):
        index, chunks = ingest_url_text(WEBSITE_TEXT)

        results = retrieve(
            index,
            chunks,
            "How much does the cloud storage plan cost per month?",
            top_k=1
        )

        self.assertIn("nineteen", results[0]["text"])

    def test_switching_from_website_to_pdf_drops_website_content(self):
        ingest_url_text(WEBSITE_TEXT)
        index, chunks = ingest_pdf_text(DOC_A_TEXT, "a.pdf")

        for chunk in chunks:
            self.assertNotIn("acme", chunk.text.lower())

        results = retrieve(
            index,
            chunks,
            "How much does the cloud storage plan cost per month?",
            top_k=3,
            min_score=0.25
        )

        self.assertEqual(results, [])

    def test_ingestion_does_not_touch_shared_artifacts(self):
        # This is the actual isolation bug: ingest_pdf/ingest_url used to
        # persist every document to the same artifacts/faiss.index and
        # artifacts/chunks.pkl, so a second document silently overwrote
        # the first on disk regardless of which session it came from.
        # Running from a throwaway empty directory means there is no
        # pre-existing artifacts/ folder to coincidentally match against.
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)

            try:
                ingest_pdf_text(DOC_A_TEXT, "a.pdf")
                self.assertFalse(os.path.exists("artifacts"))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
