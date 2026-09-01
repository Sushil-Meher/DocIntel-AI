import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag import Answer


APP_PATH = str(ROOT_DIR / "app.py")


class InitialStateTests(unittest.TestCase):

    def test_no_document_loaded_shows_helpful_message_and_no_chat_input(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)
        self.assertEqual(len(at.chat_input), 0)
        self.assertTrue(
            any("Upload a PDF" in i.value for i in at.info)
        )


class SourceStatusTests(unittest.TestCase):

    def test_pdf_ingestion_shows_current_source_status(self):
        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]

        with patch("src.ingestion.ingest_pdf", return_value=(fake_index, fake_chunks)):
            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.file_uploader[0].upload("a.pdf", b"%PDF-1.4 fake", "application/pdf")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)
        rendered = [m.value for m in at.markdown]
        captions = [c.value for c in at.caption]
        self.assertIn("**PDF · a.pdf**", rendered)
        self.assertIn("Ready", captions)

    def test_website_ingestion_shows_current_source_status(self):
        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)):
            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.segmented_control[0].set_value("Website")
            at.run(timeout=15)
            at.sidebar.text_input[0].set_value("https://acme.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)
        rendered = [m.value for m in at.markdown]
        captions = [c.value for c in at.caption]
        self.assertIn("**Website · https://acme.example.com**", rendered)
        self.assertIn("Ready", captions)


class AnswerDisplayTests(unittest.TestCase):

    def test_assistant_message_shows_answer_text_not_the_object(self):
        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]
        fake_answer = Answer(text="The project detects water quality anomalies.")

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.segmented_control[0].set_value("Website")
            at.run(timeout=15)
            at.sidebar.text_input[0].set_value("https://acme.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)
            at.chat_input[0].set_value("What does the project do?")
            at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)

        rendered_markdown = " ".join(m.value for m in at.markdown)
        self.assertIn("The project detects water quality anomalies.", rendered_markdown)
        self.assertNotIn("Answer(text=", rendered_markdown)
        self.assertNotIn("sources=", rendered_markdown)

    def test_sources_rendered_from_answer_sources(self):
        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]
        fake_answer = Answer(
            text="Answer text.",
            sources=[{"source_type": "PDF", "source": "report.pdf", "page": 4}]
        )

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.segmented_control[0].set_value("Website")
            at.run(timeout=15)
            at.sidebar.text_input[0].set_value("https://acme.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)
            at.chat_input[0].set_value("What does the report say?")
            at.run(timeout=15)

        rendered = [m.value for m in at.markdown]
        captions = [c.value for c in at.caption]
        self.assertIn("**Sources**", rendered)
        self.assertIn("Page 4 · report.pdf", captions)

    def test_rejected_question_shows_clear_message_no_sources(self):
        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]
        fake_answer = Answer(text="I could not find the answer in the provided documents.")

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.segmented_control[0].set_value("Website")
            at.run(timeout=15)
            at.sidebar.text_input[0].set_value("https://acme.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)
            at.chat_input[0].set_value("What is the capital of France?")
            at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)
        rendered = [m.value for m in at.markdown]
        self.assertIn("I could not find the answer in the provided documents.", rendered)
        self.assertNotIn("**Sources**", rendered)


class DocumentSwitchTests(unittest.TestCase):

    def test_switching_source_resets_conversation_history(self):
        index_a, chunks_a = MagicMock(name="index_a"), ["a"]
        index_b, chunks_b = MagicMock(name="index_b"), ["b"]
        fake_answer = Answer(text="An answer.")

        with patch("src.ingestion.ingest_url", side_effect=[(index_a, chunks_a), (index_b, chunks_b)]), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(APP_PATH)
            at.run(timeout=15)
            at.sidebar.segmented_control[0].set_value("Website")
            at.run(timeout=15)
            at.sidebar.text_input[0].set_value("https://a.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)
            at.chat_input[0].set_value("Question about A")
            at.run(timeout=15)

            self.assertEqual(len(at.session_state.chat_history), 1)

            at.sidebar.text_input[0].set_value("https://b.example.com")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)

        self.assertEqual(at.session_state.chat_history, [])


if __name__ == "__main__":
    unittest.main()
