import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag import Answer, answer_question, build_sources
from tests.test_document_isolation import ingest_pdf_text


def result(source, page=None, text="some retrieved text", chunk_id=0):
    entry = {"text": text, "source": source, "chunk_id": chunk_id, "distance": 0.9}

    if page is not None:
        entry["page"] = page

    return entry


class BuildSourcesTests(unittest.TestCase):

    def test_pdf_citation(self):
        sources = build_sources([result("report.pdf", page=3)])

        self.assertEqual(
            sources,
            [{"source_type": "PDF", "source": "report.pdf", "page": 3}]
        )

    def test_website_citation(self):
        sources = build_sources([result("https://acme.example.com/pricing", page=1)])

        self.assertEqual(
            sources,
            [{"source_type": "Website", "source": "https://acme.example.com/pricing"}]
        )

    def test_duplicate_source_page_is_deduplicated(self):
        results = [
            result("report.pdf", page=7, chunk_id=0),
            result("report.pdf", page=7, chunk_id=1),
            result("report.pdf", page=7, chunk_id=2),
        ]

        sources = build_sources(results)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["page"], 7)

    def test_ordering_follows_retrieval_order(self):
        results = [
            result("report.pdf", page=5),
            result("report.pdf", page=2),
            result("report.pdf", page=9),
        ]

        sources = build_sources(results)

        self.assertEqual([s["page"] for s in sources], [5, 2, 9])

    def test_missing_source_is_skipped_without_crashing(self):
        results = [
            {"text": "no source field here", "page": 4},
            result("report.pdf", page=1),
        ]

        sources = build_sources(results)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "report.pdf")

    def test_missing_page_is_omitted_not_invented(self):
        results = [{"text": "pdf chunk without a page", "source": "report.pdf"}]

        sources = build_sources(results)

        self.assertEqual(len(sources), 1)
        self.assertNotIn("page", sources[0])
        self.assertEqual(sources[0]["source_type"], "PDF")


def fake_generate_answer(prompt):
    if "Standalone question:" in prompt:
        return prompt.split("Latest question:")[-1].split("Standalone question:")[0].strip()

    return "This is the generated answer."


class AnswerQuestionProvenanceTests(unittest.TestCase):

    def test_sources_reflect_only_the_given_document(self):
        index_a, chunks_a = ingest_pdf_text(
            "Project Zephyr uses a quantum flux capacitor.", "a.pdf"
        )
        index_b, chunks_b = ingest_pdf_text(
            "Project Orion relies on a cryogenic plasma injector.", "b.pdf"
        )

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            answer_a = answer_question("What does the project use?", index_a, chunks_a)
            answer_b = answer_question("What does the project use?", index_b, chunks_b)

        self.assertTrue(all(s["source"] == "a.pdf" for s in answer_a.sources))
        self.assertTrue(all(s["source"] == "b.pdf" for s in answer_b.sources))

    def test_conversation_history_cannot_appear_as_a_source(self):
        index_a, chunks_a = ingest_pdf_text(
            "Project Zephyr uses a quantum flux capacitor.", "a.pdf"
        )

        # A history entry that happens to look like it could be cited,
        # to make sure it's never mistaken for retrieved chunk metadata.
        history = [
            {
                "question": "What is Project Zephyr?",
                "answer": "See page 99 of totally-unrelated.pdf for details."
            }
        ]

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            answer = answer_question(
                "What does it use?", index_a, chunks_a, history=history
            )

        sources_as_text = str(answer.sources)
        self.assertNotIn("totally-unrelated.pdf", sources_as_text)
        self.assertTrue(all(s["source"] == "a.pdf" for s in answer.sources))

    def test_rejected_query_has_no_sources(self):
        index_a, chunks_a = ingest_pdf_text(
            "Project Zephyr uses a quantum flux capacitor.", "a.pdf"
        )

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            answer = answer_question(
                "What is the capital of France?", index_a, chunks_a
            )

        self.assertEqual(answer.sources, [])


class SourcesRenderingTests(unittest.TestCase):
    # AppTest re-executes app.py's source directly, so patches have to
    # target src.ingestion/src.rag (looked up through the normal import
    # system) rather than app.ingest_url etc. - see Task 6's notes.

    def test_sources_section_renders_for_pdf_and_website_entries(self):
        from streamlit.testing.v1 import AppTest

        fake_index, fake_chunks = MagicMock(name="fake_index"), ["chunk"]

        fake_answer = Answer(
            text="The answer text.",
            sources=[
                {"source_type": "PDF", "source": "report.pdf", "page": 3},
                {"source_type": "Website", "source": "https://example.com/page"},
            ]
        )

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(str(ROOT_DIR / "app.py"))
            at.run()
            at.sidebar.radio[0].set_value("Company Website")
            at.run()
            at.sidebar.text_input[0].set_value("https://a.example.com")
            at.run()
            at.sidebar.button[0].click()
            at.run()
            at.chat_input[0].set_value("What does it say?")
            at.run()

        self.assertEqual(at.exception.len, 0)

        rendered = [m.value for m in at.markdown]
        self.assertIn("**Sources**", rendered)
        self.assertIn("- Page 3 — report.pdf", rendered)
        self.assertIn("- https://example.com/page", rendered)

    def test_no_sources_section_when_answer_has_none(self):
        from streamlit.testing.v1 import AppTest

        fake_index, fake_chunks = MagicMock(name="fake_index"), ["chunk"]
        fake_answer = Answer(text="I could not find the answer in the provided documents.")

        with patch("src.ingestion.ingest_url", return_value=(fake_index, fake_chunks)), \
             patch("src.rag.answer_question", return_value=fake_answer):

            at = AppTest.from_file(str(ROOT_DIR / "app.py"))
            at.run()
            at.sidebar.radio[0].set_value("Company Website")
            at.run()
            at.sidebar.text_input[0].set_value("https://a.example.com")
            at.run()
            at.sidebar.button[0].click()
            at.run()
            at.chat_input[0].set_value("What is the capital of France?")
            at.run()

        rendered = [m.value for m in at.markdown]
        self.assertNotIn("**Sources**", rendered)


if __name__ == "__main__":
    unittest.main()
