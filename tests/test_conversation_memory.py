import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag import Answer, answer_question
from tests.test_document_isolation import ingest_pdf_text


DOC_A_TEXT = (
    "Project Zephyr uses a quantum flux capacitor to stabilize the "
    "graviton lattice. The zephyr calibration constant is 42. "
    "The zephyr project lead is Dr. Mensah."
)

DOC_B_TEXT = (
    "Project Orion relies on a cryogenic plasma injector to cool the "
    "fusion core. The orion pressure threshold is 917 kPa."
)


def fake_generate_answer(prompt):
    # Stands in for the real Qwen model so tests are fast and
    # deterministic. For a query-rewrite prompt it does a naive
    # pronoun substitution using whichever project name shows up in the
    # conversation text, mimicking what the real model does (verified
    # separately against the real model - see experiments.md). For a
    # final-answer prompt it just echoes back the retrieved context, so
    # tests can assert on exactly what evidence reached the prompt.
    if "Standalone question:" in prompt:
        match = re.search(r"Latest question: (.+)", prompt)
        latest = match.group(1).strip() if match else ""

        if "zephyr" in prompt.lower():
            return latest.replace("its", "the zephyr's").replace(" it ", " the zephyr ")
        if "orion" in prompt.lower():
            return latest.replace("its", "the orion's").replace(" it ", " the orion ")

        return latest

    texts = re.findall(r"Text: (.+)", prompt)
    return " | ".join(texts) if texts else "I could not find the answer in the provided documents."


class ContextualQueryTests(unittest.TestCase):

    def test_followup_resolves_using_history(self):
        index_a, chunks_a = ingest_pdf_text(DOC_A_TEXT, "a.pdf")

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            history = []

            a1 = answer_question(
                "What is Project Zephyr?", index_a, chunks_a, history=history
            )
            history.append({"question": "What is Project Zephyr?", "answer": a1.text})

            a2 = answer_question(
                "What is its calibration constant?", index_a, chunks_a, history=history
            )

        self.assertIn("42", a2.text)

    def test_unrelated_followup_still_works(self):
        index_a, chunks_a = ingest_pdf_text(DOC_A_TEXT, "a.pdf")

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            history = [
                {
                    "question": "What is its calibration constant?",
                    "answer": "The zephyr calibration constant is 42.",
                }
            ]

            # Already standalone - no pronoun to resolve, and about a
            # different fact entirely. Should not get dragged off course
            # by the unrelated prior turn.
            answer = answer_question(
                "Who leads Project Zephyr?", index_a, chunks_a, history=history
            )

        self.assertIn("Mensah", answer.text)

    def test_history_from_wrong_document_cannot_leak_facts(self):
        # Simulates a worst-case bug where the caller mixed up documents
        # and passed A's conversation history alongside B's index/chunks.
        # The grounding rule should hold structurally: retrieve() only
        # ever searches the index/chunks it's given, so B's answer can't
        # contain A's facts even if the rewritten query gets confused.
        index_b, chunks_b = ingest_pdf_text(DOC_B_TEXT, "b.pdf")

        stale_history_from_a = [
            {"question": "What is Project Zephyr?", "answer": "Project Zephyr is a quantum system."}
        ]

        with patch("src.rag.generate_answer", side_effect=fake_generate_answer):
            answer = answer_question(
                "What is its pressure threshold?",
                index_b,
                chunks_b,
                history=stale_history_from_a
            )

        self.assertIn("917", answer.text)
        self.assertNotIn("42", answer.text)
        self.assertNotIn("zephyr", answer.text.lower())


class SessionResetTests(unittest.TestCase):
    # AppTest re-executes app.py's source directly rather than reusing
    # the imported `app` module, so mocks have to target the underlying
    # src.ingestion/src.rag functions (which are looked up via the
    # normal import system and so do pick up patches) rather than
    # `app.ingest_url` etc.

    def test_processing_new_source_resets_conversation_history(self):
        from streamlit.testing.v1 import AppTest

        index_a, chunks_a = MagicMock(name="index_a"), ["a-chunk"]
        index_b, chunks_b = MagicMock(name="index_b"), ["b-chunk"]

        calls = []

        def fake_answer_question(query, index, chunks, history=None, **kwargs):
            calls.append({"index": index, "history": list(history or [])})
            return Answer(text=f"answer about {index}")

        with patch("src.ingestion.ingest_url", side_effect=[(index_a, chunks_a), (index_b, chunks_b)]), \
             patch("src.rag.answer_question", side_effect=fake_answer_question):

            at = AppTest.from_file(str(ROOT_DIR / "app.py"))
            at.run()

            at.sidebar.radio[0].set_value("Company Website")
            at.run()
            at.sidebar.text_input[0].set_value("https://a.example.com")
            at.run()
            at.sidebar.button[0].click()
            at.run()

            self.assertEqual(at.session_state.chat_history, [])

            at.chat_input[0].set_value("What is A about?")
            at.run()

            self.assertEqual(len(at.session_state.chat_history), 1)

            at.chat_input[0].set_value("What else does it cover?")
            at.run()

            self.assertEqual(len(at.session_state.chat_history), 2)
            # second call should have seen the first turn as history
            self.assertEqual(len(calls[1]["history"]), 1)

            # switch to a different website - history must reset, and the
            # next question must retrieve against the new index only
            at.sidebar.text_input[0].set_value("https://b.example.com")
            at.run()
            at.sidebar.button[0].click()
            at.run()

            self.assertEqual(at.session_state.chat_history, [])

            at.chat_input[0].set_value("What is B about?")
            at.run()

            self.assertEqual(len(at.session_state.chat_history), 1)
            last_call = calls[-1]
            self.assertIs(last_call["index"], index_b)
            self.assertEqual(last_call["history"], [])


if __name__ == "__main__":
    unittest.main()
