import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_chunks


def load_json(path):
    with open(ROOT_DIR / path, "r", encoding="utf-8") as file:
        return json.load(file)


class ExpandedBenchmarkValidationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.positive = load_json("evaluation/questions_expanded.json")
        cls.negative = load_json("evaluation/negative_questions_expanded.json")
        chunks = load_chunks(str(ROOT_DIR / "evaluation/artifacts/chunk100_chunks.pkl"))
        cls.valid_chunk_ids = {(c.page, c.chunk_id) for c in chunks}

    def test_positive_questions_have_gold_evidence(self):
        for item in self.positive:
            self.assertTrue(
                item["relevant_chunks"],
                f"No gold chunks for: {item['question']}"
            )

    def test_referenced_chunk_ids_exist(self):
        for item in self.positive:
            for chunk in item["relevant_chunks"]:
                key = (chunk["page"], chunk["chunk_id"])
                self.assertIn(
                    key, self.valid_chunk_ids,
                    f"{key} referenced by '{item['question']}' does not exist"
                )

    def test_negative_questions_have_no_gold_evidence(self):
        for item in self.negative:
            self.assertEqual(
                item["relevant_chunks"], [],
                f"Negative question has gold evidence: {item['question']}"
            )

    def test_no_duplicate_question_text(self):
        all_questions = [item["question"] for item in self.positive] + \
                         [item["question"] for item in self.negative]

        self.assertEqual(len(all_questions), len(set(all_questions)))

    def test_expected_answers_are_non_empty(self):
        for item in self.positive + self.negative:
            self.assertTrue(item["expected_answer"].strip())

    def test_negative_expected_answers_indicate_unsupported(self):
        # Weak but useful sanity check: a negative's expected answer should
        # read as "not provided", not as an actual answer to the question.
        for item in self.negative:
            self.assertIn("does not", item["expected_answer"].lower())

    def test_benchmark_size_is_in_target_range(self):
        self.assertGreaterEqual(len(self.positive), 25)
        self.assertLessEqual(len(self.positive), 30)
        self.assertGreaterEqual(len(self.negative), 10)
        self.assertLessEqual(len(self.negative), 15)

    def test_category_field_present_and_reasonably_balanced(self):
        categories = [item["category"] for item in self.positive]
        distinct = set(categories)

        self.assertGreaterEqual(len(distinct), 5)
        # no single category should dominate more than half the benchmark
        for category in distinct:
            self.assertLessEqual(categories.count(category), len(categories) // 2 + 2)


class HistoricalBenchmarkUnchangedTests(unittest.TestCase):

    def test_historical_positive_file_unchanged(self):
        questions = load_json("evaluation/questions_chunk100.json")
        self.assertEqual(len(questions), 10)

    def test_historical_negative_file_unchanged(self):
        questions = load_json("evaluation/negative_questions.json")
        self.assertEqual(len(questions), 5)


if __name__ == "__main__":
    unittest.main()
