import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve
from src.prompt_builder import build_prompt
from src.generator import generate_answer


INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions_chunk100.json"

TOP_K = 3


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def keyword_coverage(
    expected_answer: str,
    generated_answer: str
) -> float:

    expected_tokens = set(
        normalize_text(expected_answer)
    )

    generated_tokens = set(
        normalize_text(generated_answer)
    )

    if not expected_tokens:
        return 0.0

    matched = expected_tokens.intersection(
        generated_tokens
    )

    return len(matched) / len(expected_tokens)


def evaluate_generation():

    questions = load_questions(QUESTIONS_PATH)

    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    results = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        retrieved = retrieve(
            index,
            chunks,
            question,
            top_k=TOP_K
        )

        prompt = build_prompt(
            question,
            retrieved
        )

        answer = generate_answer(prompt)

        coverage = keyword_coverage(
            item["expected_answer"],
            answer
        )

        results.append(
            {
                "question": question,
                "expected_answer": item["expected_answer"],
                "generated_answer": answer,
                "keyword_coverage": coverage,
                "retrieved_chunks": [
                    {
                        "page": result["page"],
                        "chunk_id": result["chunk_id"],
                        "distance": result["distance"]
                    }
                    for result in retrieved
                ]
            }
        )

        print()
        print("=" * 80)
        print(f"QUESTION {number}")
        print("=" * 80)

        print(f"\nQuestion:\n{question}")

        print(
            f"\nExpected answer:\n"
            f"{item['expected_answer']}"
        )

        print(
            f"\nGenerated answer:\n"
            f"{answer}"
        )

        print(
            f"\nKeyword coverage: "
            f"{coverage:.3f}"
        )

    average_coverage = (
        sum(
            result["keyword_coverage"]
            for result in results
        )
        / len(results)
    )

    evaluation = {
        "questions_evaluated": len(results),
        "top_k": TOP_K,
        "average_keyword_coverage": average_coverage,
        "questions": results
    }

    with open(
        "evaluation/generation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("GENERATION BASELINE")
    print("=" * 80)

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    print(
        f"Average keyword coverage: "
        f"{average_coverage:.3f}"
    )


if __name__ == "__main__":
    evaluate_generation()