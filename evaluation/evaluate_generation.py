import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve
from src.prompt_builder import build_prompt
from src.generator import generate_answer
from src.embedding import create_embeddings


INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions_chunk100.json"

TOP_K = 10


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


def semantic_similarity(
    expected_answer: str,
    generated_answer: str
) -> float:

    # create_embeddings() already L2-normalizes, so the dot product
    # of the two vectors is the cosine similarity.
    expected_vector, generated_vector = create_embeddings(
        [expected_answer, generated_answer]
    )

    return float(np.dot(expected_vector, generated_vector))


def evaluate_generation(questions_path: str = QUESTIONS_PATH):

    questions = load_questions(questions_path)

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
            top_k=TOP_K,
            min_score=0.25
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

        similarity = semantic_similarity(
            item["expected_answer"],
            answer
        )

        results.append(
            {
                "question": question,
                "expected_answer": item["expected_answer"],
                "generated_answer": answer,
                "keyword_coverage": coverage,
                "semantic_similarity": similarity,
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

        print(
            f"Semantic similarity: "
            f"{similarity:.3f}"
        )

    average_coverage = (
        sum(
            result["keyword_coverage"]
            for result in results
        )
        / len(results)
    )

    average_similarity = (
        sum(
            result["semantic_similarity"]
            for result in results
        )
        / len(results)
    )

    evaluation = {
        "questions_evaluated": len(results),
        "top_k": TOP_K,
        "average_keyword_coverage": average_coverage,
        "average_semantic_similarity": average_similarity,
        "questions": results
    }

    results_path = (
        "evaluation/expanded_generation_results.json"
        if questions_path == "evaluation/questions_expanded.json"
        else "evaluation/generation_results.json"
    )

    with open(
        results_path,
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

    print(
        f"Average semantic similarity: "
        f"{average_similarity:.3f}"
    )


if __name__ == "__main__":

    # Default (no argument) reproduces the historical 10-question
    # benchmark exactly, writing to evaluation/generation_results.json as
    # always. "expanded" runs the larger benchmark and writes to a
    # separate file so the historical results are never overwritten.
    if len(sys.argv) > 1 and sys.argv[1] == "expanded":
        evaluate_generation("evaluation/questions_expanded.json")
    else:
        evaluate_generation()