import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve
from src.reranker import rerank
from src.prompt_builder import build_prompt
from src.generator import generate_answer


INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions_chunk100.json"

INITIAL_TOP_K = 5
FINAL_TOP_K = 3


def load_questions(path: str):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def keyword_coverage(
    expected_answer: str,
    generated_answer: str
) -> float:

    expected_words = set(
        expected_answer.lower().split()
    )

    generated_words = set(
        generated_answer.lower().split()
    )

    if not expected_words:
        return 0.0

    return len(
        expected_words.intersection(
            generated_words
        )
    ) / len(expected_words)


def main():

    questions = load_questions(
        QUESTIONS_PATH
    )

    index = load_index(
        INDEX_PATH
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    results = []
    coverage_scores = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        query = item["question"]

        retrieved = retrieve(
            index,
            chunks,
            query,
            top_k=INITIAL_TOP_K,
            min_score=0.25
        )

        reranked = rerank(
            query,
            retrieved,
            top_k=FINAL_TOP_K
        )

        prompt = build_prompt(
            query,
            reranked
        )

        answer = generate_answer(
            prompt
        ).strip()

        coverage = keyword_coverage(
            item["expected_answer"],
            answer
        )

        coverage_scores.append(
            coverage
        )

        results.append(
            {
                "question": query,
                "expected_answer":
                    item["expected_answer"],
                "generated_answer": answer,
                "keyword_coverage": coverage,
                "retrieved_chunks": [
                    {
                        "page": result["page"],
                        "chunk_id": result["chunk_id"],
                        "faiss_score": result["distance"],
                        "rerank_score":
                            result["rerank_score"]
                    }
                    for result in reranked
                ]
            }
        )

        print()
        print("=" * 80)
        print(f"QUESTION {number}")
        print("=" * 80)
        print(f"Question: {query}")
        print()
        print(f"Keyword coverage: {coverage:.3f}")
        print(f"Answer: {answer}")

    average_coverage = (
        sum(coverage_scores)
        / len(coverage_scores)
    )

    output = {
        "questions_evaluated": len(questions),
        "retrieval": {
            "initial_top_k": INITIAL_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "threshold": 0.25,
            "reranking": True
        },
        "average_keyword_coverage":
            average_coverage,
        "questions": results
    }

    output_path = (
        "evaluation/reranked_generation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("RERANKED GENERATION EVALUATION")
    print("=" * 80)
    print(
        f"Questions evaluated: "
        f"{len(questions)}"
    )
    print(
        f"Average keyword coverage: "
        f"{average_coverage:.3f}"
    )
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()