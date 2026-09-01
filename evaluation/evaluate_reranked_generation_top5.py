import json
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------
# RAG imports
# ---------------------------------------------------------

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve
from src.reranker import rerank
from src.prompt_builder import build_prompt
from src.generator import generate_answer


# ---------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------

INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions_chunk100.json"

INITIAL_TOP_K = 5
FINAL_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.25


# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

def load_questions(path: str) -> list[dict]:

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Keyword coverage
# ---------------------------------------------------------

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

    return (
        len(
            expected_words.intersection(
                generated_words
            )
        )
        / len(expected_words)
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

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

    print()
    print("=" * 80)
    print("RERANKED GENERATION EVALUATION - TOP 5")
    print("=" * 80)

    print(
        f"Initial Top-K: {INITIAL_TOP_K}"
    )

    print(
        f"Final Top-K:   {FINAL_TOP_K}"
    )

    print(
        f"Threshold:     {MIN_RELEVANCE_SCORE}"
    )

    # -----------------------------------------------------
    # Evaluate every question
    # -----------------------------------------------------

    for number, item in enumerate(
        questions,
        start=1
    ):

        query = item["question"]

        expected_answer = item[
            "expected_answer"
        ]

        # -------------------------------------------------
        # Stage 1: FAISS retrieval
        # -------------------------------------------------

        retrieved = retrieve(
            index,
            chunks,
            query,
            top_k=INITIAL_TOP_K,
            min_score=MIN_RELEVANCE_SCORE
        )

        # -------------------------------------------------
        # Stage 2: Cross-encoder reranking
        # -------------------------------------------------

        reranked = rerank(
            query,
            retrieved,
            top_k=FINAL_TOP_K
        )

        # -------------------------------------------------
        # Stage 3: Build grounded prompt
        # -------------------------------------------------

        prompt = build_prompt(
            query,
            reranked
        )

        # -------------------------------------------------
        # Stage 4: Generate answer
        # -------------------------------------------------

        answer = generate_answer(
            prompt
        ).strip()

        # -------------------------------------------------
        # Stage 5: Keyword coverage
        # -------------------------------------------------

        coverage = keyword_coverage(
            expected_answer,
            answer
        )

        coverage_scores.append(
            coverage
        )

        # -------------------------------------------------
        # Save question result
        # -------------------------------------------------

        results.append(
            {
                "question": query,

                "expected_answer":
                    expected_answer,

                "generated_answer":
                    answer,

                "keyword_coverage":
                    coverage,

                "retrieved_chunks": [
                    {
                        "page":
                            result["page"],

                        "chunk_id":
                            result["chunk_id"],

                        "faiss_score":
                            result["distance"],

                        "rerank_score":
                            result["rerank_score"]
                    }
                    for result in reranked
                ]
            }
        )

        # -------------------------------------------------
        # Console output
        # -------------------------------------------------

        print()
        print("-" * 80)

        print(
            f"QUESTION {number}"
        )

        print("-" * 80)

        print(
            f"Question: {query}"
        )

        print()

        print(
            f"Keyword coverage: "
            f"{coverage:.3f}"
        )

        print()

        print(
            "Answer:"
        )

        print(answer)

    # -----------------------------------------------------
    # Average score
    # -----------------------------------------------------

    average_coverage = (
        sum(coverage_scores)
        / len(coverage_scores)
    )

    # -----------------------------------------------------
    # Final output object
    # -----------------------------------------------------

    output = {

        "questions_evaluated":
            len(questions),

        "retrieval": {

            "initial_top_k":
                INITIAL_TOP_K,

            "final_top_k":
                FINAL_TOP_K,

            "threshold":
                MIN_RELEVANCE_SCORE,

            "reranking":
                True
        },

        "average_keyword_coverage":
            average_coverage,

        "questions":
            results
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output_path = (
        "evaluation/"
        "reranked_generation_top5_results.json"
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

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Questions evaluated: "
        f"{len(questions)}"
    )

    print(
        f"Average keyword coverage: "
        f"{average_coverage:.3f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()