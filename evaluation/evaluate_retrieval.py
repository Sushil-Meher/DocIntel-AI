import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))


from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


TOP_K_VALUES = [1, 3, 5, 8, 10]

QUESTIONS_PATH = "evaluation/questions_chunk100.json"
RESULTS_PATH = "evaluation/results.json"


def load_questions(path: str):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def chunk_key(result: dict) -> tuple:
    return (
        result["page"],
        result["chunk_id"]
    )


def evaluate_retrieval(questions_path: str = QUESTIONS_PATH):

    questions = load_questions(
        questions_path
    )

    # Both benchmarks retrieve against the same chunk100 index/chunks -
    # the production chunking/embedding configuration is unchanged here,
    # only the question set differs.
    index = load_index(
        "evaluation/artifacts/chunk100.index"
    )

    chunks = load_chunks(
        "evaluation/artifacts/chunk100_chunks.pkl"
    )

    metrics = {}

    for k in TOP_K_VALUES:

        hits = 0
        reciprocal_ranks = []

        for item in questions:

            results = retrieve(
                index,
                chunks,
                item["question"],
                top_k=k
            )

            relevant_chunks = {
                (
                    chunk["page"],
                    chunk["chunk_id"]
                )
                for chunk in item["relevant_chunks"]
            }

            first_relevant_rank = None

            for rank, result in enumerate(
                results,
                start=1
            ):

                if chunk_key(result) in relevant_chunks:

                    first_relevant_rank = rank
                    break

            if first_relevant_rank is not None:

                hits += 1

                reciprocal_ranks.append(
                    1 / first_relevant_rank
                )

            else:

                reciprocal_ranks.append(0)

        total_questions = len(questions)

        metrics[f"recall@{k}"] = (
            hits / total_questions
        )

        metrics[f"mrr@{k}"] = (
            sum(reciprocal_ranks)
            / total_questions
        )

    return metrics


if __name__ == "__main__":

    # Default (no argument) reproduces the historical 10-question
    # benchmark exactly, writing to evaluation/results.json as always.
    # "expanded" runs the larger benchmark and writes to a separate file
    # so the historical results are never overwritten.
    if len(sys.argv) > 1 and sys.argv[1] == "expanded":
        questions_path = "evaluation/questions_expanded.json"
        results_path = "evaluation/expanded_results.json"
    else:
        questions_path = QUESTIONS_PATH
        results_path = RESULTS_PATH

    results = evaluate_retrieval(questions_path)

    print()
    print("RAG RETRIEVAL EVALUATION")
    print(f"Questions: {questions_path}")
    print("=" * 35)

    for metric, value in results.items():

        print(
            f"{metric}: {value:.3f}"
        )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )