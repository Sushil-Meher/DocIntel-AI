import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))


from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


TOP_K_VALUES = [1, 3, 5]


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


def evaluate_retrieval():

    questions = load_questions(
        "evaluation/questions.json"
    )

    index = load_index(
        "evaluation/artifacts/baseline.index"
    )

    chunks = load_chunks(
        "evaluation/artifacts/baseline_chunks.pkl"
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

    results = evaluate_retrieval()

    print()
    print("RAG RETRIEVAL BASELINE")
    print("=" * 35)

    for metric, value in results.items():

        print(
            f"{metric}: {value:.3f}"
        )

    with open(
        "evaluation/results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )