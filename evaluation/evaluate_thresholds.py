import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions_chunk100.json"

TOP_K = 5

THRESHOLDS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.35,
    0.40,
]


def load_questions(path: str):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def chunk_key(item):
    return (
        item["page"],
        item["chunk_id"]
    )


def calculate_recall(
    results,
    relevant_chunks,
    k
):
    retrieved = {
        chunk_key(result)
        for result in results[:k]
    }

    relevant = {
        (
            chunk["page"],
            chunk["chunk_id"]
        )
        for chunk in relevant_chunks
    }

    return 1.0 if retrieved.intersection(relevant) else 0.0


def calculate_mrr(
    results,
    relevant_chunks,
    k
):
    relevant = {
        (
            chunk["page"],
            chunk["chunk_id"]
        )
        for chunk in relevant_chunks
    }

    for rank, result in enumerate(
        results[:k],
        start=1
    ):
        if chunk_key(result) in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_threshold(threshold):
    questions = load_questions(
        QUESTIONS_PATH
    )

    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []

    mrr_at_1 = []
    mrr_at_3 = []
    mrr_at_5 = []

    rejected = 0

    for item in questions:

        results = retrieve(
            index,
            chunks,
            item["question"],
            top_k=TOP_K,
            min_score=threshold
        )

        if not results:
            rejected += 1

        relevant_chunks = item[
            "relevant_chunks"
        ]

        recall_at_1.append(
            calculate_recall(
                results,
                relevant_chunks,
                1
            )
        )

        recall_at_3.append(
            calculate_recall(
                results,
                relevant_chunks,
                3
            )
        )

        recall_at_5.append(
            calculate_recall(
                results,
                relevant_chunks,
                5
            )
        )

        mrr_at_1.append(
            calculate_mrr(
                results,
                relevant_chunks,
                1
            )
        )

        mrr_at_3.append(
            calculate_mrr(
                results,
                relevant_chunks,
                3
            )
        )

        mrr_at_5.append(
            calculate_mrr(
                results,
                relevant_chunks,
                5
            )
        )

    n = len(questions)

    return {
        "threshold": threshold,
        "recall@1": sum(recall_at_1) / n,
        "mrr@1": sum(mrr_at_1) / n,
        "recall@3": sum(recall_at_3) / n,
        "mrr@3": sum(mrr_at_3) / n,
        "recall@5": sum(recall_at_5) / n,
        "mrr@5": sum(mrr_at_5) / n,
        "rejected_queries": rejected,
    }


def main():

    all_results = []

    print()
    print("=" * 90)
    print("RAG RETRIEVAL THRESHOLD EVALUATION")
    print("=" * 90)

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            threshold
        )

        all_results.append(result)

        print()
        print(
            f"Threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"Recall@1: "
            f"{result['recall@1']:.3f}"
        )

        print(
            f"MRR@1: "
            f"{result['mrr@1']:.3f}"
        )

        print(
            f"Recall@3: "
            f"{result['recall@3']:.3f}"
        )

        print(
            f"MRR@3: "
            f"{result['mrr@3']:.3f}"
        )

        print(
            f"Recall@5: "
            f"{result['recall@5']:.3f}"
        )

        print(
            f"MRR@5: "
            f"{result['mrr@5']:.3f}"
        )

        print(
            f"Rejected queries: "
            f"{result['rejected_queries']}"
        )

    output_path = (
        "evaluation/threshold_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    print()
    print("=" * 90)
    print(
        f"Results saved to: "
        f"{output_path}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()