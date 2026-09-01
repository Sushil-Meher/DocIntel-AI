import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve
from src.reranker import rerank


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


def chunk_key(result: dict):

    return (
        result["page"],
        result["chunk_id"]
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
            item["page"],
            item["chunk_id"]
        )
        for item in relevant_chunks
    }

    return (
        1.0
        if retrieved.intersection(relevant)
        else 0.0
    )


def calculate_mrr(
    results,
    relevant_chunks,
    k
):

    relevant = {
        (
            item["page"],
            item["chunk_id"]
        )
        for item in relevant_chunks
    }

    for rank, result in enumerate(
        results[:k],
        start=1
    ):

        if chunk_key(result) in relevant:

            return 1.0 / rank

    return 0.0


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

    baseline_recall_1 = []
    baseline_mrr_1 = []

    baseline_recall_3 = []
    baseline_mrr_3 = []

    baseline_recall_5 = []
    baseline_mrr_5 = []

    reranked_recall_1 = []
    reranked_mrr_1 = []

    reranked_recall_3 = []
    reranked_mrr_3 = []

    reranked_recall_5 = []
    reranked_mrr_5 = []

    detailed_results = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        query = item["question"]

        relevant_chunks = item[
            "relevant_chunks"
        ]

        # ------------------------------------------------
        # Stage 1: FAISS retrieval
        # ------------------------------------------------

        initial_results = retrieve(
            index,
            chunks,
            query,
            top_k=INITIAL_TOP_K,
            min_score=0.25
        )

        # ------------------------------------------------
        # Stage 2: Cross-encoder reranking
        # ------------------------------------------------

        reranked_results = rerank(
            query,
            initial_results,
            top_k=FINAL_TOP_K
        )

        # ------------------------------------------------
        # Baseline metrics
        # ------------------------------------------------

        baseline_recall_1.append(
            calculate_recall(
                initial_results,
                relevant_chunks,
                1
            )
        )

        baseline_mrr_1.append(
            calculate_mrr(
                initial_results,
                relevant_chunks,
                1
            )
        )

        baseline_recall_3.append(
            calculate_recall(
                initial_results,
                relevant_chunks,
                3
            )
        )

        baseline_mrr_3.append(
            calculate_mrr(
                initial_results,
                relevant_chunks,
                3
            )
        )

        baseline_recall_5.append(
            calculate_recall(
                initial_results,
                relevant_chunks,
                5
            )
        )

        baseline_mrr_5.append(
            calculate_mrr(
                initial_results,
                relevant_chunks,
                5
            )
        )

        # ------------------------------------------------
        # Reranked metrics
        # ------------------------------------------------

        reranked_recall_1.append(
            calculate_recall(
                reranked_results,
                relevant_chunks,
                1
            )
        )

        reranked_mrr_1.append(
            calculate_mrr(
                reranked_results,
                relevant_chunks,
                1
            )
        )

        reranked_recall_3.append(
            calculate_recall(
                reranked_results,
                relevant_chunks,
                3
            )
        )

        reranked_mrr_3.append(
            calculate_mrr(
                reranked_results,
                relevant_chunks,
                3
            )
        )

        reranked_recall_5.append(
            calculate_recall(
                reranked_results,
                relevant_chunks,
                5
            )
        )

        reranked_mrr_5.append(
            calculate_mrr(
                reranked_results,
                relevant_chunks,
                5
            )
        )

        detailed_results.append(
            {
                "question_number": number,
                "question": query,
                "initial_results": [
                    {
                        "page": result["page"],
                        "chunk_id": result["chunk_id"],
                        "faiss_score": result["distance"]
                    }
                    for result in initial_results
                ],
                "reranked_results": [
                    {
                        "page": result["page"],
                        "chunk_id": result["chunk_id"],
                        "faiss_score": result["distance"],
                        "rerank_score": result[
                            "rerank_score"
                        ]
                    }
                    for result in reranked_results
                ]
            }
        )

    n = len(questions)

    results = {
        "questions_evaluated": n,

        "baseline": {
            "recall@1": sum(
                baseline_recall_1
            ) / n,

            "mrr@1": sum(
                baseline_mrr_1
            ) / n,

            "recall@3": sum(
                baseline_recall_3
            ) / n,

            "mrr@3": sum(
                baseline_mrr_3
            ) / n,

            "recall@5": sum(
                baseline_recall_5
            ) / n,

            "mrr@5": sum(
                baseline_mrr_5
            ) / n,
        },

        "reranked": {
            "recall@1": sum(
                reranked_recall_1
            ) / n,

            "mrr@1": sum(
                reranked_mrr_1
            ) / n,

            "recall@3": sum(
                reranked_recall_3
            ) / n,

            "mrr@3": sum(
                reranked_mrr_3
            ) / n,

            "recall@5": sum(
                reranked_recall_5
            ) / n,

            "mrr@5": sum(
                reranked_mrr_5
            ) / n,
        },

        "questions": detailed_results
    }

    output_path = (
        "evaluation/reranking_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print("=" * 80)
    print("RERANKING EXPERIMENT")
    print("=" * 80)

    print("\nBASELINE")
    print(
        f"Recall@1: {results['baseline']['recall@1']:.3f}"
    )
    print(
        f"MRR@1:    {results['baseline']['mrr@1']:.3f}"
    )
    print(
        f"Recall@3: {results['baseline']['recall@3']:.3f}"
    )
    print(
        f"MRR@3:    {results['baseline']['mrr@3']:.3f}"
    )
    print(
        f"Recall@5: {results['baseline']['recall@5']:.3f}"
    )
    print(
        f"MRR@5:    {results['baseline']['mrr@5']:.3f}"
    )

    print("\nRERANKED")
    print(
        f"Recall@1: {results['reranked']['recall@1']:.3f}"
    )
    print(
        f"MRR@1:    {results['reranked']['mrr@1']:.3f}"
    )
    print(
        f"Recall@3: {results['reranked']['recall@3']:.3f}"
    )
    print(
        f"MRR@3:    {results['reranked']['mrr@3']:.3f}"
    )
    print(
        f"Recall@5: {results['reranked']['recall@5']:.3f}"
    )
    print(
        f"MRR@5:    {results['reranked']['mrr@5']:.3f}"
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()