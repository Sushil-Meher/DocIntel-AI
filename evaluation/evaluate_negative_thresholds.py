import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


INDEX_PATH = "artifacts/faiss.index"
CHUNKS_PATH = "artifacts/chunks.pkl"
QUESTIONS_PATH = "evaluation/negative_questions.json"

THRESHOLDS = [
    0.20,
    0.25,
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


def main():

    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    questions = load_questions(QUESTIONS_PATH)

    print()
    print("=" * 80)
    print("NEGATIVE QUERY THRESHOLD EVALUATION")
    print("=" * 80)

    all_results = []

    for threshold in THRESHOLDS:

        rejected = 0

        print()
        print(f"Threshold: {threshold:.2f}")

        for item in questions:

            results = retrieve(
                index,
                chunks,
                item["question"],
                top_k=3,
                min_score=threshold
            )

            if not results:
                rejected += 1

        total = len(questions)

        rejection_rate = (
            rejected / total
            if total
            else 0.0
        )

        result = {
            "threshold": threshold,
            "rejected": rejected,
            "total": total,
            "rejection_rate": rejection_rate
        }

        all_results.append(result)

        print(
            f"Rejected: {rejected}/{total}"
        )

        print(
            f"Rejection rate: "
            f"{rejection_rate:.3f}"
        )

    output_path = (
        "evaluation/negative_threshold_results.json"
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
    print("=" * 80)
    print(
        f"Results saved to: {output_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()