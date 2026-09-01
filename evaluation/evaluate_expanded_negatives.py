import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


# Same production threshold and index as the expanded positive benchmark -
# this script only reports how 0.25 behaves on a larger negative set, it
# does not recalibrate it.
INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/negative_questions_expanded.json"
THRESHOLD = 0.25


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    questions = load_questions(QUESTIONS_PATH)

    results = []
    rejected = 0

    for item in questions:

        retrieved = retrieve(
            index,
            chunks,
            item["question"],
            top_k=1,
            min_score=THRESHOLD
        )

        was_rejected = not retrieved

        if was_rejected:
            rejected += 1

        results.append(
            {
                "question": item["question"],
                "category": item.get("category"),
                "rejected": was_rejected,
                "top_score": retrieved[0]["distance"] if retrieved else None
            }
        )

    total = len(questions)

    output = {
        "threshold": THRESHOLD,
        "total": total,
        "rejected": rejected,
        "rejection_rate": rejected / total if total else 0.0,
        "questions": results
    }

    with open(
        "evaluation/expanded_negative_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=4)

    print()
    print("EXPANDED NEGATIVE-QUERY REJECTION")
    print("=" * 40)
    print(f"Threshold: {THRESHOLD}")
    print(f"Rejected: {rejected}/{total}")
    print(f"Rejection rate: {output['rejection_rate']:.3f}")

    for item in results:
        if not item["rejected"]:
            print(f"NOT REJECTED: {item['question']} (score {item['top_score']:.3f})")


if __name__ == "__main__":
    main()
