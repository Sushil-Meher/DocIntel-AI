import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))


from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


def load_questions(path: str):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


if __name__ == "__main__":

    questions = load_questions(
        "evaluation/questions.json"
    )

    index = load_index(
        "evaluation/artifacts/baseline.index"
    )

    chunks = load_chunks(
        "evaluation/artifacts/baseline_chunks.pkl"
    )

    print(f"Total chunks: {len(chunks)}")

    for question_number, item in enumerate(
        questions,
        start=1
    ):

        print("\n" + "=" * 80)
        print(f"QUESTION {question_number}")
        print("=" * 80)

        print(item["question"])

        results = retrieve(
            index,
            chunks,
            item["question"],
            top_k=5
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "-" * 80)
            print(f"RANK: {rank}")
            print(f"CHUNK ID: {result['chunk_id']}")
            print(f"PAGE: {result['page']}")
            print(f"DISTANCE: {result['distance']:.4f}")
            print(f"TEXT: {result['text']}")