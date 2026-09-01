import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


INDEX_PATH = "evaluation/artifacts/chunk100.index"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"
QUESTIONS_PATH = "evaluation/questions.json"


with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


index = load_index(INDEX_PATH)
chunks = load_chunks(CHUNKS_PATH)


for number, item in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {number}")
    print("=" * 80)

    print(item["question"])

    results = retrieve(
        index,
        chunks,
        item["question"],
        top_k=5
    )

    for rank, result in enumerate(results, start=1):

        print("\n" + "-" * 80)
        print(f"RANK: {rank}")
        print(f"CHUNK ID: {result['chunk_id']}")
        print(f"PAGE: {result['page']}")
        print(f"DISTANCE: {result['distance']:.4f}")
        print(f"TEXT: {result['text']}")