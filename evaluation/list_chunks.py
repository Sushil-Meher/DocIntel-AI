import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_chunks


chunks = load_chunks(
    "evaluation/artifacts/baseline_chunks.pkl"
)

with open(
    "evaluation/questions.json",
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


print(f"TOTAL CHUNKS: {len(chunks)}")


for q_num, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {q_num}")
    print(question["question"])
    print("=" * 80)

    keywords = question.get("expected_keywords", [])

    scored_chunks = []

    for chunk in chunks:

        text = chunk.text.lower()

        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword.lower() in text
        ]

        if matched_keywords:

            score = len(matched_keywords)

            scored_chunks.append(
                (
                    score,
                    chunk.chunk_id,
                    chunk.page,
                    matched_keywords,
                    chunk.text
                )
            )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    for score, chunk_id, page, matched, text in scored_chunks[:5]:

        print("\n" + "-" * 80)
        print(f"CHUNK ID: {chunk_id}")
        print(f"PAGE: {page}")
        print(f"MATCHES: {matched}")
        print(f"SCORE: {score}")
        print(f"TEXT: {text}")