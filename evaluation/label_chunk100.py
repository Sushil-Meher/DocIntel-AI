import json
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_chunks


QUESTIONS_PATH = "evaluation/questions.json"
CHUNKS_PATH = "evaluation/artifacts/chunk100_chunks.pkl"

OUTPUT_TXT = "evaluation/chunk100_candidates.txt"
OUTPUT_JSON = "evaluation/chunk100_candidates.json"


def tokenize(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


chunks = load_chunks(CHUNKS_PATH)

text_output = []
json_output = []


for number, question in enumerate(
    questions,
    start=1
):

    expected_answer = question["expected_answer"]

    expected_tokens = tokenize(
        expected_answer
    )

    scored = []

    for chunk in chunks:

        chunk_tokens = tokenize(chunk.text)

        matches = expected_tokens.intersection(
            chunk_tokens
        )

        score = len(matches)

        if score > 0:

            scored.append(
                {
                    "score": score,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "matches": sorted(matches),
                    "text": chunk.text
                }
            )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Keep the strongest 5 candidates.
    top_candidates = scored[:5]

    json_output.append(
        {
            "question_number": number,
            "question": question["question"],
            "candidates": top_candidates
        }
    )

    text_output.append(
        "\n" + "=" * 100
    )
    text_output.append(
        f"QUESTION {number}"
    )
    text_output.append(
        "=" * 100
    )
    text_output.append(
        question["question"]
    )

    for rank, candidate in enumerate(
        top_candidates,
        start=1
    ):

        text_output.append(
            "\n" + "-" * 100
        )
        text_output.append(
            f"CANDIDATE RANK: {rank}"
        )
        text_output.append(
            f"SCORE: {candidate['score']}"
        )
        text_output.append(
            f"PAGE: {candidate['page']}"
        )
        text_output.append(
            f"CHUNK ID: {candidate['chunk_id']}"
        )
        text_output.append(
            f"MATCHES: {candidate['matches']}"
        )
        text_output.append(
            "\nTEXT:"
        )
        text_output.append(
            candidate["text"]
        )


with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(text_output)
    )


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        json_output,
        file,
        indent=4,
        ensure_ascii=False
    )


print("Candidate chunk files created:")
print(f"TXT : {OUTPUT_TXT}")
print(f"JSON: {OUTPUT_JSON}")