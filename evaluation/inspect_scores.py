import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vector_store import load_index, load_chunks
from src.retriever import retrieve


index = load_index(
    "artifacts/faiss.index"
)

chunks = load_chunks(
    "artifacts/chunks.pkl"
)


queries = [
    "What are the water-quality variables identified as inputs to the project?",
    "What is the main objective of the project?",
    "What are the mandatory final submission deliverables?",
    "What is the capital of France?"
]


for query in queries:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = retrieve(
        index,
        chunks,
        query,
        top_k=3
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank} | "
            f"Score: {result['distance']:.4f} | "
            f"Page: {result['page']} | "
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Text: {result['text'][:400]}"
        )