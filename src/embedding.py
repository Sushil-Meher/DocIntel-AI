from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def create_embeddings(texts: list[str]) -> list[list[float]]:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts)

    return embeddings.tolist()

