import faiss
import numpy as np
import pickle

from .embedding import create_embeddings
from .chunking import chunk_document
from .document_loader import load_pdf

def create_index(chunks):

    texts = [chunk.text for chunk in chunks]

    embeddings = create_embeddings(texts)

    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def save_index(index, path: str):
    faiss.write_index(index, path)


def load_index(path: str):
    return faiss.read_index(path)

def save_chunks(chunks, path: str):
    with open(path, "wb") as file:
        pickle.dump(chunks, file)


def load_chunks(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)


