# src/indexer.py

import faiss
import numpy as np
import json
import os


EMBEDDINGS_PATH = "src/data/embeddings.npy"
METADATA_PATH = "src/data/embeddings_metadata.jsonl"
INDEX_PATH = "src/data/index.faiss"


def build_faiss_index(embeddings_path: str = EMBEDDINGS_PATH,
                      index_path: str = INDEX_PATH):
    """
    Build a FAISS index from precomputed embeddings and save it to disk.
    """

    # Load embeddings
    embeddings = np.load(embeddings_path).astype("float32")

    # Dimensionality of SigLIP embeddings
    dim = embeddings.shape[1]

    # Build index (L2 normalized → use IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(dim)

    # Add vectors
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    print(f"[FAISS] Built index with {embeddings.shape[0]} vectors → {index_path}")


def load_faiss(index_path: str = INDEX_PATH):
    """
    Load a FAISS index from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(index_path)
    return index


def search_faiss(index, query_vector: np.ndarray, top_k: int = 5):
    """
    Run FAISS top‑K search.
    query_vector must be shape (1, dim).
    Returns (ids, distances)
    """

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # FAISS expects float32
    query_vector = query_vector.astype("float32")

    distances, ids = index.search(query_vector, top_k)

    return ids[0].tolist(), distances[0].tolist()


def load_metadata(metadata_path: str = METADATA_PATH):
    """
    Load metadata JSONL into a list where index = FAISS vector ID.
    """
    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def get_metadata_for_ids(metadata_store, ids):
    """
    Given FAISS IDs, return the corresponding metadata objects.
    """
    return [metadata_store[i] for i in ids]

if __name__ == "__main__":
    build_faiss_index()

