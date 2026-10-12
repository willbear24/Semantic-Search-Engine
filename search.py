"""
Module 7 Project — Semantic Search Engine
==========================================
search.py — query ChromaDB and return ranked results

Import this module into app.py:
    from search import search, get_collection_stats

Semantic Search (25 points)
Accept a text query and return the top-N most relevant chunks
Display results with: the matched text, source file, similarity/distance score, and relevance indicator
Support configurable n_results and similarity threshold
Handle edge cases: empty queries, no results above threshold
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, util
from chromadb.errors import NotFoundError

# ── Configuration (must match ingest.py) ─────────────────────────────────────
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """Return the embedding model, loading it on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    """Return the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except NotFoundError as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. Run `python ingest.py` first."
        ) from e


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:
    """
    Search the ChromaDB collection and return ranked results.

    Args:
        query:              Natural language search query.
        n_results:          Maximum number of results to return.
        sources:            If provided, only return chunks from these filenames.
        distance_threshold: If provided, exclude results with distance above this
                            value (lower = more similar).

    Returns:
        List of result dicts sorted by distance ascending (best first):
            {
                "text":        str,
                "source":      str,
                "chunk_index": int,
                "distance":    float,
                "score":       float,  # 1 - distance
            }
        Returns [] for empty queries or if the collection has no documents.
    """
    query = query.strip()
    if not query:
        return []

    collection = get_collection()
    collection_count = collection.count()
    if not collection_count:
        return []

    model = get_model()
    query_embedding = model.encode([query]).tolist()

    where = None
    if sources:
        where = {"source": {"$in": sources}}

    actual_n_results = min(n_results, collection_count)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=actual_n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    search_results = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        if distance_threshold is not None and distance > distance_threshold:
            continue
        search_results.append(
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
                "score": 1 - distance,
            }
        )

    return search_results


def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks":   int,
            "unique_sources": int,
            "source_names":   list[str],
        }
    """
    collection = get_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        return {"total_chunks": 0, "unique_sources": 0, "source_names": []}
    all_metadata = collection.get(include=["metadatas"])
    unique_sources = set()
    for item in all_metadata["metadatas"]:
        unique_sources.add(item["source"])
    source_names = sorted(unique_sources)
    return {
        "total_chunks": total_chunks,
        "unique_sources": len(source_names),
        "source_names": source_names,
    }
