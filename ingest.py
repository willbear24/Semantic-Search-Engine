"""
Module 7 Project — Semantic Search Engine
==========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50

Document Ingestion (20 points)
Load documents from a docs/ folder (at least 8 documents, 5+ pages of total content)
Implement a clear chunking strategy with configurable chunk size and overlap
Store chunks in a persistent ChromaDB collection with metadata (source file, chunk index)
Documents can be: course notes, technical documentation, Wikipedia articles, or the provided starter corpus
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR = Path("docs")
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100



def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be greater than 0, and overlap must be between 0 and chunk_size - 1."
        )

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If this isn't the last chunk, try to end at the last sentence boundary in the window.
        if end < len(text):
            window = text[start:end]
            boundaries = list(re.finditer(r"[.!?]\s+", window))
            # Only use a boundary that leaves more than `overlap` chars, so the loop always advances.
            if boundaries and boundaries[-1].end() > overlap:
                end = start + boundaries[-1].end()

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Returns:
        List of dicts: {"filename": str, "text": str}
    """
    docs = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".txt", ".md")):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        docs.append({"filename": filename, "text": content})
    return docs


def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )


def ingest(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Each chunk is stored with metadata: source filename, chunk index,
    and the chunk size used — so experiments with different sizes can
    be compared without ambiguity.
    """

    collection = get_collection(
        chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME
    )
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No .txt or .md files found in {DOCS_DIR}.")
        return 0
    texts = []
    metadatas = []
    ids = []

    for document in documents:
        collection.delete(where={"source": document["filename"]})
        chunks = chunk_text(
            text=document["text"], chunk_size=chunk_size, overlap=overlap
        )
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": document["filename"],
                    "chunk_index": i,
                    "chunk_size": chunk_size,
                }
            )
            ids.append(f"{document['filename']}_{i}")

    model = SentenceTransformer(MODEL_NAME)
    chunk_embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=chunk_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Indexed {len(texts)} chunks from {len(documents)} documents.")
    return len(texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
