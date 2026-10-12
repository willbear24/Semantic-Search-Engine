"""
Module 7 Project — Semantic Search Engine
==========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.

Chunking Experiment (20 points)
Re-index documents with at least 2 different chunk sizes (e.g., 200 chars vs 500 chars)
Run the same 5 test queries against both versions
Compare results: which chunk size produced better results for each query?
Document your findings in the README (which size worked better overall and why)
"""

import argparse
from search import search
import re

# Define your evaluation set here.
# Each entry needs a query and the source filenames you expect to be relevant.
# Use at least 5 queries for the chunking experiment.
EVAL_SET = [
    {
        "query": "What is the definition and use case of a Model in FastAPI?",
        "relevant_sources": ["fastapi.txt"],
    },
    {
        "query": "How do I make sure only authorized users can access my API?",
        "relevant_sources": ["rest-apis.txt", "fastapi.txt"],
    },
    {
        "query": "How can I avoid recomputing expensive results on repeated calls?",
        "relevant_sources": ["python-advanced.txt", "streamlit.txt"],
    },
    {
        "query": "How do I stop a language model from making up facts?",
        "relevant_sources": ["llms-and-ai.txt", "embeddings-and-vectors.txt"],
    },
    {
        "query": "What is the difference between an INNER JOIN and a LEFT JOIN?",
        "relevant_sources": ["sql-databases.txt"],
    },
    {
        "query": "How do I handle errors and make sure resources get cleaned up?",
        "relevant_sources": ["python-fundamentals.txt", "python-advanced.txt"],
    },
]


def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """
    Compute source-level precision and recall.

    Returns (precision, recall) as floats in [0, 1].
    """
    retrieved_set = set(retrieved_sources)
    relevant_set = set(relevant_sources)

    overlap = retrieved_set & relevant_set

    precision = len(overlap) / len(retrieved_set) if retrieved_sources else 0.0

    recall = len(overlap) / len(relevant_set) if relevant_sources else 0.0

    return float(precision), float(recall)


def evaluate(n_results: int = 5, distance_threshold: float = None):
    """
    Run every query in EVAL_SET and print per-query and average precision/recall.
    """

    if not EVAL_SET:
        print("There is no data in the evaluation set.")
        return (0.0, 0.0)

    total_precision = 0
    total_recall = 0

    print(
        f"\nEvaluation (n_results={n_results}, distance_threshold={distance_threshold})"
    )
    print("=" * 60)

    for item in EVAL_SET:
        query = item["query"]
        relevant = item["relevant_sources"]
        results = search(
            query, n_results=n_results, distance_threshold=distance_threshold
        )
        retrieved = [result["source"] for result in results]
        precision, recall = precision_recall(retrieved, relevant)
        total_precision += precision
        total_recall += recall

        print(f"Query: {query}")
        print(f"  Retrieved: {sorted(set(retrieved))}")
        print(f"  Relevant:  {sorted(set(relevant))}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")

    avg_precision = total_precision / len(EVAL_SET)
    avg_recall = total_recall / len(EVAL_SET)

    print("=" * 60)
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall:    {avg_recall:.2f}")

    return avg_precision, avg_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate(n_results=args.n_results, distance_threshold=args.threshold)
