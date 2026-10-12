"""
Module 7 Project — Semantic Search Engine
==========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py

Code Quality & Documentation (15 points)
Modular code (separate files for ingestion, search, evaluation, UI)
Functions with docstrings
Working requirements.txt
README with: setup instructions, architecture description, chunking experiment results
"""

import streamlit as st
from search import search, get_collection_stats
from ingest import ingest, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")

try:
    stats = get_collection_stats()
except RuntimeError:
    st.error(
        "Run `python ingest.py` or use the sidebar to build the search index first."
    )
    stats = {"total_chunks": 0, "unique_sources": 0, "source_names": []}


# --- Sidebar: Ingestion Controls ---
with st.sidebar:
    st.title("🔍 Semantic Search")
    st.metric("Documents Indexed", stats["unique_sources"])

    chunk_size = st.number_input(
        "Chunk size (characters)", min_value=50, value=DEFAULT_CHUNK_SIZE, step=50
    )
    overlap = st.number_input(
        "Overlap (characters)", min_value=0, value=DEFAULT_OVERLAP, step=10
    )

    if st.button("🔄 Re-index Documents"):
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            try:
                with st.spinner("Indexing documents..."):
                    indexed = ingest(chunk_size, overlap)
            except Exception as e:
                indexed = None
                st.error(f"Re-indexing failed: {e}")
            if indexed:
                st.toast(f"Re-indexed {indexed} chunks!", icon="✅")
                st.rerun()
            elif indexed == 0:
                st.error("No documents found in docs/ to index.")

    source_filter = st.multiselect("Filter by source", list(stats["source_names"]))

    # Number of results options

    n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)

    # Distance Threshold options
    use_threshold = st.checkbox("Use Distance Threshold")

    threshold = st.slider(
        "Max Distance",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        disabled=not use_threshold,
    )

    if use_threshold:
        distance_threshold = threshold
    else:
        distance_threshold = None

# --- Main Search Interface ---
st.title("🔍 Semantic Search Engine")
st.caption("Search by meaning, not just keywords.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Chunks", stats["total_chunks"])
with col2:
    st.metric("Total Documents", stats["unique_sources"])
results_metric = col3.empty()

query = st.text_input(
    "Enter your search query", placeholder="e.g. What is Object Oriented Programming?"
).strip()

results = []
search_failed = False

if not query:
    st.info("Enter a question above to search.")
elif stats["total_chunks"] == 0:
    st.error("Use 'Re-Index Documents' button on sidebar before searching")
else:
    try:
        with st.spinner("Searching documents"):
            results = search(
                query,
                n_results=n_results,
                sources=source_filter,
                distance_threshold=distance_threshold,
            )
    except Exception as e:
        search_failed = True
        st.error(f"Search failed: {e}")

results_metric.metric("Total Results", len(results))

if query and not results and not search_failed:
    st.warning(
        "No results found. Try loosening the threshold or clearing the source filter."
    )
elif results:
    for i, r in enumerate(results, start=1):
        distance = r["distance"]
        if distance < 0.5:
            relevance = "🟢 High"
        elif distance < 0.7:
            relevance = "🟡 Medium"
        else:
            relevance = "🔴 Low"

        with st.expander(
            f"{relevance} | {i}. {r['source']} - score {r['score']:.2f}",
            expanded=(i == 1),
        ):
            st.write(r["text"])
            st.caption(f"Chunk {r['chunk_index']} - distance {r['distance']:.2f}")
