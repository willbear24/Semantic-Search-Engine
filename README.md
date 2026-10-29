#Semantic Search Engine

## Overview

A semantic search tool over a small document collection. Documents in `docs/`
are split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored
in a persistent ChromaDB collection. A Streamlit app lets you type a question
in plain English and see the most relevant chunks, ranked by cosine distance.
An evaluation script measures search quality, and a chunking experiment
(below) compares different chunk sizes.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Index the document corpus:

   ```bash
   python ingest.py
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

Optional: re-index with custom chunking from the command line (the app's
sidebar offers the same controls):

```bash
python ingest.py --chunk-size 200 --overlap 40
```

Evaluate search quality against the labelled queries in `evaluate.py`:

```bash
python evaluate.py --n-results 5 --threshold 0.7
```

## Architecture

```
docs/*.txt ──► ingest.py ──► chroma_data/ (ChromaDB, cosine) ──► search.py ──► app.py
               load → chunk     chunk text + embedding +         query →        Streamlit UI
               → embed → upsert metadata                         ranked results       ▲
                                                                       └── evaluate.py (precision / recall)
```

| File               | Purpose                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| `app.py`           | Streamlit interface: search box, sidebar controls, metrics, results           |
| `ingest.py`        | Loads documents, chunks text, embeds chunks, stores them in ChromaDB          |
| `search.py`        | Embeds a query, queries ChromaDB, returns scored results and collection stats |
| `evaluate.py`      | Labelled test queries and source-level precision/recall                       |
| `docs/`            | Document corpus (8 `.txt` files, about 2,900 words)                           |
| `chroma_data/`     | Persistent ChromaDB storage (created by `ingest.py`)                          |
| `requirements.txt` | Python dependencies                                                           |

### Design notes

- **Chunking:** character-based sliding window (default 500 characters, 100
  overlap) that prefers to break at sentence boundaries. Both values are
  configurable.
- **Metadata:** every chunk stores its `source` filename, `chunk_index` and the
  `chunk_size` used.
- **Distance metric:** the collection uses cosine distance, so
  `score = 1 - distance` is meaningful. The metric is fixed when the collection
  is created; changing it requires deleting `chroma_data/` and re-indexing.
- **Re-indexing:** each file's old chunks are deleted before its new chunks are
  upserted, so re-indexing with a different chunk size replaces stale chunks.
- **Relevance indicator:** results are labelled 🟢 High (distance < 0.5),
  🟡 Medium (< 0.7) or 🔴 Low (0.7 or more).
- **Edge cases:** empty or whitespace-only queries, an empty or missing index,
  results filtered out by the threshold, and invalid chunk/overlap values are
  all handled with messages instead of errors.

## Chunking Experiment Results

**Chunk sizes tested:** 200, 500 and 1000 characters (overlap about 20% of the
chunk size: 40, 100 and 200). All runs used `n_results=5`, no distance
threshold and the same queries. Metrics are source-level: a result counts as
correct if its file is one of the labelled relevant files.

**Test queries** (labelled with the files that should be retrieved):

1. What is the definition and use case of a Model in FastAPI? → `fastapi.txt`
2. How do I make sure only authorized users can access my API? → `fastapi.txt`, `rest-apis.txt`
3. How can I avoid recomputing expensive results on repeated calls? → `python-advanced.txt`, `streamlit.txt`
4. How do I stop a language model from making up facts? → `embeddings-and-vectors.txt`, `llms-and-ai.txt`
5. What is the difference between an INNER JOIN and a LEFT JOIN? → `sql-databases.txt`
6. How do I handle errors and make sure resources get cleaned up? → `python-advanced.txt`, `python-fundamentals.txt`

**Results (precision / recall):**

| Query                 | 200 chars       | 500 chars       | 1000 chars      |
| --------------------- | --------------- | --------------- | --------------- |
| 1. FastAPI Model      | 1.00 / 1.00     | 1.00 / 1.00     | 0.33 / 1.00     |
| 2. Authorization      | 1.00 / 0.50     | 1.00 / 1.00     | 0.67 / 1.00     |
| 3. Caching            | 0.33 / 0.50     | 0.25 / 0.50     | 0.33 / 0.50     |
| 4. Hallucination      | 1.00 / 1.00     | 1.00 / 1.00     | 1.00 / 1.00     |
| 5. INNER vs LEFT JOIN | 0.50 / 1.00     | 0.33 / 1.00     | 0.25 / 1.00     |
| 6. Errors and cleanup | 0.50 / 1.00     | 0.50 / 1.00     | 0.40 / 1.00     |
| **Average**           | **0.72 / 0.83** | **0.68 / 0.92** | **0.50 / 0.92** |

**Findings:**

- **500 characters worked best overall.** It matched the best recall (0.92) and
  had precision close to the top (0.68).
- **200 characters** gave the highest precision (0.72) but the lowest recall
  (0.83). Small chunks are very focused, so specific queries such as the JOIN
  question did well. But several chunks from one file can fill the top 5 and
  crowd out other relevant files, as with the authorization query, which lost
  `rest-apis.txt`.
- **1000 characters** had the lowest precision (0.50). Large chunks cover
  several topics, so their embeddings are less specific and unrelated files
  rank higher. The FastAPI Model query dropped from 1.00 to 0.33 precision.
- **Per query:** the 500-character setting won or tied on queries 1, 2, 4 and 6. The 200-character setting won on query 5 (specific lookups), and 1000 tied
  on query 3.
- **The caching query struggled at every size** (recall stayed at 0.50). It
  never retrieved `streamlit.txt`. This looks like a vocabulary mismatch
  between the query and the document, not a chunking problem.
- **Caveats:** there are only 6 queries and 8 short documents, so one query
  moves an average by about 0.17. Treat these results as indicative, not
  conclusive.
