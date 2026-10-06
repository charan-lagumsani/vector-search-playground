# vector-search-playground — FAISS vs ChromaDB Semantic Search

A hands-on comparison of two vector search engines on the same corpus:
**FAISS** (raw speed, manual index management) vs **ChromaDB** (batteries-included
vector database with metadata filtering). Includes a reproducible benchmark
(recall@k vs brute force, p50/p95 latency) and a **Streamlit UI** for
side-by-side interactive search.

## Quickstart

```bash
pip install -r requirements.txt

# Run the benchmark
python benchmark.py
# → prints recall/latency table, writes results.json

# Launch the interactive playground
streamlit run app.py
```

## How it works

- **Embeddings** (`src/embed.py`) — TF-IDF over the corpus (zero downloads,
  deterministic). The `Embedder` class mirrors the interface used in `finrag`;
  swap in `sentence-transformers` for dense embeddings.
- **FAISS path** — `IndexFlatIP` over L2-normalized vectors (exact cosine
  search), metadata kept in a parallel list.
- **ChromaDB path** — `PersistentClient` collection with the same vectors plus
  per-document metadata (`category`), demonstrating metadata-filtered search.
- **Benchmark** (`benchmark.py`) — for each test query: brute-force cosine
  ranking as ground truth → recall@k for each engine, plus p50/p95 query
  latency over repeated runs.
- **UI** (`app.py`) — query box, top-k slider, category filter; shows FAISS
  and ChromaDB results side by side with per-engine latency.

Sample benchmark output (36 docs, TF-IDF, 8 queries):

```
engine    recall@5  p50     p95
faiss     1.000     0.7ms   1.3ms
chromadb  1.000     1.6ms   3.4ms
brute     1.000     0.4ms   0.8ms
```

(On a tiny corpus everything is fast — the point is the harness, not the
absolute numbers. Scale `data/docs.jsonl` to feel the difference.)

## License

MIT — see [LICENSE](LICENSE).
