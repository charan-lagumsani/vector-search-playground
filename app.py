"""Streamlit playground: FAISS vs ChromaDB side-by-side semantic search."""
from __future__ import annotations

import time
from functools import lru_cache

import streamlit as st

from src.backends import ChromaBackend, FaissBackend
from src.embed import Embedder, load_docs


@lru_cache(maxsize=1)
def get_backends():
    docs = load_docs()
    embedder = Embedder().fit([d["text"] for d in docs])
    return docs, FaissBackend(embedder, docs), ChromaBackend(embedder, docs)


st.set_page_config(page_title="Vector Search Playground", layout="wide")
st.title("Vector Search Playground")
st.caption("Same corpus, same embeddings — FAISS vs ChromaDB, side by side.")

docs, faiss_backend, chroma_backend = get_backends()

query = st.text_input("Search query", "how do I get a refund")
k = st.slider("Top-k", 1, 10, 5)
categories = ["all"] + sorted({d["category"] for d in docs})
category = st.selectbox("Category filter", categories)
cat = None if category == "all" else category

if query:
    col1, col2 = st.columns(2)
    for col, backend in ((col1, faiss_backend), (col2, chroma_backend)):
        t0 = time.perf_counter()
        results = backend.search(query, k=k, category=cat)
        elapsed = (time.perf_counter() - t0) * 1000
        with col:
            st.subheader(f"{backend.name} — {elapsed:.1f} ms")
            for r in results:
                st.markdown(f"**{r['id']}** `[{r['category']}]` "
                            f"(score {r['score']:.3f})")
                st.write(r["text"][:160] + "…")
