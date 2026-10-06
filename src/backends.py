"""Search backends: FAISS, ChromaDB, and brute-force ground truth."""
from __future__ import annotations

from pathlib import Path

import chromadb
import faiss
import numpy as np

from .embed import Embedder, load_docs


class FaissBackend:
    name = "faiss"

    def __init__(self, embedder: Embedder, docs: list[dict]):
        self.docs = docs
        self.embedder = embedder
        vecs = embedder.encode([d["text"] for d in docs]).astype(np.float32)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[dict]:
        q = self.embedder.encode([query]).astype(np.float32)
        # over-fetch when filtering, then filter down
        fetch = k if category is None else len(self.docs)
        scores, idxs = self.index.search(q, fetch)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc = self.docs[idx]
            if category and doc["category"] != category:
                continue
            out.append({"id": doc["id"], "text": doc["text"],
                        "category": doc["category"], "score": float(score)})
            if len(out) == k:
                break
        return out


class ChromaBackend:
    name = "chromadb"

    def __init__(self, embedder: Embedder, docs: list[dict],
                 persist_dir: Path = Path("chroma_db")):
        import shutil

        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        self.docs = docs
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.create_collection("docs")
        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"category": d["category"]} for d in docs],
            embeddings=embedder.encode(
                [d["text"] for d in docs]).astype(float).tolist(),
        )
        self.embedder = embedder

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[dict]:
        q = self.embedder.encode([query]).astype(float).tolist()[0]
        where = {"category": category} if category else None
        res = self.collection.query(
            query_embeddings=[q], n_results=k, where=where,
            include=["documents", "metadatas", "distances"])
        out = []
        for doc_id, text, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0]):
            out.append({"id": doc_id, "text": text,
                        "category": meta["category"],
                        "score": round(1 - dist, 4)})
        return out


class BruteForceBackend:
    name = "brute"

    def __init__(self, embedder: Embedder, docs: list[dict]):
        self.docs = docs
        self.embedder = embedder
        self.vecs = embedder.encode([d["text"] for d in docs])

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[dict]:
        q = self.embedder.encode([query])[0]
        scores = self.vecs @ q
        order = np.argsort(-scores)
        out = []
        for idx in order:
            doc = self.docs[int(idx)]
            if category and doc["category"] != category:
                continue
            out.append({"id": doc["id"], "text": doc["text"],
                        "category": doc["category"],
                        "score": float(scores[idx])})
            if len(out) == k:
                break
        return out
