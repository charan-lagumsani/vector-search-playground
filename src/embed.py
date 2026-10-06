"""Shared TF-IDF embedder and corpus loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_docs(path: Path = Path("data/docs.jsonl")) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class Embedder:
    """TF-IDF embedder with L2-normalized outputs (cosine via inner product)."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=10000
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "Embedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    @property
    def dim(self) -> int:
        return len(self._vectorizer.vocabulary_)
