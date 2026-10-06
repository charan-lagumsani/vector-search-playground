"""Benchmark FAISS vs ChromaDB: recall@k vs brute force + latency percentiles."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from src.backends import BruteForceBackend, ChromaBackend, FaissBackend
from src.embed import Embedder, load_docs

K = 5
REPEATS = 30


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    idx = min(len(xs) - 1, int(p / 100 * len(xs)))
    return xs[idx]


def main() -> None:
    docs = load_docs()
    queries = [json.loads(l) for l in
               Path("data/queries.jsonl").read_text().splitlines()]
    embedder = Embedder().fit([d["text"] for d in docs])

    backends = [
        FaissBackend(embedder, docs),
        ChromaBackend(embedder, docs),
        BruteForceBackend(embedder, docs),
    ]
    truth = {b.name: b for b in backends}["brute"]

    results = {}
    for backend in backends:
        recalls, latencies = [], []
        for item in queries:
            expected = set(item["expected_ids"])
            t0 = time.perf_counter()
            got = backend.search(item["query"], k=K)
            latencies.append((time.perf_counter() - t0) * 1000)
            recalls.append(len(expected & {r["id"] for r in got}) / len(expected))
        # steady-state latency
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            backend.search(queries[0]["query"], k=K)
            latencies.append((time.perf_counter() - t0) * 1000)
        results[backend.name] = {
            "recall@5": round(sum(recalls) / len(recalls), 3),
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
        }

    Path("results.json").write_text(json.dumps(results, indent=2))
    print(f"{'engine':<10}{'recall@5':<10}{'p50':<10}{'p95'}")
    for name, r in results.items():
        print(f"{name:<10}{r['recall@5']:<10}{r['p50_ms']:<10}{r['p95_ms']}")


if __name__ == "__main__":
    main()
