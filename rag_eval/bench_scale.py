"""Latency and ANN recall at 100K and 1M vectors: LanceDB (flat, IVF-PQ) vs FAISS (flat, HNSW).

    python -m rag_eval.bench_scale --sizes 100000 1000000

The corpus has only ~430 chunks, so larger collections are synthesised from the real
bge-small embeddings: each synthetic vector is a real chunk vector plus Gaussian noise,
renormalised (interpolated between two real chunks; see synth). This keeps the clustered structure of real embeddings (pure random vectors
would make ANN look worse than it is); it is still a proxy, stated as such in the README.
Queries are the eval questions' real embeddings plus perturbed copies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import faiss
import lancedb
import numpy as np
import pyarrow as pa

from rag.store import Store
from rag_eval.metrics import pct

OUT = Path("results/p1")
DB = Path("data/bench/lancedb")
K = 10


def synth(base: np.ndarray, n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Points on segments between two random real embeddings, plus noise, renormalised.
    (A first version used one seed + noise; with ~430 seeds that made ~2,300 near-duplicates per
    seed at 1M, so 'exact top-10' was arbitrary among twins and ANN recall looked far worse than it is.)"""
    i, j = rng.integers(0, len(base), n), rng.integers(0, len(base), n)
    a = rng.random((n, 1), dtype=np.float32)
    v = a * base[i] + (1 - a) * base[j] + rng.normal(0, sigma, (n, base.shape[1])).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def timed(fn, queries: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    fn(queries[0])  # warm-up
    ids, ms = [], []
    for q in queries:
        t0 = time.perf_counter()
        ids.append(fn(q))
        ms.append((time.perf_counter() - t0) * 1000)
    return ids, ms


def recall(found: list[np.ndarray], truth: list[np.ndarray]) -> float:
    return float(np.mean([len(set(f[:K]) & set(t[:K])) / K for f, t in zip(found, truth)]))


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def bench(n: int, base: np.ndarray, queries: np.ndarray, rng: np.random.Generator) -> list[dict]:
    data = synth(base, n, 0.02, rng)
    out = []

    # ---- FAISS exact (inner product == cosine on unit vectors) ----
    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(data.shape[1])
    flat.add(data)
    build = time.perf_counter() - t0
    truth, ms = timed(lambda q: flat.search(q[None], K)[1][0], queries)
    out.append({"system": "FAISS IndexFlatIP (exact, in RAM)", "build_s": round(build, 1), "size_mb": round(data.nbytes / 1e6, 1),
                "p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2), "recall@10": 1.0})

    # ---- FAISS HNSW ----
    t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(data.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(data)
    build = time.perf_counter() - t0
    for ef in (64, 128, 256):
        hnsw.hnsw.efSearch = ef
        found, ms = timed(lambda q: hnsw.search(q[None], K)[1][0], queries)
        out.append({"system": f"FAISS HNSW (M=32, efSearch={ef}, in RAM)", "build_s": round(build, 1),
                    "size_mb": round((data.nbytes + n * 32 * 2 * 4) / 1e6, 1), "p50_ms": round(pct(ms, 50), 2),
                    "p95_ms": round(pct(ms, 95), 2), "recall@10": round(recall(found, truth), 4)})

    # ---- LanceDB: same data on disk ----
    shutil.rmtree(DB, ignore_errors=True)
    db = lancedb.connect(str(DB))
    tbl = pa.table({"id": pa.array(np.arange(n), pa.int64()),
                    "vector": pa.FixedSizeListArray.from_arrays(pa.array(data.reshape(-1)), data.shape[1])})
    t0 = time.perf_counter()
    t = db.create_table("bench", tbl)
    write = time.perf_counter() - t0

    def lance(q, **kw):
        s = t.search(q.tolist()).distance_type("cosine").limit(K).select(["id"])
        for k, v in kw.items():
            s = getattr(s, k)(v)
        return np.array([r["id"] for r in s.to_list()])

    found, ms = timed(lambda q: lance(q), queries)
    out.append({"system": "LanceDB flat scan (exact, on disk)", "build_s": round(write, 1), "size_mb": round(dir_mb(DB), 1),
                "p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2), "recall@10": round(recall(found, truth), 4)})

    t0 = time.perf_counter()
    parts = int(np.sqrt(n))
    t.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=48)
    build = time.perf_counter() - t0
    for nprobes, refine in [(20, 10), (50, 20), (100, 50)]:
        found, ms = timed(lambda q: lance(q, nprobes=nprobes, refine_factor=refine), queries)
        out.append({"system": f"LanceDB IVF-PQ ({parts} parts, 48 subvec, nprobes={nprobes}, refine={refine})", "build_s": round(build, 1),
                    "size_mb": round(dir_mb(DB), 1), "p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2),
                    "recall@10": round(recall(found, truth), 4)})
    for r in out:
        r["vectors"] = n
    del data, flat, hnsw
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--queries", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    rows = Store().table.search().select(["vector"]).limit(None).to_list()
    base = np.array([r["vector"] for r in rows], dtype=np.float32)
    from rag.embed import embed_query
    from rag_eval.run_retrieval import load_questions

    real_q = np.stack([embed_query(q["question"]) for q in load_questions()])
    queries = synth(real_q, a.queries, 0.01, rng)
    queries[: len(real_q)] = real_q
    results = []
    for n in a.sizes:
        print(f"benchmarking {n:,} vectors ...", flush=True)
        results += bench(n, base, queries, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale_bench.json").write_text(json.dumps({"k": K, "queries": len(queries), "dim": base.shape[1], "results": results}, indent=2))
    md = ["# Problem 1: scale benchmark (single query, k = 10, laptop CPU)", "",
          f"{len(queries)} queries; synthetic vectors = interpolations between two real bge-small chunk embeddings + noise (see module docstring). "
          "Recall@10 is measured against FAISS exact search on the same vectors.", "",
          "| Vectors | System | Build / write (s) | Size (MB) | p50 ms | p95 ms | Recall@10 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['vectors']:,} | {r['system']} | {r['build_s']} | {r['size_mb']} | {r['p50_ms']} | {r['p95_ms']} | {r['recall@10']} |" for r in results]
    (OUT / "scale_bench.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
