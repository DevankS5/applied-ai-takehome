"""LanceDB vector store: schema, idempotent upsert, filtered top-k search."""

from __future__ import annotations

import time
import warnings

import lancedb
import numpy as np
import pyarrow as pa

from rag.config import get_settings


def schema(dim: int) -> pa.Schema:
    s = get_settings()
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("doc_type", pa.string()),  # metadata filter: pdf | html | md
            pa.field("title", pa.string()),
            pa.field("heading", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("n_tokens", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("ingested_at", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ],
        # Record how the vectors were produced, next to the vectors.
        metadata={"embed_model": s.embed_model, "embed_dim": str(dim)},
    )


COLS = ["chunk_id", "doc_id", "source", "doc_type", "title", "heading", "page", "chunk_index", "text"]
FILTER_KEYS = {"doc_type", "doc_id", "source", "title"}


def _quote(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


class Store:
    def __init__(self, db_path: str | None = None, table: str | None = None):
        s = get_settings()
        self.db = lancedb.connect(db_path or s.db_path)
        self.name = table or s.table
        self.dim = s.embed_dim
        self.table = self.db.create_table(self.name, schema=schema(self.dim), exist_ok=True)

    def count(self) -> int:
        return self.table.count_rows()

    def chunk_ids_for(self, doc_id: str) -> set[str]:
        rows = self.table.search().where(f"doc_id = {_quote(doc_id)}").select(["chunk_id"]).limit(None).to_list()
        return {r["chunk_id"] for r in rows}

    def upsert(self, rows: list[dict]) -> None:
        """merge_insert keyed on chunk_id: re-running ingestion can never duplicate a vector."""
        if rows:
            (
                self.table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(pa.Table.from_pylist(rows, schema=schema(self.dim)))
            )

    def delete_except(self, doc_id: str, keep: set[str]) -> int:
        """Drop a document's chunks that are no longer produced (doc edited or re-chunked)."""
        stale = self.chunk_ids_for(doc_id) - keep
        if stale:
            ids = ", ".join(_quote(c) for c in stale)
            self.table.delete(f"doc_id = {_quote(doc_id)} AND chunk_id IN ({ids})")
        return len(stale)

    def delete_docs_not_in(self, doc_ids: set[str]) -> int:
        existing = {r["doc_id"] for r in self.table.search().select(["doc_id"]).limit(None).to_list()}
        gone = existing - doc_ids
        before = self.count()
        for d in gone:
            self.table.delete(f"doc_id = {_quote(d)}")
        return before - self.count()

    def build_text_index(self) -> None:
        """BM25 full-text index on chunk text; rebuilt after each ingest so new rows are searchable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.table.create_fts_index("text", replace=True, stem=True, remove_stop_words=True)

    def _where(self, filters: dict[str, str] | None) -> str | None:
        if not filters:
            return None
        bad = set(filters) - FILTER_KEYS
        if bad:
            raise ValueError(f"unsupported filter keys: {sorted(bad)}; allowed: {sorted(FILTER_KEYS)}")
        return " AND ".join(f"{k} = {_quote(str(v))}" for k, v in filters.items())

    def _vector(self, qvec: np.ndarray, n: int, where: str | None) -> list[dict]:
        q = self.table.search(qvec.tolist()).distance_type("cosine").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        return q.select(COLS + ["_distance"]).to_list()

    def _bm25(self, text: str, n: int, where: str | None) -> list[dict]:
        q = self.table.search(text, query_type="fts").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        try:
            return q.select(COLS + ["vector"]).to_list()
        except Exception:  # no FTS index yet (fresh table): behave like an empty keyword leg
            return []

    def search(
        self, qvec: np.ndarray, k: int, filters: dict[str, str] | None = None,
        text: str | None = None, mode: str = "hybrid", rrf_k: int = 60,
    ) -> tuple[list[dict], float]:
        """Top-k chunks. Every row carries `score` = cosine similarity to the query (used by the
        no-context guard) and, in hybrid mode, `rrf` = reciprocal-rank-fusion score (used to rank).

        vector: dense cosine top-k.  bm25: keyword top-k.  hybrid: both legs fetch 2k candidates,
        fused with RRF (sum of 1 / (rrf_k + rank)), which needs no score calibration between legs."""
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("mode must be vector, bm25 or hybrid")
        where = self._where(filters)
        t0 = time.perf_counter()
        if mode == "vector" or not text:
            rows = self._vector(qvec, k, where)
            for r in rows:
                r["score"] = 1.0 - float(r.pop("_distance"))
        else:
            n = k if mode == "bm25" else 2 * k
            kw = self._bm25(text, n, where)
            for r in kw:  # vectors are unit length, so cosine is a dot product
                r["score"] = float(np.dot(qvec, np.asarray(r.pop("vector"), dtype=np.float32)))
                r.pop("_score", None)
            if mode == "bm25":
                rows = kw
            else:
                dense = self._vector(qvec, n, where)
                for r in dense:
                    r["score"] = 1.0 - float(r.pop("_distance"))
                fused: dict[str, dict] = {}
                for leg in (dense, kw):
                    for rank, r in enumerate(leg, start=1):
                        f = fused.setdefault(r["chunk_id"], {**r, "rrf": 0.0})
                        f["rrf"] += 1.0 / (rrf_k + rank)
                rows = sorted(fused.values(), key=lambda r: r["rrf"], reverse=True)[:k]
        return rows, (time.perf_counter() - t0) * 1000
