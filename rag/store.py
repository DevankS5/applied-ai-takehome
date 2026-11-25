"""LanceDB vector store: schema, idempotent upsert, filtered top-k search."""

from __future__ import annotations

import time

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

    def search(self, qvec: np.ndarray, k: int, filters: dict[str, str] | None = None) -> tuple[list[dict], float]:
        """Top-k by cosine similarity, with optional equality filters on metadata columns."""
        q = self.table.search(qvec.tolist()).distance_type("cosine").limit(k)
        if filters:
            allowed = {"doc_type", "doc_id", "source", "title"}
            bad = set(filters) - allowed
            if bad:
                raise ValueError(f"unsupported filter keys: {sorted(bad)}; allowed: {sorted(allowed)}")
            q = q.where(" AND ".join(f"{k_} = {_quote(str(v))}" for k_, v in filters.items()), prefilter=True)
        t0 = time.perf_counter()
        cols = ["chunk_id", "doc_id", "source", "doc_type", "title", "heading", "page", "chunk_index", "text", "_distance"]
        rows = q.select(cols).to_list()
        ms = (time.perf_counter() - t0) * 1000
        for r in rows:
            r["score"] = 1.0 - float(r.pop("_distance"))  # cosine similarity
        return rows, ms
