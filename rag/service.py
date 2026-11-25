"""Query pipeline: embed -> filtered top-k -> no-context guard -> grounded answer -> log."""

from __future__ import annotations

import json
import logging
import sys
import time

from rag.config import get_settings
from rag.embed import embed_query
from rag.generate import NO_ANSWER_TEXT, generate
from rag.store import Store

log = logging.getLogger("rag")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(get_settings().log_level)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def answer(question: str, k: int | None = None, filters: dict | None = None, generate_answer: bool = True) -> dict:
    s = get_settings()
    k = k or s.top_k
    if not 1 <= k <= s.max_top_k:
        raise ValueError(f"k must be between 1 and {s.max_top_k}")
    t0 = time.perf_counter()
    qvec = embed_query(question)
    t_embed = (time.perf_counter() - t0) * 1000
    rows, t_retrieve = store().search(qvec, k, filters)

    top = rows[0]["score"] if rows else 0.0
    gen = None
    if not rows or top < s.min_score:
        reason = "no_chunks" if not rows else "below_threshold"
    elif generate_answer:
        gen = generate(question, rows)
        reason = "model_abstained" if gen.abstained else None
    else:
        reason = "retrieval_only"

    sources = [
        {"n": i, "chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "title": r["title"], "doc_type": r["doc_type"],
         "page": r["page"], "heading": r["heading"], "score": round(r["score"], 4), "text": r["text"],
         "cited": bool(gen and i in gen.cited)}
        for i, r in enumerate(rows, start=1)
    ]
    out = {
        "question": question,
        "answer": gen.text if gen else NO_ANSWER_TEXT,
        "abstained": gen is None or gen.abstained,
        "abstain_reason": reason,
        "citations": gen.cited if gen else [],
        "invalid_citations": gen.invalid_citations if gen else [],
        "sources": sources,
        "k": k,
        "filters": filters or {},
        "top_score": round(top, 4),
        "latency_ms": {
            "embed": round(t_embed, 1),
            "retrieve": round(t_retrieve, 1),
            "generate": round(gen.latency_ms, 1) if gen else 0.0,
            "total": round((time.perf_counter() - t0) * 1000, 1),
        },
        "chunks_retrieved": len(rows),
        "chunks_cited": len(gen.cited) if gen else 0,
        "tokens": {"prompt": gen.prompt_tokens if gen else 0, "completion": gen.completion_tokens if gen else 0},
        "model": {"embed": s.embed_model, "embed_dim": s.embed_dim, "generator": gen.model if gen else None},
    }
    log.info(json.dumps({
        "event": "query", "q_chars": len(question), "k": k, "filters": filters or {}, "top_score": out["top_score"],
        "abstained": out["abstained"], "abstain_reason": reason, "chunks_retrieved": out["chunks_retrieved"],
        "chunks_cited": out["chunks_cited"], "latency_ms": out["latency_ms"], "tokens": out["tokens"],
    }))
    return out
