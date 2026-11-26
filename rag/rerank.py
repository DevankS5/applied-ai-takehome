"""Optional cross-encoder reranking (off by default: +ranking quality, ~1 s per query on CPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.config import get_settings
from rag.embed import CACHE_DIR


@lru_cache
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(get_settings().rerank_model, cache_dir=CACHE_DIR)


def rerank(question: str, rows: list[dict], k: int) -> list[dict]:
    scores = list(_model().rerank(question, [r["text"] for r in rows]))
    for r, s in zip(rows, scores):
        r["rerank_score"] = float(s)
    return [rows[i] for i in np.argsort(scores)[::-1][:k]]
