"""Local ONNX embeddings (fastembed). No API calls, no GPU, $0 per query."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

from rag.config import get_settings

CACHE_DIR = os.environ.get("RAG_MODEL_CACHE", ".cache/fastembed")


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(get_settings().embed_model, cache_dir=CACHE_DIR, threads=os.cpu_count())


@lru_cache
def tokenizer() -> Tokenizer:
    """The embedding model's own tokenizer, so chunk sizes are in the units the model sees."""
    tok = _model().model.tokenizer  # fastembed keeps the HF tokenizer on the ONNX wrapper
    tok = Tokenizer.from_str(tok.to_str())  # private copy: counting must not truncate at 512
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(texts: list[str]) -> list[int]:
    return [len(e.ids) for e in tokenizer().encode_batch(texts, add_special_tokens=False)]


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _normalise(np.array(list(_model().embed(texts, batch_size=batch_size)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    q = get_settings().query_instruction + text
    return _normalise(np.array(list(_model().embed([q])), dtype=np.float32))[0]
