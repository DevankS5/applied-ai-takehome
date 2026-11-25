"""Runtime configuration. Everything comes from the environment (or .env); no secrets in code."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    db_path: str = "data/lancedb"
    table: str = "chunks"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # bge-*-v1.5 recommends this instruction on short queries (not on passages).
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    chunk_size: int = 400  # tokens (bge tokenizer)
    chunk_overlap: int = 60  # tokens
    top_k: int = 5
    search_mode: str = "hybrid"  # vector | bm25 | hybrid (dense + BM25 fused with RRF)
    rerank: bool = False  # cross-encoder over 4k hybrid candidates: better ranking, ~1 s CPU
    rerank_model: str = "jinaai/jina-reranker-v1-tiny-en"
    max_top_k: int = 20
    # Cosine-similarity floor: below it we abstain without calling the LLM.
    # Calibrated on the eval set (rag_eval/run_retrieval.py: lowest answerable top-1 minus 0.05; results/p1/retrieval.md).
    min_score: float = 0.58
    gen_provider: str = "groq"
    gen_model: str = "llama-3.3-70b-versatile"
    gen_max_tokens: int = 400
    llm_cache_dir: str | None = None  # eval harnesses set this; serverless disks are read-only
    log_level: str = "INFO"
    # Public demo protection
    rate_limit_per_min: int = 6
    daily_llm_cap: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
