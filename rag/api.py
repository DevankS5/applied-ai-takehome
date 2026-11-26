"""HTTP API for the RAG service.

    uvicorn rag.api:app --port 8000
    curl -s localhost:8000/query -H 'content-type: application/json' \
         -d '{"question": "What is the default hnsw.ef_search in pgvector?", "k": 5}'
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.llm import LLMError
from rag import service
from rag.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Cost-efficient RAG on LanceDB", version="1.0",
              description="Grounded, cited answers over a 17-document corpus. Problem 1 of the take-home.")


class Filters(BaseModel):
    doc_type: Literal["pdf", "html", "md"] | None = None
    doc_id: str | None = Field(None, max_length=64)


class QueryIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    filters: Filters | None = None


# ponytail: in-memory limits are per serverless instance, so they are best-effort; a shared
# counter (Redis/Upstash) would make them global if the demo ever gets real traffic.
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": time.strftime("%Y-%m-%d"), "llm_calls": 0}
_lock = threading.Lock()
MAX_TRACKED_IPS = 10_000


def client_ip(request: Request) -> str:
    """Socket peer by default. x-real-ip is honoured only when RAG_TRUST_X_REAL_IP=1, i.e. when the
    deployment sits behind a proxy known to overwrite it (Vercel's edge does). Never X-Forwarded-For,
    whose left-most entry the client controls."""
    if get_settings().trust_x_real_ip and request.headers.get("x-real-ip"):
        return request.headers["x-real-ip"]
    return request.client.host if request.client else "unknown"


def _check_rate(ip: str) -> None:
    s = get_settings()
    now = time.time()
    with _lock:
        if len(_hits) > MAX_TRACKED_IPS:  # bound memory: forget idle clients
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 60]:
                del _hits[k]
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= s.rate_limit_per_min:
            raise HTTPException(429, f"rate limit: {s.rate_limit_per_min} questions per minute")
        q.append(now)


def _reserve_llm_call() -> bool:
    """Atomically take one slot of today's LLM budget (False when it is spent)."""
    with _lock:
        today = time.strftime("%Y-%m-%d")
        if _day["date"] != today:
            _day.update(date=today, llm_calls=0)
        if _day["llm_calls"] >= get_settings().daily_llm_cap:
            return False
        _day["llm_calls"] += 1
        return True


def _refund_llm_call() -> None:
    with _lock:
        _day["llm_calls"] = max(0, _day["llm_calls"] - 1)


@app.post("/query")
def query(body: QueryIn, request: Request) -> dict:
    _check_rate(client_ip(request))
    filters = {k: v for k, v in (body.filters.model_dump() if body.filters else {}).items() if v}
    reserved = _reserve_llm_call()
    try:
        out = service.answer(body.question, body.k, filters, generate_answer=reserved)
    except ValueError as e:
        if reserved:
            _refund_llm_call()
        raise HTTPException(422, str(e)) from e
    except LLMError as e:
        raise HTTPException(503, "generator unavailable, try again shortly") from e
    if reserved and not out["model"]["generator"]:
        _refund_llm_call()  # guard abstained before the LLM: the slot was not used
    over_cap = not reserved
    if over_cap:
        out["note"] = "daily LLM budget for this public demo is used up; showing retrieval results only"
    return out


@app.get("/stats")
def stats() -> dict:
    s = get_settings()
    t = service.store().table
    docs = t.search().select(["doc_id", "doc_type", "title"]).limit(None).to_list()
    by_doc: dict[str, dict] = {}
    for d in docs:
        by_doc.setdefault(d["doc_id"], {"doc_id": d["doc_id"], "doc_type": d["doc_type"], "title": d["title"], "chunks": 0})
        by_doc[d["doc_id"]]["chunks"] += 1
    return {
        "vectors": t.count_rows(), "embed_model": s.embed_model, "embed_dim": s.embed_dim,
        "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "min_score": s.min_score,
        "generator": f"{s.gen_provider}/{s.gen_model}", "documents": sorted(by_doc.values(), key=lambda x: x["doc_id"]),
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


if (ROOT / "web").exists():
    app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
