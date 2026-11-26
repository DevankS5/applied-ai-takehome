"""Hugging Face Space entry point.

Free Spaces must use the Gradio SDK (Docker Spaces are paid), so this file does what the Dockerfile does:
clone the public repo, verify the pinned corpus, build the LanceDB index, then serve the same FastAPI app
(a Gradio UI on /, JSON API on /query and /stats).
GROQ_API_KEY comes from the Space's secrets, never from the repo.
"""

import os
import subprocess
import sys

REPO = "https://github.com/DevankS5/applied-ai-takehome"
REF = os.environ.get("REPO_REF", "main")
SRC = os.path.abspath("src")

if not os.path.isdir(SRC):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, REPO, SRC], check=True)
os.chdir(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("RAG_MODEL_CACHE", os.path.join(SRC, ".cache", "fastembed"))
os.environ.setdefault("RAG_RATE_LIMIT_PER_MIN", "12")
subprocess.run([sys.executable, "scripts/fetch_corpus.py", "--verify"], check=True)
subprocess.run([sys.executable, "-m", "rag.ingest", "data/corpus/raw"], check=True)

import gradio as gr  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from rag.api import Filters, QueryIn, app, query  # noqa: E402

try:  # ZeroGPU (the free hardware) expects one GPU-decorated function; the RAG path is CPU-only and never calls it.
    import spaces

    @spaces.GPU
    def _unused() -> None:
        return None
except ImportError:
    pass


def ask(question: str, doc_type: str, request: gr.Request):
    try:
        body = QueryIn(question=question, k=5, filters=Filters(doc_type=doc_type) if doc_type != "any" else None)
        out = query(body, request)  # same rate limit and daily LLM budget as POST /query
    except HTTPException as e:
        return f"Error {e.status_code}: {e.detail}", {}
    except ValueError as e:
        return f"Invalid request: {e}", {}
    keep = ["abstained", "citations", "top_score", "chunks_retrieved", "chunks_cited", "latency_ms", "tokens", "model"]
    return out["answer"], {k: out[k] for k in keep}


with gr.Blocks(title="Cost-efficient RAG on LanceDB") as panel:
    gr.Markdown("## Cost-efficient RAG on LanceDB\nCited answers over 17 documents (RAG, vector search, IR metrics, LLM-as-judge). "
                "JSON API: `POST /query` with `{question, k, filters}`; `GET /stats`. Source: "
                "[github.com/DevankS5/applied-ai-takehome](https://github.com/DevankS5/applied-ai-takehome)")
    q = gr.Textbox(label="Question", value="What is the default hnsw.ef_search in pgvector?")
    t = gr.Dropdown(["any", "pdf", "html", "md"], value="any", label="doc_type filter")
    a = gr.Textbox(label="Answer")
    meta = gr.JSON(label="Latency, chunks, tokens")
    gr.Button("Ask").click(ask, [q, t], [a, meta])

# ZeroGPU Spaces must start through Blocks.launch(), so Gradio owns port 7860 and the panel is the UI on /.
# The service's JSON routes (/query, /stats, /health) are then attached to Gradio's own FastAPI app.
panel.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), prevent_thread_lock=True)
server = getattr(panel, "server_app", None) or panel.app
for route in app.router.routes:
    if getattr(route, "path", "") in ("/query", "/stats", "/health"):
        server.router.routes.insert(0, route)
panel.block_thread()
