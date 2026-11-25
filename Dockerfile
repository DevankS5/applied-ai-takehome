# RAG service image (used by the Hugging Face Space; runs anywhere Docker does).
#   docker build -t rag . && docker run -p 7860:7860 -e GROQ_API_KEY=... rag
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    RAG_MODEL_CACHE=/app/.cache/fastembed HOME=/app RAG_RATE_LIMIT_PER_MIN=12
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Build the index into the image: fetch the pinned corpus, chunk, embed (CPU), write LanceDB + BM25 index.
RUN python scripts/fetch_corpus.py && python -m rag.ingest data/corpus/raw \
    && useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 7860
# No --proxy-headers: X-Forwarded-For is client-controlled; without a trusted x-real-ip the limiter
# falls back to one shared bucket, which is the safe failure for a public demo key.
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
