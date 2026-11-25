# Hugging Face Space (Docker SDK): builds the service straight from the public GitHub repo.
# Bump CACHE_BUST (or "Factory rebuild" in the Space settings) to pick up new commits.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ARG REPO=https://github.com/DevankS5/applied-ai-takehome
ARG REF=main
ARG CACHE_BUST=1
RUN git clone --depth 1 --branch ${REF} ${REPO} /app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    RAG_MODEL_CACHE=/app/.cache/fastembed HOME=/app RAG_RATE_LIMIT_PER_MIN=12
RUN pip install -r requirements.txt
RUN python scripts/fetch_corpus.py --verify && python -m rag.ingest data/corpus/raw \
    && useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
# GROQ_API_KEY comes from the Space's secrets, never from the repo.
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
