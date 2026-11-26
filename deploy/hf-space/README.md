---
title: Cost-efficient RAG on LanceDB
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Cited answers over 17 docs; LanceDB + hybrid search
---

Live demo for Problem 1 of an applied AI take-home: grounded, cited answers over a 17-document corpus
(RAG, vector search, IR metrics, LLM-as-judge), served from an embedded LanceDB table with hybrid
(dense + BM25) retrieval and local bge-small embeddings. `app.py` clones the public repo, verifies the
pinned corpus and builds the index at startup, then serves the same FastAPI app as the repo.

Source, evaluation and write-up: https://github.com/DevankS5/applied-ai-takehome
