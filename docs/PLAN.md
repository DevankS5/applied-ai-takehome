# Build plan

Applied AI / ML take-home: Problem 1 (cost-efficient RAG) and Problem 2 (LLM-as-judge with bias handling).
One repo, two packages, one deployment, one submission document.

## Repo layout

```
rag/            Problem 1: ingestion, chunking, embeddings, LanceDB store, retrieval, generation, FastAPI app
rag_eval/       Problem 1: fixed question set, retrieval / answer / latency harness, scale benchmark, cost model
judge/          Problem 2: suite loader, rubric, prompts, providers, robust verdict parser, aggregation, A/B, bias probes
judge_eval/     Problem 2: validation runs (human-label agreement, test-retest, adversarial probes), reports
common/         LLM provider clients (Gemini, Groq), disk cache, JSONL audit log, token/cost accounting
results/        Every number quoted in the README and the submission doc, as JSON + Markdown + PNG charts
docs/           Diagrams, screenshots, this plan
```

## Problem 1: decisions

| Topic | Decision | Why |
|---|---|---|
| Vector store | **LanceDB** (embedded, Apache-2.0) | Storage and compute are decoupled: the index is files on local disk or S3, so a large, lightly queried index costs object-storage rates instead of always-on pods. SQL `where` filters, IVF-PQ and HNSW indexes, merge-insert for upserts. |
| Bonus store | **FAISS** benchmarked on the same vectors | In-memory baseline for latency and ANN recall at 100K and 1M vectors. |
| Corpus | ~15 public docs on RAG, vector search and LLM evaluation, mixed PDF (arXiv), HTML (Wikipedia), MD (project READMEs) | Mixed formats exercise every loader; the topic makes the demo self-explanatory. Fetched by a pinned script, not committed (licences). |
| Chunking | Token-aware recursive splitter, default **400 tokens, 60 overlap**; swept 200/400/800 | Defaults justified by the sweep, not asserted. |
| Embeddings | **BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU), 384-d**, normalised, cosine | $0 per query, no GPU, small vectors (1.5 KB each) keep the cost curve flat. |
| Idempotency | `chunk_id = sha256(doc_id, chunk config, chunk text)`; `merge_insert` on `chunk_id`; stale chunks of a changed doc deleted | Re-ingest leaves the count unchanged; proven by a test and a screenshot. |
| Filters | `doc_type`, `source`, `title` via LanceDB `where` | At least one metadata filter, exposed in the API. |
| No-context guard | Similarity floor calibrated on the eval set (skip the LLM below it) + abstain instruction + citation validation | Two layers; abstention measured on unanswerable questions. |
| API | FastAPI `POST /query`, `GET /stats`, `GET /health`, small web UI | Logs per-query latency (embed / retrieve / generate), chunk count, tokens. |
| Config | pydantic-settings, `.env.example`, no secrets in git | |

**Evaluation (fixed set):** 25 answerable + 5 unanswerable questions, each answerable one with a gold short answer and verbatim evidence quotes.
A chunk is relevant iff it contains an evidence quote (normalised, tolerant of a chunk boundary), so relevance does not depend on chunk settings.
Retrieval: Hit@k, Recall@k, MRR, nDCG@k, context precision. Answer: faithfulness (claim-level, LLM judge from a different family), answer relevance, EM/F1, abstention accuracy. Latency: p50/p95 retrieval over repeated runs, p95 end to end.
**Scale:** synthetic 100K and 1M vectors built from real embeddings plus noise; LanceDB flat vs IVF-PQ vs FAISS flat/HNSW: p50/p95, ANN recall vs exact, build time, size.
**Cost:** `cost_model.py` with every assumption in one YAML; 100K / 1M / 10M vectors; LanceDB on EC2+EBS and on S3 + serverless vs Pinecone Serverless Standard (official Nov 2026 rates) and a provisioned managed cluster.

## Problem 2: decisions

| Topic | Decision |
|---|---|
| Modes | Pointwise reference-based (scoring) + pairwise A-vs-B (comparison, position bias). README explains all four. |
| Rubric | Correctness, faithfulness, completeness, instruction-following, tone/safety (+ unsupported-length penalty); 1-5 anchors, weights, per-criterion rationale that must quote evidence. |
| Judge vs generator | Generator: Meta Llama family. Judges: Google Gemini and OpenAI gpt-oss (different families), optional ensemble. |
| Parsing | Provider JSON mode, then strict pydantic, then repair (fences, trailing commas, quotes, first balanced object), then one re-ask with the error, then `parse_failed` (never a silent 0). Fuzz-tested. |
| Audit | Every judge prompt and raw response to JSONL with model, params, tokens, latency, cost; `replay` re-parses offline; disk cache by prompt hash. |
| Aggregation | Per-case weighted score + gates; suite pass rate, mean per criterion, bootstrap CIs; A/B win rate with ties and a significance call. |
| Bias | Position: both orders, flip rate, agreement-or-tie. Verbosity: padded-answer probe, length-score correlation. Self-enhancement: cross-family judges, measured self-preference. Sycophancy: confidently-wrong probes, evidence grounding. Clustering: naive 1-10 vs anchored 1-5, spread/entropy. Each measured before and after. |
| Validation | Agreement + Cohen's kappa against real human expert labels (MT-Bench human judgments, CC-BY-4.0), test-retest flip rate, adversarial probe set. |

## Delivery

1. Public GitHub repo, narrated commit history, one PR.
2. One Vercel deployment: RAG API + UI + both evaluation reports (serverless, scales to zero, which is the cost thesis in practice).
3. Filled `submission_template.docx` (+ PDF) with diagrams, charts and real screenshots. The reflection prompts that the template says must be written without AI are left for the candidate.
4. Gmail draft to the recruiter with repo, live links, submission doc and resume attached. Not sent.
