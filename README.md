# Applied AI take-home: cost-efficient RAG and a bias-aware LLM judge

Two problems, one repo. Every number below is produced by a script in this repo and saved under [`results/`](results/); nothing is typed by hand.

| | |
|---|---|
| **Live demo (Problem 1)** | https://devsrivastava-cost-efficient-rag.hf.space (UI, `POST /query`, `/docs`; rate-limited, daily LLM budget) |
| **Problem 1** | RAG over a 17-document PDF/HTML/MD corpus on an embedded **LanceDB** table, hybrid retrieval, local embeddings, cited answers, a no-context guard. Evaluated on retrieval, answers, latency and cost. |
| **Problem 2** | LLM-as-judge pipeline: suite in, structured verdicts out, a written rubric, five judge biases named, mitigated in code and measured before/after, validated against MT-Bench expert labels. |

## Repository map

```
rag/            Problem 1 service: loaders, chunker, embeddings, LanceDB store, hybrid search, generation, FastAPI app
rag_eval/       Problem 1 evaluation: question sets, IR/QA metrics, retrieval + answer harnesses, scale bench, cost model
judge/          Problem 2 pipeline: rubric, schema, prompts, robust parser, judge runner, aggregation, CLI
judge_eval/     Problem 2 bias + validation experiments: probes, MT-Bench human labels, test-retest, self-preference
common/llm.py   Gemini + Groq clients over HTTPS: retry, rate spacing, disk cache, JSONL audit log, token/cost accounting
suites/         Problem 2 test suite (24 cases)          configs/   judge and generator configs (independent)
results/        every reported number (JSON + Markdown)  docs/      plan, diagrams, charts, screenshots, submission
tests/          unit tests for metrics, parser, aggregation and citation parsing (22 tests)
```

## Quick start (about 10 minutes)

Prerequisites: Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), about 1 GB disk. CPU only; no GPU, no database server.
Free API keys: [Groq](https://console.groq.com/keys) (generators) and [Gemini](https://aistudio.google.com/apikey) (judges).

```bash
git clone https://github.com/DevankS5/applied-ai-takehome && cd applied-ai-takehome
uv sync                                   # installs runtime + dev dependencies
cp .env.example .env                      # then paste GROQ_API_KEY and GEMINI_API_KEY into .env
uv run python scripts/fetch_corpus.py --verify     # 17 pinned documents; fails if any content drifted
uv run python -m rag.ingest data/corpus/raw         # chunk 400/60 tokens, embed, write LanceDB (~1 min)
uv run uvicorn rag.api:app --port 8000              # UI at http://localhost:8000, API docs at /docs
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What is the default hnsw.ef_search in pgvector?", "k": 5, "filters": {"doc_type": "md"}}'
uv run pytest                                        # 22 unit tests
```

Or with Docker: `docker build -t rag . && docker run -p 7860:7860 -e GROQ_API_KEY=... rag`. The live demo is a Hugging Face Space ([`deploy/hf-space/`](deploy/hf-space/)) that clones this repo, verifies the corpus and builds the index at startup.

Environment variables (names only; values live in `.env`, which is gitignored): `GROQ_API_KEY`, `GEMINI_API_KEY`,
`RAG_DB_PATH`, `RAG_TABLE`, `RAG_EMBED_MODEL`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_SEARCH_MODE`, `RAG_RERANK`,
`RAG_MIN_SCORE`, `RAG_GEN_PROVIDER`, `RAG_GEN_MODEL`, `RAG_RATE_LIMIT_PER_MIN`, `RAG_TRUST_X_REAL_IP`, `JUDGE_PROVIDER`, `JUDGE_MODEL`.

Reproduce every number:

```bash
uv run python -m rag_eval.run_retrieval      # retrieval metrics, modes, chunk sweep, latency, guard calibration
uv run python -m rag_eval.run_answers        # faithfulness, relevance, EM/F1, abstention, end-to-end latency
uv run python -m rag_eval.bench_scale        # 100K / 1M vector latency + ANN recall, LanceDB vs FAISS
uv run python -m rag_eval.cost_model         # monthly cost table from rag_eval/cost_assumptions.yaml
uv run python -m judge ab --suite suites/core.yaml --gen-a configs/gen_a.yaml --gen-b configs/gen_b.yaml
uv run python -m judge_eval.run_bias         # bias before/after + judge validation
uv run python scripts/make_charts.py         # every chart in this README
```

LLM calls are cached on disk by request hash, so a rerun reproduces the same outputs for free.

---

# Problem 1: cost-efficient RAG

![Problem 1 architecture](docs/img/p1_architecture.png)

## Design

**Store: LanceDB (embedded, Apache-2.0).** The background says the managed bill is driven by stored vectors on always-on pods. LanceDB separates storage from compute: the table is a set of Lance files on local disk or S3, and any process that opens them can search. A large, rarely queried index therefore costs object-storage rates plus compute only when a query runs, and the same code runs in a laptop, a container or a serverless function. It also gives SQL `WHERE` prefiltering on metadata, IVF-PQ / HNSW indexes for scale, `merge_insert` for idempotent upserts and a built-in BM25 index for hybrid search. Rejected: pgvector (excellent, but it needs an always-on Postgres, which is the cost shape we are trying to escape), Qdrant (its fast path is RAM-resident), Chroma (its embedded mode is local-disk only, so scaling out means running its server), sqlite-vec (brute force only, no ANN index yet), FAISS alone (a library, not a store: no metadata, no upserts, no persistence model). FAISS is benchmarked as the bonus second store.

**Chunking: 400 tokens, 60 overlap (defaults), token-aware.** Sizes are counted with the embedding model's own tokenizer and chunks are packed from whole sentences, never crossing a paragraph boundary; an oversized sentence is cut at token boundaries. Each chunk is embedded with its document title and section heading in front (a contextual header) and stores page (PDF) or heading (HTML/MD) for citations.

**Embeddings: `BAAI/bge-small-en-v1.5`, 384 dimensions, through fastembed's ONNX runtime on CPU.** A query embedding costs nothing and takes about 12 ms. 384-d float32 is 1.5 KB per vector, a quarter of a 1536-d API embedding, which is what keeps the cost curve flat. The model and dimension are written into the table schema.

**Retrieval: hybrid.** Dense cosine top-2k and BM25 top-2k fused with reciprocal rank fusion (`1 / (60 + rank)`), metadata filters (`doc_type`, `doc_id`, `source`, `title`) applied as a `WHERE` prefilter on both legs. A cross-encoder reranker (`jina-reranker-v1-tiny-en`) is available behind `RAG_RERANK=true`.

**No relevant context.** Two layers. (1) If the best cosine similarity among retrieved chunks is below `RAG_MIN_SCORE` (0.58, calibrated below), the service abstains without calling the LLM. (2) Otherwise the prompt allows only facts from the numbered passages, requires a citation on every sentence and an exact `INSUFFICIENT_CONTEXT` token when the passages do not answer. Citations are parsed and checked against the passages actually sent.

**Idempotent re-ingest.** `chunk_id = sha256(doc_id, embedding model, chunk size, overlap, chunk text)`. Chunks already in the table are not re-embedded, writes go through `merge_insert` on `chunk_id`, and a document's chunks that the new run no longer produces are deleted. Re-running on the same corpus embeds nothing and leaves the count at 431; editing one Wikipedia loader rule re-embedded only the 24 affected chunks and deleted the 18 they replaced.

![idempotent re-ingest](docs/img/p1_idempotent_reingest.png)

**Service.** `POST /query {question, k, filters}` returns the answer, cited passage numbers, every retrieved chunk with its score, latency per stage (embed / retrieve / generate / total), chunk counts and token usage, and logs the same as one JSON line. Config comes from environment variables only. The public demo has a per-client rate limit and an atomic daily LLM budget that falls back to retrieval-only answers.

## Evaluation set

30 fixed questions in [`rag_eval/questions.jsonl`](rag_eval/questions.jsonl): 25 answerable, spread over all 17 documents and all three formats, each with a gold short answer and verbatim **evidence quotes**; 5 unanswerable (off-topic, and in-domain near misses such as Weaviate defaults or GPT-4's parameter count). **A chunk is relevant iff it contains an evidence quote** (normalised; a chunk boundary may cut it as long as 60% survives as one run), so relevance does not depend on chunk settings and can be recomputed for every configuration. A second file rewords the same 25 questions with deliberately different vocabulary ([`questions_paraphrased.jsonl`](rag_eval/questions_paraphrased.jsonl)), because hand-written questions tend to copy the source's words and flatter keyword search.

## Retrieval results

Hybrid, k = 5, 431 chunks ([`results/p1/retrieval.md`](results/p1/retrieval.md)):

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Context precision@k |
|---|---|---|---|---|---|
| 1 | 0.44 | 0.29 | 0.44 | 0.44 | 0.44 |
| 3 | 0.80 | 0.63 | 0.61 | 0.55 | 0.62 |
| **5** | **0.92** | **0.79** | **0.64** | **0.62** | **0.62** |
| 10 | 0.92 | 0.83 | 0.64 | 0.63 | 0.60 |

Recall@k counts all relevant chunks (a quote can sit in two overlapping chunks); context precision is the RAGAS definition (mean precision@i over the ranks holding a relevant chunk).

![retrieval by mode](docs/img/p1_retrieval_modes.png)

| wording | mode | Hit@5 | MRR@10 | nDCG@5 | p50 ms |
|---|---|---|---|---|---|
| original | vector | 0.80 | 0.552 | 0.499 | 50 |
| original | BM25 | 0.92 | 0.808 | 0.766 | 34 |
| original | **hybrid** | 0.92 | 0.641 | 0.617 | 89 |
| original | hybrid + rerank | 0.92 | 0.800 | 0.772 | 2,257 |
| paraphrased | vector | 0.48 | 0.320 | 0.298 | 67 |
| paraphrased | BM25 | 0.60 | 0.522 | 0.463 | 57 |
| paraphrased | **hybrid** | 0.68 | 0.445 | 0.432 | 74 |
| paraphrased | hybrid + rerank | 0.76 | 0.494 | 0.507 | 2,456 |

What this says. BM25 wins on the original wording because those questions share words with the evidence; that advantage shrinks when the wording changes. Dense-only retrieval with a small embedding model is the weakest in both settings. Hybrid is never worse than either leg at Hit@5 and is the most robust choice without a reranker. The reranker buys the best ranking on both sets but costs about 2 s per query on a laptop CPU, so it is off by default. The `doc_type` filter works as a prefilter (Hit@5 0.92 with the gold document's type).

**Chunk size at an equal token budget.** Bigger chunks look better at a fixed k only because they send more text. Compared at the same number of tokens handed to the LLM, 400/60 sits with the best configurations:

![chunk size vs token budget](docs/img/p1_chunk_budget.png)

| size / overlap | Hit@5 (pooled) | context at k=5 | note |
|---|---|---|---|
| 200 / 30 | 0.78 | ~840 tokens | at k = 10 (~1.7K tokens) reaches 0.88: 4 of 50 questions better than 400/60 at k = 5, within noise |
| 400 / 0 | 0.76 | ~1.8K tokens | no overlap loses 2 questions to cut evidence |
| **400 / 60** | **0.80** | ~1.8K tokens | default |
| 800 / 100 | 0.86 | ~3.7K tokens | at ~2.2K tokens (k = 3) only 0.70 |

## Answer quality

All 30 questions through the full service (hybrid, k = 5). Generator `openai/gpt-oss-120b` on Groq; judge `gemini-3.1-flash-lite`, a different model family ([`results/p1/answers.md`](results/p1/answers.md)):

| Metric | Value | How |
|---|---|---|
| Faithfulness | **1.00** | judge splits each answer into atomic claims, checks each against the passages sent; supported / all claims (24 answers) |
| Answer relevance | **1.00** | judge 1-5, rescaled to 0-1 |
| Judge control | **11 / 11 caught** | first number in each answer changed (e.g. `ef_search is 40` -> `127`) and re-judged; every planted error was marked unsupported |
| Exact match / contains-match / token F1 | 0.00 / 0.21 / 0.31 | SQuAD normalisation against the gold short answer |
| Answered, answerable | 24 / 25 | the one miss (q15) is a retrieval miss: no relevant chunk in the top 5, and the model declined instead of guessing |
| Declined, unanswerable | **5 / 5** | 2 by the score guard (no LLM call), 3 by the model's `INSUFFICIENT_CONTEXT` token |
| Answers citing a valid passage | 1.00 | 0 out-of-range citation markers |
| End-to-end latency p50 / p95 | 871 / 1,646 ms | query embedding + retrieval + generation API time |

A perfect faithfulness score is a ceiling, and a ceiling is exactly where a lenient judge hides, so the run carries its own negative control: the same judge, given the same passages and an answer with one wrong number, flagged 11 of 11. EM is 0 by construction (answers are full cited sentences, the gold is a short span), which is why contains-match and F1 are reported beside it.

## Latency

Retrieval on the 431-chunk corpus, laptop CPU (AMD Ryzen 7 7435HS, 8 cores / 16 threads, 24 GB RAM, Windows 11), warm, k = 5, 600 queries per mode:

| mode | store p50 | store p95 | + query embedding p95 |
|---|---|---|---|
| vector | 43.9 ms | 89.5 ms | 116.5 ms |
| hybrid (default) | 47.1 ms | 95.5 ms | 116.9 ms |

At this size the time is LanceDB's per-query planning overhead, not search. Scale behaviour, on synthetic vectors built from the real embeddings ([`results/p1/scale_bench.md`](results/p1/scale_bench.md)):

![latency vs recall](docs/img/p1_scale.png)

| Vectors | System | p50 ms | p95 ms | Recall@10 |
|---|---|---|---|---|
| 100K | LanceDB flat scan (exact, on disk) | 133.6 | 156.7 | 1.00 |
| 100K | LanceDB IVF-PQ, nprobes 50, refine 20 | 15.3 | 16.5 | 0.96 |
| 100K | LanceDB IVF-PQ, nprobes 100, refine 50 | 24.0 | 30.3 | 0.99 |
| 100K | FAISS HNSW in RAM, efSearch 128 | 0.47 | 0.81 | 0.94 |
| 1M | LanceDB flat scan (exact, on disk) | 1,089.8 | 1,306.2 | 1.00 |
| 1M | LanceDB IVF-PQ, nprobes 50, refine 20 | 15.9 | 17.9 | 0.83 |
| **1M** | **LanceDB IVF-PQ, nprobes 100, refine 50** | **24.9** | **27.4** | **0.94** |
| 1M | FAISS HNSW in RAM, efSearch 256 | 1.53 | 2.21 | 0.83 |

LanceDB's IVF-PQ latency barely moves from 100K to 1M (15 to 16 ms at nprobes 50) while its exact scan grows linearly, so an index is mandatory beyond ~100K. FAISS HNSW is 10-20x faster because the whole graph lives in RAM (1.8 GB at 1M vs reading a 1.6 GB table from disk), which is exactly the always-on memory bill this design avoids. Caveat: the synthetic 1M set (interpolations between pairs of the 431 real embeddings plus noise) is denser than real data, so neighbours are nearly equidistant and absolute recall at 1M understates what real embeddings get; the useful reading is the recall-for-latency trade each knob buys.

## Cost

![monthly cost](docs/img/p1_cost.png)

Monthly USD at 100K queries/month ([`results/p1/cost.md`](results/p1/cost.md); every input is in [`rag_eval/cost_assumptions.yaml`](rag_eval/cost_assumptions.yaml)):

| Vectors | Data | LanceDB on S3 + function (ours) | LanceDB on one VM (ours) | Pinecone Serverless | Qdrant Cloud (provisioned) |
|---|---|---|---|---|---|
| 100K | 0.32 GB | **$2.46** | $13.86 | $50.00 (plan minimum) | $26.10 |
| 1M | 3.19 GB | **$2.54** | $26.13 | $50.00 (plan minimum) | $34.17 |
| 10M | 31.9 GB | **$3.41** | $54.03 | $74.23 | $205.02 |

Assumptions: 384-d float32 + 1,650 B text/metadata = 3,186 B per record; 10% of records rewritten monthly; us-east-1 on-demand prices checked 2026-11-25. Pinecone Standard rates from pinecone.io/pricing ($0.33/GB-month, $16 per million read units with one RU per GB of namespace per query, $4 per million write units, $50/month minimum). Qdrant figures read off cloud.qdrant.io/calculator (384-d, one replica, no quantization: 2 GiB node; 1 node for 3 GiB; 3 x 8 GiB nodes). LanceDB on S3: S3 Standard at $0.023/GB-month with 1.3x index/version overhead, 40 GETs per query, Lambda arm64 at 1,769 MB for 350 ms per query. LanceDB on a VM: t4g.small / medium / large plus gp3 EBS. Single replica everywhere, so no option is charged for HA.

Sensitivity: at 10M queries/month the serverless option costs ~$245 at every scale (compute per query dominates), the always-on VM stays at $14-$54 and Pinecone reaches $512 (1M vectors) to $5,121 (10M vectors). The cheapest design depends on traffic, not on vector count.

## Discussion

**What works.** Hybrid retrieval puts a relevant chunk in the top 5 for 23 of 25 questions; generation never cited a passage it was not given, never answered an unanswerable question, and the one false abstention traces to retrieval, not to the model. The store costs single-digit dollars a month at every scale tested, because nothing runs between queries.

**What does not, yet.** Rewording hurts: Hit@5 falls from 0.92 to 0.68 when the questions stop sharing words with the source. That is the price of a 384-d CPU embedding model; the reranker recovers part of it (0.76) at about 2 s per query on CPU. The serverless cost advantage inverts above roughly 1M queries a month, where an always-on VM is cheaper.

**How far to trust these numbers.** 25 answerable questions written by one person against a 17-document corpus: a single question moves Hit@5 by 0.04, so differences under about 0.08 are noise. The paraphrased set exists because hand-written questions flatter keyword search. Latency is a laptop, warm; the scale bench uses synthetic vectors (caveat above).

**Next.** Query rewriting or HyDE for paraphrase robustness before a bigger embedding model; the reranker on a GPU or a smaller cross-encoder; an evaluation set mined from real user questions; and IVF-PQ index creation in the ingest path once a table passes ~100K rows.

---

# Problem 2: LLM-as-judge with bias handling

![Problem 2 pipeline](docs/img/p2_pipeline.png)

## Design

**In / out.** A suite (YAML or JSON) of `{id, input, system_prompt, model_output?, expected_output?, context?, criteria?}`. Cases without `model_output` are generated first by a generator config. Out: a per-case structured verdict and a suite report (pass rate with bootstrap 95% CI, mean score, per-criterion and per-category means, score spread, parse statistics, tokens, cost). Judge and generators are configured in separate YAML files (`configs/judge_*.yaml`, `configs/gen_*.yaml`).

**Judge: Gemini 3.5 Flash-Lite (Google), temperature 0. Generators: gpt-oss-20b (OpenAI).** Different families, so the judge wrote neither side of the A/B.

**Modes.** Pointwise reference-based scoring against an explicit rubric for "is it good enough"; pairwise A-vs-B, always in both orders, for "which config is better".

**Rubric** ([`judge/rubric.yaml`](judge/rubric.yaml)): correctness 0.30, faithfulness 0.20, completeness 0.20, instruction-following 0.20, tone/safety 0.10, each scored 1-5 with written anchors for 1, 3 and 5. The judge must quote evidence before each score. Weights and the pass gate (overall >= 3.5, correctness >= 3, tone/safety >= 3) are applied in code, so the judge cannot argue the total up.

**Structured verdicts.** JSON mode plus a Pydantic schema. A parse ladder (strict, extract the first `{...}`, repair quotes/commas/literals/truncation, light type coercion) then one re-ask carrying the validation error; after that the case is `parse_failed` and counted, never silently scored. Every call is in `logs/judge_audit.jsonl` and `python -m judge replay` re-parses the whole log offline. Across 401 logged judge calls, 386 parsed first time and all 15 failures were recovered by the re-ask (typically the judge writing scores where the schema wanted `A`/`B`/`tie`).

## Bias handling: measured before (naive prompt) and after (v2)

Same judge model throughout; only the prompt and protocol change. The naive baseline is a bare "score 1-10" / "which is better" prompt.

| Bias | Mitigation in code | Before -> after |
|---|---|---|
| Position | `judge_pair_both_orders()`: run A/B and B/A, keep the winner only if both orders agree, else tie | MT-Bench pairs (n = 30): flip rate **23.3% -> 16.7%**, first-slot win rate when decisive 61% -> 54%; final verdicts flip 0% by construction. A/B suite flip rate 12.5% (3 of 24, all became ties) |
| Verbosity / length | length policy in the prompt, evidence before score, padded-answer probe | The naive judge's bias ran the other way: it **failed 9 of 12 correct-but-padded answers** (score 1/10, the same as a wrong answer); v2 fails 0 of 12, docking only instruction-following. Verbose-but-wrong passed: 0/12 -> 0/12 |
| Self-enhancement | judge from a different family; measured with a neutral third judge (Qwen) on 12 cases answered by gpt-oss-120b and Gemini | 33 of 36 both-order verdicts were ties; Gemini-for-Gemini -0.17, gpt-oss-for-gpt-oss +0.08: one or two cases each, no measurable self-preference at this n |
| Sycophancy / confident style | "polished wording is not evidence", reference as ground truth, confidently-wrong probe | Confident-but-wrong passed: 0/12 -> 0/12 |
| Score clustering | anchored 1/3/5 scale, calibration examples, weighted overall in code | Naive scores are bimodal (29 of 48 probes got 1, 12 got 10); normalised entropy 0.47 -> 0.85; good-vs-bad ranking AUC 0.90 -> **1.00** |

![bias before and after](docs/img/p2_bias_before_after.png)

## Judge validation

| Check | Result | Notes |
|---|---|---|
| Agreement with human experts | **66.7%** (v2, both orders), 63.3% naive | 30 MT-Bench first-turn pairs with expert votes (`lmsys/mt_bench_human_judgments`, CC BY 4.0), labels A / B / tie. On pairs where the human picked a side: 84% single-order. The MT-Bench paper reports about 66% for GPT-4 with ties and 85% without, so this is in the same range |
| Cohen's kappa vs humans | **0.47** (v2), 0.43 naive | 3 classes; moderate agreement |
| Probe set vs gold-by-construction | accuracy 81% -> **100%**, kappa 0.63 -> **1.00** | 48 probes: terse-correct, padded-correct, confident-wrong and verbose-wrong versions of 12 suite cases |
| Test-retest at temperature 0.7 | pointwise verdict flips **0%** (12 items x 3 samples, overall-score std 0.08); pairwise single-order winner flips **30%** (3 of 10 close MT-Bench pairs) | why production runs at temperature 0, cached, and pairwise only in both orders |
| Planted errors (Problem 1 judge) | 11 / 11 caught | see Problem 1 |

![raw judge prompt and response](docs/img/p2_audit.png)

## A/B comparison

`gpt-oss-20b` with the case prompt as given (A) vs the same model with prompt v2, which adds constraint re-reading, false-premise and no-padding guidance (B):

| Config | Pass rate | Mean score (1-5) | Pairwise (both orders) |
|---|---|---|---|
| A: prompt v1 | 100% | 4.95 | 1 win, 23 ties |
| B: prompt v2 | 100% | 4.95 | 0 wins |

**Declared result: no significant difference** (sign test p = 1.0). The suite is saturated for this model: both configs pass every case, so there is nothing for v2 to fix. The pipeline refuses to crown A on a 1-0 record, which is the behaviour a release gate needs. A harder suite (or a weaker model) is the next step to make this A/B informative.

## Would I let it gate a release?

For regressions, yes: fixed versioned suite and judge config, temperature 0, both-order pairwise with a significance test before any "B beats A", a minimum parse-success rate, and the probe set as a canary that must stay correctly classified. A human stays in the loop for safety failures, any pair where the orders disagree, cases near the pass threshold, a periodic re-labelled sample to re-measure agreement, and any change to the judge model or prompt.

**Limits.** 30 human-labelled pairs and 12-case probe groups give wide intervals; the suite is easy for current models; self-preference was not measurable at this size; one judge model, no ensemble.
