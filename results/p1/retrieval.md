# Problem 1: retrieval evaluation

Default config: hybrid search (dense bge-small + BM25, RRF), table `chunks`, 431 vectors, mean chunk 354.5 tokens, 25 answerable questions.

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |
|---|---|---|---|---|---|---|
| 1 | 0.440 | 0.290 | 0.440 | 0.440 | 0.440 | 0.440 |
| 3 | 0.800 | 0.630 | 0.613 | 0.550 | 0.307 | 0.617 |
| 5 | 0.920 | 0.793 | 0.641 | 0.617 | 0.232 | 0.621 |
| 10 | 0.920 | 0.833 | 0.641 | 0.631 | 0.124 | 0.599 |

## Retrieval mode x question wording (k = 5)

| question set | mode | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Ctx precision@5 | p50 ms (incl. embed) |
|---|---|---|---|---|---|---|---|---|
| original | vector | 0.36 | 0.80 | 0.63 | 0.552 | 0.499 | 0.543 | 49.6 |
| original | bm25 | 0.72 | 0.92 | 0.83 | 0.808 | 0.766 | 0.791 | 34.1 |
| original | hybrid | 0.44 | 0.92 | 0.79 | 0.641 | 0.617 | 0.621 | 88.6 |
| original | hybrid+rerank | 0.68 | 0.92 | 0.83 | 0.800 | 0.772 | 0.803 | 2256.8 |
| paraphrased | vector | 0.16 | 0.48 | 0.41 | 0.320 | 0.298 | 0.286 | 66.8 |
| paraphrased | bm25 | 0.44 | 0.60 | 0.51 | 0.522 | 0.463 | 0.503 | 56.5 |
| paraphrased | hybrid | 0.28 | 0.68 | 0.58 | 0.445 | 0.432 | 0.408 | 74.2 |
| paraphrased | hybrid+rerank | 0.28 | 0.76 | 0.66 | 0.494 | 0.507 | 0.484 | 2455.8 |

Metadata filter check (hybrid, `doc_type` = gold document's type): Hit@5 0.92, MRR@10 0.635, nDCG@5 0.624.

## Chunk size / overlap sweep (hybrid, k = 5, both question sets pooled)

| size / overlap | vectors | mean tokens | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Ctx precision@5 |
|---|---|---|---|---|---|---|---|
| 200 / 30 | 876 | 168.2 | 0.78 | 0.67 | 0.514 | 0.510 | 0.498 |
| 400 / 0 | 377 | 363.7 | 0.76 | 0.64 | 0.574 | 0.538 | 0.552 |
| 400 / 60 | 431 | 354.5 | 0.80 | 0.69 | 0.543 | 0.525 | 0.514 |
| 800 / 100 | 207 | 733.2 | 0.86 | 0.77 | 0.623 | 0.620 | 0.608 |

## Latency (laptop CPU, warm)

| mode | samples | store p50 ms | store p95 ms | embed p50 ms | embed + store p95 ms |
|---|---|---|---|---|---|
| vector | 600 | 43.87 | 89.51 | 14.11 | 116.5 |
| bm25 | 600 | 9.96 | 34.93 | 13.11 | 66.25 |
| hybrid | 600 | 47.14 | 95.45 | 11.92 | 116.9 |

## No-context guard

Lowest top-1 cosine of an answerable question (both wordings): 0.6279; highest of an unanswerable one: 0.7404. Floor = lowest answerable - 0.05 = **0.58**: keeps 100% of answerable questions and stops 40% of unanswerable ones before any LLM call.
