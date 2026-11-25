# Problem 1: scale benchmark (single query, k = 10, laptop CPU)

300 queries; synthetic vectors = interpolations between two real bge-small chunk embeddings + noise (see module docstring). Recall@10 is measured against FAISS exact search on the same vectors.

| Vectors | System | Build / write (s) | Size (MB) | p50 ms | p95 ms | Recall@10 |
|---|---|---|---|---|---|---|
| 100,000 | FAISS IndexFlatIP (exact, in RAM) | 0.0 | 153.6 | 9.06 | 10.14 | 1.0 |
| 100,000 | FAISS HNSW (M=32, efSearch=64, in RAM) | 5.2 | 179.2 | 0.28 | 0.46 | 0.8763 |
| 100,000 | FAISS HNSW (M=32, efSearch=128, in RAM) | 5.2 | 179.2 | 0.47 | 0.81 | 0.9447 |
| 100,000 | FAISS HNSW (M=32, efSearch=256, in RAM) | 5.2 | 179.2 | 0.94 | 1.47 | 0.9803 |
| 100,000 | LanceDB flat scan (exact, on disk) | 0.4 | 153.8 | 133.6 | 156.72 | 1.0 |
| 100,000 | LanceDB IVF-PQ (316 parts, 48 subvec, nprobes=20, refine=10) | 16.7 | 159.8 | 12.22 | 14.04 | 0.882 |
| 100,000 | LanceDB IVF-PQ (316 parts, 48 subvec, nprobes=50, refine=20) | 16.7 | 159.8 | 15.3 | 16.48 | 0.964 |
| 100,000 | LanceDB IVF-PQ (316 parts, 48 subvec, nprobes=100, refine=50) | 16.7 | 159.8 | 23.95 | 30.31 | 0.9913 |
| 1,000,000 | FAISS IndexFlatIP (exact, in RAM) | 0.3 | 1536.0 | 80.0 | 96.99 | 1.0 |
| 1,000,000 | FAISS HNSW (M=32, efSearch=64, in RAM) | 77.6 | 1792.0 | 0.59 | 1.57 | 0.592 |
| 1,000,000 | FAISS HNSW (M=32, efSearch=128, in RAM) | 77.6 | 1792.0 | 0.94 | 1.64 | 0.7123 |
| 1,000,000 | FAISS HNSW (M=32, efSearch=256, in RAM) | 77.6 | 1792.0 | 1.53 | 2.21 | 0.83 |
| 1,000,000 | LanceDB flat scan (exact, on disk) | 5.6 | 1538.4 | 1089.79 | 1306.16 | 1.0 |
| 1,000,000 | LanceDB IVF-PQ (1000 parts, 48 subvec, nprobes=20, refine=10) | 37.1 | 1591.7 | 12.39 | 18.58 | 0.6763 |
| 1,000,000 | LanceDB IVF-PQ (1000 parts, 48 subvec, nprobes=50, refine=20) | 37.1 | 1591.7 | 15.9 | 17.91 | 0.8267 |
| 1,000,000 | LanceDB IVF-PQ (1000 parts, 48 subvec, nprobes=100, refine=50) | 37.1 | 1591.7 | 24.92 | 27.43 | 0.9397 |
