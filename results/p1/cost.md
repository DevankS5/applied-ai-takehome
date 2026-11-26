# Problem 1: monthly cost across scale

Record = 384-d float32 vector + 1650 B text/metadata = 3,186 B. Workload: 100,000 queries/month, 10% of records rewritten monthly. Assumptions and sources: `rag_eval/cost_assumptions.yaml`.

| Vectors | Data | LanceDB on S3 + serverless function (ours, scale-to-zero) | LanceDB on one VM + EBS (ours, always-on) | Pinecone Serverless, Standard plan (managed) | Qdrant Cloud, provisioned cluster (managed, always-on) |
|---|---|---|---|---|---|
| 100,000 | 0.32 GB | $2.46 | $13.86 | $50.00 | $26.10 |
| 1,000,000 | 3.19 GB | $2.54 | $26.13 | $50.00 | $34.17 |
| 10,000,000 | 31.86 GB | $3.41 | $54.03 | $74.23 | $205.02 |

## Sensitivity to query volume

| Queries / month | Vectors | LanceDB on S3 + serverless function (ours, scale-to-zero) | LanceDB on one VM + EBS (ours, always-on) | Pinecone Serverless, Standard plan (managed) | Qdrant Cloud, provisioned cluster (managed, always-on) |
|---|---|---|---|---|---|
| 100,000 | 100,000 | $2.46 | $13.86 | $50.00 | $26.10 |
| 100,000 | 1,000,000 | $2.54 | $26.13 | $50.00 | $34.17 |
| 100,000 | 10,000,000 | $3.41 | $54.03 | $74.23 | $205.02 |
| 1,000,000 | 100,000 | $24.47 | $13.86 | $50.00 | $26.10 |
| 1,000,000 | 1,000,000 | $24.55 | $26.13 | $53.30 | $34.17 |
| 1,000,000 | 10,000,000 | $25.42 | $54.03 | $533.02 | $205.02 |
| 10,000,000 | 100,000 | $244.56 | $13.86 | $51.21 | $26.10 |
| 10,000,000 | 1,000,000 | $244.65 | $26.13 | $512.09 | $34.17 |
| 10,000,000 | 10,000,000 | $245.52 | $54.03 | $5,120.86 | $205.02 |
