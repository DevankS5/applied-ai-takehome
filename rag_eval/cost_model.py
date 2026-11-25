"""Monthly cost of serving N vectors: LanceDB (serverless on S3, or one VM) vs managed DBs.

    python -m rag_eval.cost_model     # reads rag_eval/cost_assumptions.yaml, writes results/p1/cost.{json,md}

Each line item is computed from a named assumption so the table can be audited and re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

A = yaml.safe_load(Path("rag_eval/cost_assumptions.yaml").read_text(encoding="utf-8"))
OUT = Path("results/p1")


def record_bytes() -> float:
    w = A["workload"]
    return w["dim"] * w["bytes_per_float"] + w["text_and_metadata_bytes"]


def gb(n: int) -> float:
    return n * record_bytes() / 1e9


def lancedb_serverless(n: int, qpm: int) -> dict:
    c, w = A["lancedb_serverless"], A["workload"]
    storage = gb(n) * c["storage_overhead"] * c["s3_storage_per_gb_month"]
    gets = qpm * c["gets_per_query"] / 1000 * c["s3_get_per_1000"]
    puts = n * w["monthly_churn_fraction"] / 1000 * c["puts_per_1000_records_written"] / 1000 * c["s3_put_per_1000"]
    compute = qpm * c["function_gb"] * c["function_seconds_per_query"] * c["function_price_per_gb_second"]
    requests = qpm / 1e6 * c["function_price_per_million_requests"]
    items = {"s3_storage": storage, "s3_requests": gets + puts, "function_compute": compute + requests}
    return {"total": sum(items.values()), "items": items}


def lancedb_vm(n: int, qpm: int) -> dict:
    c = A["lancedb_vm"]
    inst = c["instances"][n]
    disk = max(c["ebs_min_gb"], gb(n) * A["lancedb_serverless"]["storage_overhead"] * 1.5)  # 50% headroom
    items = {f"ec2_{inst['type']}": inst["per_hour"] * c["hours_per_month"], "ebs_gp3": disk * c["ebs_gp3_per_gb_month"]}
    return {"total": sum(items.values()), "items": items}


def pinecone(n: int, qpm: int) -> dict:
    c, w = A["pinecone_serverless"], A["workload"]
    size = gb(n)
    storage = size * c["storage_per_gb_month"]
    reads = qpm * max(c["min_read_units_per_query"], size) / 1e6 * c["read_units_per_million"]
    writes = n * w["monthly_churn_fraction"] * record_bytes() / 1000 * c["write_units_per_kb"] / 1e6 * c["write_units_per_million"]
    usage = storage + reads + writes
    items = {"storage": storage, "reads": reads, "writes": writes, "minimum_top_up": max(0.0, c["monthly_minimum"] - usage)}
    return {"total": max(usage, c["monthly_minimum"]), "items": items}


def qdrant(n: int, qpm: int) -> dict:
    v = A["qdrant_cloud"]["monthly_by_scale"][n]
    return {"total": v, "items": {"provisioned_cluster": v}}


OPTIONS = {
    "LanceDB on S3 + serverless function (ours, scale-to-zero)": lancedb_serverless,
    "LanceDB on one VM + EBS (ours, always-on)": lancedb_vm,
    "Pinecone Serverless, Standard plan (managed)": pinecone,
    "Qdrant Cloud, provisioned cluster (managed, always-on)": qdrant,
}


def main() -> None:
    w = A["workload"]
    rows = []
    for n in w["scales"]:
        for name, fn in OPTIONS.items():
            r = fn(n, w["queries_per_month"])
            rows.append({"vectors": n, "option": name, "monthly_usd": round(r["total"], 2), "items": {k: round(v, 4) for k, v in r["items"].items()}})
    sens = []
    for qpm in w["sensitivity_queries_per_month"]:
        for n in w["scales"]:
            sens.append({"queries_per_month": qpm, "vectors": n,
                         **{name: round(fn(n, qpm)["total"], 2) for name, fn in OPTIONS.items()}})
    res = {"record_bytes": record_bytes(), "data_gb": {n: round(gb(n), 3) for n in w["scales"]}, "base": rows, "sensitivity": sens,
           "assumptions": A}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def usd(x: float) -> str:
        return f"${x:,.2f}"

    by = {(r["vectors"], r["option"]): r["monthly_usd"] for r in rows}
    names = list(OPTIONS)
    md = ["# Problem 1: monthly cost across scale", "",
          f"Record = {w['dim']}-d float32 vector + {w['text_and_metadata_bytes']} B text/metadata = {record_bytes():,.0f} B. "
          f"Workload: {w['queries_per_month']:,} queries/month, {w['monthly_churn_fraction']:.0%} of records rewritten monthly. "
          "Assumptions and sources: `rag_eval/cost_assumptions.yaml`.", "",
          "| Vectors | Data | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for n in w["scales"]:
        md.append(f"| {n:,} | {gb(n):.2f} GB | " + " | ".join(usd(by[(n, o)]) for o in names) + " |")
    md += ["", "## Sensitivity to query volume", "", "| Queries / month | Vectors | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for s in sens:
        md.append(f"| {s['queries_per_month']:,} | {s['vectors']:,} | " + " | ".join(usd(s[o]) for o in names) + " |")
    (OUT / "cost.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
