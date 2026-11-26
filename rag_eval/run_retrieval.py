"""Retrieval evaluation on the fixed question set (no LLM needed).

    python -m rag_eval.run_retrieval              # modes x question sets, chunk sweep, latency, threshold
    python -m rag_eval.run_retrieval --no-sweep

Two question sets share the same gold evidence: `questions.jsonl` (wording close to the
source text) and `questions_paraphrased.jsonl` (same intent, deliberately different words),
so keyword and dense retrieval are each tested where they should be weak.
Writes results/p1/retrieval.json and results/p1/retrieval.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rag.embed import embed_query
from rag.ingest import ingest
from rag.store import Store
from rag_eval import metrics as M

QFILE = Path("rag_eval/questions.jsonl")
PFILE = Path("rag_eval/questions_paraphrased.jsonl")
OUT = Path("results/p1")
KS = [1, 3, 5, 10]
MODES = ["vector", "bm25", "hybrid", "hybrid+rerank"]
SWEEP = [(200, 30), (400, 0), (400, 60), (800, 100)]  # (chunk_size, overlap); 400/60 is the default
TYPE_OF = {"md": "md", "wiki": "html"}


def load_questions(paraphrased: bool = False) -> list[dict]:
    qs = [json.loads(line) for line in QFILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if paraphrased:
        para = {p["id"]: p["question"] for p in map(json.loads, PFILE.read_text(encoding="utf-8").splitlines()) if p}
        qs = [{**q, "question": para[q["id"]]} for q in qs if q["id"] in para]
    return qs


def relevant_ids(store: Store, questions: list[dict]) -> dict[str, set[str]]:
    rows = store.table.search().select(["chunk_id", "text"]).limit(None).to_list()
    return {q["id"]: {r["chunk_id"] for r in rows if M.is_relevant(r["text"], q["evidence"])} for q in questions if q["evidence"]}


def search_fn(store: Store, mode: str, filtered: bool = False):
    def run(q: dict, k: int) -> list[dict]:
        flt = {"doc_type": TYPE_OF.get(q["doc_id"].split("-")[0], "pdf")} if filtered else None
        v = embed_query(q["question"])
        if mode == "hybrid+rerank":
            from rag.rerank import rerank

            rows, _ = store.search(v, 4 * k, flt, text=q["question"], mode="hybrid")
            return rerank(q["question"], rows, k)
        return store.search(v, k, flt, text=q["question"], mode=mode)[0]

    return run


def evaluate(store: Store, questions: list[dict], mode: str = "hybrid", filtered: bool = False) -> dict:
    rel = relevant_ids(store, questions)
    fn, kmax, per_q, lat = search_fn(store, mode, filtered), max(KS), [], []
    for q in questions:
        if not q["evidence"]:
            continue
        t0 = time.perf_counter()
        rows = fn(q, kmax)
        lat.append((time.perf_counter() - t0) * 1000)
        rels = [r["chunk_id"] in rel[q["id"]] for r in rows]
        n_rel = len(rel[q["id"]])
        row = {"id": q["id"], "n_relevant": n_rel, "first_relevant_rank": next((i + 1 for i, r in enumerate(rels) if r), None)}
        for k in KS:
            row |= {f"hit@{k}": M.hit_at_k(rels, k), f"recall@{k}": M.recall_at_k(rels, k, n_rel), f"mrr@{k}": M.reciprocal_rank(rels, k),
                    f"ndcg@{k}": M.ndcg_at_k(rels, k, n_rel), f"precision@{k}": M.precision_at_k(rels, k),
                    f"context_precision@{k}": M.context_precision(rels, k)}
        per_q.append(row)
    agg = {key: round(float(np.mean([r[key] for r in per_q])), 4) for key in per_q[0] if "@" in key}
    tokens = store.table.search().select(["n_tokens"]).limit(None).to_list()
    return {"table": store.name, "mode": mode, "filtered": filtered, "vectors": store.count(),
            "mean_chunk_tokens": round(float(np.mean([t["n_tokens"] for t in tokens])), 1), "questions": len(per_q),
            "questions_without_relevant_chunk": [qid for qid, ids in rel.items() if not ids], "metrics": agg,
            "query_ms_p50_incl_embed": round(M.pct(lat, 50), 1), "per_question": per_q}


def latency(store: Store, questions: list[dict], k: int = 5, repeats: int = 20) -> dict:
    """p50/p95 of the store call alone and of embed + store, per mode, after a warm-up pass."""
    out = {}
    for mode in ["vector", "bm25", "hybrid"]:
        embed_ms, search_ms = [], []
        for q in questions:
            store.search(embed_query(q["question"]), k, text=q["question"], mode=mode)
        for _ in range(repeats):
            for q in questions:
                t0 = time.perf_counter()
                v = embed_query(q["question"])
                embed_ms.append((time.perf_counter() - t0) * 1000)
                search_ms.append(store.search(v, k, text=q["question"], mode=mode)[1])
        out[mode] = {"k": k, "samples": len(search_ms), "vectors": store.count(),
                     "retrieval_p50_ms": round(M.pct(search_ms, 50), 2), "retrieval_p95_ms": round(M.pct(search_ms, 95), 2),
                     "embed_p50_ms": round(M.pct(embed_ms, 50), 2), "embed_p95_ms": round(M.pct(embed_ms, 95), 2),
                     "embed_plus_retrieval_p95_ms": round(M.pct(list(np.add(embed_ms, search_ms)), 95), 2)}
    return out


def calibrate_threshold(store: Store, questions: list[dict], margin: float = 0.05) -> dict:
    """Top cosine similarity of answerable vs unanswerable questions, and the guard floor.
    Floor = (lowest answerable top score - margin): the guard must not cost recall on paraphrases
    it has not seen; the LLM's abstain rule is the second layer for whatever passes."""
    top = {q["id"]: store.search(embed_query(q["question"]), 1, mode="vector")[0][0]["score"] for q in questions}
    ans = {k: v for k, v in top.items() if k.startswith("q")}
    una = {k: v for k, v in top.items() if k.startswith("u")}
    floor = round(min(ans.values()) - margin, 2)
    grid = [{"threshold": round(float(t), 2), "answerable_kept": round(float(np.mean([s >= t for s in ans.values()])), 3),
             "unanswerable_rejected": round(float(np.mean([s < t for s in una.values()])), 3)} for t in np.arange(0.40, 0.81, 0.01)]
    return {"top1_cosine": {k: round(v, 4) for k, v in top.items()}, "min_answerable": round(min(ans.values()), 4),
            "max_unanswerable": round(max(una.values()), 4), "margin": margin, "floor": floor,
            "at_floor": {"answerable_kept": float(np.mean([s >= floor for s in ans.values()])),
                         "unanswerable_rejected": float(np.mean([s < floor for s in una.values()]))}, "grid": grid}


def to_markdown(res: dict) -> str:
    d = res["default"]["metrics"]
    L = ["# Problem 1: retrieval evaluation", "",
         f"Default config: hybrid search (dense bge-small + BM25, RRF), table `{res['default']['table']}`, {res['default']['vectors']} vectors, "
         f"mean chunk {res['default']['mean_chunk_tokens']} tokens, {res['default']['questions']} answerable questions.", "",
         "| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |", "|---|---|---|---|---|---|---|"]
    for k in KS:
        L.append(f"| {k} | {d[f'hit@{k}']:.3f} | {d[f'recall@{k}']:.3f} | {d[f'mrr@{k}']:.3f} | {d[f'ndcg@{k}']:.3f} | {d[f'precision@{k}']:.3f} | {d[f'context_precision@{k}']:.3f} |")
    L += ["", "## Retrieval mode x question wording (k = 5)", "",
          "| question set | mode | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Ctx precision@5 | p50 ms (incl. embed) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["modes"]:
        m = r["metrics"]
        L.append(f"| {r['set']} | {r['mode']} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['recall@5']:.2f} | {m['mrr@10']:.3f} | {m['ndcg@5']:.3f} | {m['context_precision@5']:.3f} | {r['query_ms_p50_incl_embed']} |")
    f = res["filtered"]["metrics"]
    L += ["", f"Metadata filter check (hybrid, `doc_type` = gold document's type): Hit@5 {f['hit@5']:.2f}, MRR@10 {f['mrr@10']:.3f}, nDCG@5 {f['ndcg@5']:.3f}.", ""]
    if res.get("sweep"):
        L += ["## Chunk size / overlap sweep (hybrid, k = 5, both question sets pooled)", "",
              "| size / overlap | vectors | mean tokens | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Ctx precision@5 |", "|---|---|---|---|---|---|---|---|"]
        for s in res["sweep"]:
            m = s["metrics"]
            L.append(f"| {s['chunk_size']} / {s['overlap']} | {s['vectors']} | {s['mean_chunk_tokens']} | {m['hit@5']:.2f} | {m['recall@5']:.2f} | {m['mrr@10']:.3f} | {m['ndcg@5']:.3f} | {m['context_precision@5']:.3f} |")
    L += ["", "## Latency (laptop CPU, warm)", "", "| mode | samples | store p50 ms | store p95 ms | embed p50 ms | embed + store p95 ms |", "|---|---|---|---|---|---|"]
    for mode, x in res["latency"].items():
        L.append(f"| {mode} | {x['samples']} | {x['retrieval_p50_ms']} | {x['retrieval_p95_ms']} | {x['embed_p50_ms']} | {x['embed_plus_retrieval_p95_ms']} |")
    c = res["threshold"]
    L += ["", "## No-context guard", "", f"Lowest top-1 cosine of an answerable question (both wordings): {c['min_answerable']}; highest of an unanswerable one: "
          f"{c['max_unanswerable']}. Floor = lowest answerable - {c['margin']} = **{c['floor']}**: keeps {c['at_floor']['answerable_kept']:.0%} of answerable "
          f"questions and stops {c['at_floor']['unanswerable_rejected']:.0%} of unanswerable ones before any LLM call."]
    return "\n".join(L) + "\n"


def pooled(a: dict, b: dict) -> dict:
    return {k: round((a[k] + b[k]) / 2, 4) for k in a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    orig, para = load_questions(), load_questions(paraphrased=True)
    store = Store()
    res = {"default": evaluate(store, orig, "hybrid"), "filtered": evaluate(store, orig, "hybrid", filtered=True)}
    res["filtered"].pop("per_question")
    res["modes"] = []
    for set_name, qs in [("original", orig), ("paraphrased", para)]:
        for mode in MODES:
            r = evaluate(store, qs, mode)
            res["modes"].append({"set": set_name, "mode": mode, "metrics": r["metrics"], "query_ms_p50_incl_embed": r["query_ms_p50_incl_embed"],
                                 "per_question": r["per_question"]})
    if not a.no_sweep:
        res["sweep"] = []
        for size, ov in SWEEP:
            name = f"sweep_c{size}_o{ov}"
            ingest(Path("data/corpus/raw"), size, ov, table=name)
            st = Store(table=name)
            r1, r2 = evaluate(st, orig, "hybrid"), evaluate(st, para, "hybrid")
            res["sweep"].append({"chunk_size": size, "overlap": ov, "vectors": r1["vectors"], "mean_chunk_tokens": r1["mean_chunk_tokens"],
                                 "metrics": pooled(r1["metrics"], r2["metrics"]), "original": r1["metrics"], "paraphrased": r2["metrics"],
                                 "questions_without_relevant_chunk": r1["questions_without_relevant_chunk"]})
    res["latency"] = latency(store, orig)
    res["threshold"] = calibrate_threshold(store, orig + [{**q, "id": q["id"] + "p"} for q in para])
    (OUT / "retrieval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (OUT / "retrieval.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
