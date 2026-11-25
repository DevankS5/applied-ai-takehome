"""Render every chart in the README / submission from results/*.json (no numbers typed by hand).

    python scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
IMG = Path("docs/img")
C = {"blue": "#2456d6", "orange": "#d9822b", "green": "#1f8a5b", "red": "#c23b3b", "grey": "#8a93a3", "purple": "#7a4fd1"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 160, "savefig.bbox": "tight"})


def load(p: str) -> dict | None:
    f = R / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def retrieval_modes(res: dict) -> None:
    modes = ["vector", "bm25", "hybrid", "hybrid+rerank"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axes, ["original", "paraphrased"]):
        rows = {r["mode"]: r["metrics"] for r in res["modes"] if r["set"] == s}
        x = range(len(modes))
        b1 = ax.bar([i - 0.2 for i in x], [rows[m]["hit@5"] for m in modes], 0.4, label="Hit@5", color=C["blue"])
        b2 = ax.bar([i + 0.2 for i in x], [rows[m]["mrr@10"] for m in modes], 0.4, label="MRR@10", color=C["orange"])
        bar_labels(ax, b1)
        bar_labels(ax, b2)
        ax.set_xticks(list(x), modes)
        ax.set_title(f"{s} wording (25 questions)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Retrieval quality by mode: hybrid holds up when the wording changes", fontsize=11)
    fig.savefig(IMG / "p1_retrieval_modes.png")
    plt.close(fig)


def chunk_budget(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [C["grey"], C["purple"], C["blue"], C["orange"]]
    for s, col in zip(res["sweep"], colors):
        ks = [1, 3, 5, 10]
        xs = [s["mean_chunk_tokens"] * k for k in ks]
        ys = [s["metrics"][f"hit@{k}"] for k in ks]
        ax.plot(xs, ys, marker="o", color=col, label=f"{s['chunk_size']} / {s['overlap']} tokens")
        if (s["chunk_size"], s["overlap"]) == (400, 60):
            ax.annotate("default: k = 5", (xs[2], ys[2]), xytext=(10, -18), textcoords="offset points", fontsize=8, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("context sent to the LLM (tokens = mean chunk size x k)")
    ax.set_ylabel("Hit@k (both question sets)")
    ax.set_title("Chunk size at an equal token budget", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(IMG / "p1_chunk_budget.png")
    plt.close(fig)


def cost(res: dict) -> None:
    names = list(dict.fromkeys(r["option"] for r in res["base"]))
    short = ["LanceDB on S3 + function (ours)", "LanceDB on 1 VM (ours)", "Pinecone Serverless", "Qdrant Cloud (provisioned)"]
    cols = [C["green"], C["blue"], C["orange"], C["red"]]
    scales = sorted({r["vectors"] for r in res["base"]})
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    w = 0.2
    for i, (n, s, col) in enumerate(zip(names, short, cols)):
        vals = [next(r["monthly_usd"] for r in res["base"] if r["vectors"] == v and r["option"] == n) for v in scales]
        bars = ax.bar([j + (i - 1.5) * w for j in range(len(scales))], vals, w, label=s, color=col)
        bar_labels(ax, bars, "${:,.0f}")
    ax.set_xticks(range(len(scales)), [f"{v:,} vectors" for v in scales])
    ax.set_yscale("log")
    ax.set_ylabel("USD / month (log scale)")
    ax.set_title("Monthly cost at 100K queries/month", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(IMG / "p1_cost.png")
    plt.close(fig)


def scale(res: dict) -> None:
    rows = res["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, n in zip(axes, sorted({r["vectors"] for r in rows})):
        rs = [r for r in rows if r["vectors"] == n]
        for r in rs:
            col = C["blue"] if "LanceDB" in r["system"] else C["orange"]
            ax.scatter(r["p50_ms"], r["recall@10"], color=col, s=40)
            label = r["system"].replace(" (exact, in RAM)", " exact").replace(" (exact, on disk)", " exact").split(" (")[0]
            extra = r["system"].split("(")[-1].split(",")[-2:] if "IVF" in r["system"] or "HNSW" in r["system"] else []
            ax.annotate(label + (" " + ",".join(e.strip(") ") for e in extra) if extra else ""), (r["p50_ms"], r["recall@10"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("p50 latency, ms (log)")
        ax.set_ylabel("recall@10 vs exact")
        ax.set_title(f"{n:,} vectors")
        ax.set_ylim(0.4, 1.05)
    fig.suptitle("Latency vs recall (blue LanceDB on disk, orange FAISS in RAM)", fontsize=11)
    fig.savefig(IMG / "p1_scale.png")
    plt.close(fig)


def p2(res: dict) -> None:
    if "mtbench" in res and "probes" in res:
        m, p = res["mtbench"], res["probes"]
        labels = ["position flip rate\n(MT-Bench pairs)", "fooled: confident\nwrong", "fooled: verbose\nwrong", "fooled: terse\ncorrect",
                  "padded scored above\nterse (share)"]
        vp = p["verbosity"]

        def share(v):
            return v["padded_higher"] / v["pairs"] if v["pairs"] else 0

        naive = [m["naive"]["flip_rate"], p["naive"]["fooled"]["confident_wrong"]["rate"], p["naive"]["fooled"]["verbose_wrong"]["rate"],
                 p["naive"]["fooled"]["terse_correct"]["rate"], share(vp["naive_score_1_10"])]
        v2 = [m["v2"]["flip_rate"], p["v2"]["fooled"]["confident_wrong"]["rate"], p["v2"]["fooled"]["verbose_wrong"]["rate"],
              p["v2"]["fooled"]["terse_correct"]["rate"], share(vp["v2_overall_1_5"])]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = range(len(labels))
        b1 = ax.bar([i - 0.2 for i in x], naive, 0.4, label="naive prompt (before)", color=C["red"])
        b2 = ax.bar([i + 0.2 for i in x], v2, 0.4, label="v2 rubric prompt (after)", color=C["green"])
        bar_labels(ax, b1)
        bar_labels(ax, b2)
        ax.set_xticks(list(x), labels, fontsize=8)
        ax.set_ylabel("rate (lower is better)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
        ax.set_title("Judge biases before and after mitigation (same judge model)", fontsize=11)
        fig.savefig(IMG / "p2_bias_before_after.png")
        plt.close(fig)

        rows = p["rows"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, key, scale_max, title in [(axes[0], "naive_score", 10, "naive: bare 1-10 score"), (axes[1], "v2_overall", 5, "v2: weighted rubric overall (1-5)")]:
            good = [r[key] for r in rows if r[key] is not None and r["gold"] == "pass"]
            bad = [r[key] for r in rows if r[key] is not None and r["gold"] == "fail"]
            bins = [b / 2 for b in range(2, 2 * scale_max + 2)] if scale_max == 5 else list(range(1, 12))
            ax.hist([good, bad], bins=bins, label=["gold pass", "gold fail"], color=[C["green"], C["red"]], alpha=0.85)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("score")
        axes[0].set_ylabel("probe items")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Score distributions on 48 labelled probes", fontsize=11)
        fig.savefig(IMG / "p2_score_distribution.png")
        plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    if r := load("p1/retrieval.json"):
        retrieval_modes(r)
        chunk_budget(r)
    if r := load("p1/cost.json"):
        cost(r)
    if r := load("p1/scale_bench.json"):
        scale(r)
    if r := load("p2/bias_validation.json"):
        p2(r)
    print(sorted(p.name for p in IMG.glob("*.png")))


if __name__ == "__main__":
    main()
