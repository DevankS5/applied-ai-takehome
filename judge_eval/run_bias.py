"""Bias measurement and judge validation. Every number in the Problem 2 write-up comes from here.

    python -m judge_eval.run_bias                  # all experiments
    python -m judge_eval.run_bias --only probes mtbench

Experiments (naive prompt = baseline, v2 = mitigated; same judge model):
  probes     48 labelled adversarial items -> accuracy/kappa vs constructed gold, fooled rates,
             verbosity delta (padded - terse), score clustering (spread, separation)
  mtbench    30 MT-Bench expert pairs -> position flip rate, first-position win rate,
             agreement + Cohen's kappa with human experts, single order vs both-order consensus
  retest     test-retest at temperature 0.7: verdict flips across 3 samples (pointwise + pairwise)
  selfpref   Llama vs Gemini answers judged by a Gemini judge, a Llama judge and a neutral gpt-oss judge
LLM responses are cached on disk, so re-running only pays for calls not made before.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from judge.__main__ import load_suite
from judge.aggregate import cohens_kappa, overall, passes, score_spread
from judge.generate import GenConfig, generate
from judge.judge import JudgeConfig, judge_pair, judge_pair_both_orders, judge_pointwise
from judge_eval import mtbench, probes

OUT = Path("results/p2")
V2 = JudgeConfig(name="gemini-v2", prompt="v2")
NAIVE = JudgeConfig(name="gemini-naive", prompt="naive")


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a gold-pass item > score of a gold-fail item), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def exp_probes(cases) -> dict:
    items = probes.build(cases)
    rows = []
    for it in items:
        v2 = judge_pointwise(it, V2, tag="probe")
        nv = judge_pointwise(it, NAIVE, tag="probe")
        rows.append({
            "id": it.id, "kind": it.category, "gold": it.gold,
            "v2_ok": v2.ok, "v2_overall": overall(v2.verdict) if v2.ok else None,
            "v2_pass": passes(v2.verdict) if v2.ok else None,
            "v2_scores": {k: s["score"] for k, s in v2.verdict["criteria"].items()} if v2.ok else None,
            "naive_ok": nv.ok, "naive_score": nv.verdict["score"] if nv.ok else None,
            "naive_pass": (nv.verdict["score"] >= 7) if nv.ok else None,
            "v2_raw": v2.raw[-1] if v2.raw else None,
        })

    def judge_stats(prefix: str) -> dict:
        ok = [r for r in rows if r[f"{prefix}_pass"] is not None]
        gold = ["pass" if r["gold"] == "pass" else "fail" for r in ok]
        pred = ["pass" if r[f"{prefix}_pass"] else "fail" for r in ok]
        fooled = {}
        for kind in ["terse_correct", "padded_correct", "confident_wrong", "verbose_wrong"]:
            ks = [r for r in ok if r["kind"] == kind]
            bad = [r for r in ks if (r[f"{prefix}_pass"] != (r["gold"] == "pass"))]
            fooled[kind] = {"n": len(ks), "fooled": len(bad), "rate": round(len(bad) / len(ks), 3) if ks else None}
        return {"n": len(ok), "accuracy": round(float(np.mean([g == p for g, p in zip(gold, pred)])), 4),
                "kappa": round(cohens_kappa(gold, pred), 4), "fooled": fooled}

    def verbosity(score_key: str) -> dict:
        by = {}
        for r in rows:
            base, kind = r["id"].split(":")
            by.setdefault(base, {})[kind] = r[score_key]
        d = [v["padded_correct"] - v["terse_correct"] for v in by.values()
             if v.get("padded_correct") is not None and v.get("terse_correct") is not None]
        return {"pairs": len(d), "mean_delta": round(float(np.mean(d)), 3), "padded_higher": sum(x > 0 for x in d),
                "padded_equal": sum(x == 0 for x in d), "padded_lower": sum(x < 0 for x in d)}

    def clustering(key: str, scale: int) -> dict:
        vals = [r[key] for r in rows if r[key] is not None]
        pos = [r[key] for r in rows if r[key] is not None and r["gold"] == "pass"]
        neg = [r[key] for r in rows if r[key] is not None and r["gold"] == "fail"]
        sp = score_spread(vals, scale)
        lo, hi = (1, scale)
        return {**sp, "std_over_range": round(sp["std"] / (hi - lo), 3), "mean_gold_pass": round(float(np.mean(pos)), 3),
                "mean_gold_fail": round(float(np.mean(neg)), 3),
                "separation_over_range": round((np.mean(pos) - np.mean(neg)) / (hi - lo), 3), "auc": round(auc(pos, neg), 4)}

    return {"items": len(items), "naive": judge_stats("naive"), "v2": judge_stats("v2"),
            "verbosity": {"naive_score_1_10": verbosity("naive_score"), "v2_overall_1_5": verbosity("v2_overall")},
            "clustering": {"naive_1_10": clustering("naive_score", 10), "v2_overall_1_5": clustering("v2_overall", 5)},
            "rows": rows}


def exp_mtbench(n: int = 30) -> dict:
    items = mtbench.sample(n)
    rows = []
    for it in items:
        rec = {"id": it["case"].id, "human": it["human"], "models": [it["model_a"], it["model_b"]]}
        for cfg in (NAIVE, V2):
            r = judge_pair_both_orders(it["case"], it["a"], it["b"], cfg, tag="mtbench")
            # presented-position winners: in the swapped call, B is the original A
            rec[cfg.prompt] = {"ab": r["order_xy"], "ba_mapped": r["order_yx_mapped"], "flipped": r["flipped"], "final": r["final"],
                               "first_pos_wins": [r["order_xy"] == "A", r["calls"][1]["verdict"]["winner"] == "A" if r["calls"][1]["ok"] else None]}
        rows.append(rec)

    def stats(p: str) -> dict:
        ok = [r for r in rows if r[p]["ab"] and r[p]["ba_mapped"]]
        flips = [r[p]["flipped"] for r in ok]
        decisive_first = [w for r in ok for w, raw in zip(r[p]["first_pos_wins"], [r[p]["ab"], r[p]["ba_mapped"]]) if raw != "tie" and w is not None]
        human = [r["human"] for r in ok]
        single = [r[p]["ab"] for r in ok]
        both = [r[p]["final"] for r in ok]
        nt = [(h, s, b) for h, s, b in zip(human, single, both) if h != "tie"]
        return {
            "pairs": len(ok), "flip_rate": round(float(np.mean(flips)), 4),
            "first_position_win_rate_when_decisive": round(float(np.mean(decisive_first)), 4) if decisive_first else None,
            "agreement_single_order": round(float(np.mean([h == s for h, s in zip(human, single)])), 4),
            "kappa_single_order": round(cohens_kappa(human, single), 4),
            "agreement_both_orders": round(float(np.mean([h == b for h, b in zip(human, both)])), 4),
            "kappa_both_orders": round(cohens_kappa(human, both), 4),
            "agreement_non_tie_humans_single": round(float(np.mean([h == s for h, s, _ in nt])), 4) if nt else None,
            "agreement_non_tie_humans_both": round(float(np.mean([h == b for h, _, b in nt])), 4) if nt else None,
            "judge_label_counts_both": dict(Counter(both)), "human_label_counts": dict(Counter(human)),
        }

    return {"source": "lmsys/mt_bench_human_judgments (expert votes, first turn)", "naive": stats("naive"), "v2": stats("v2"), "rows": rows}


def exp_retest(cases, temps=(0.7,), samples: int = 3, n_point: int = 16, n_pair: int = 15) -> dict:
    items = [p for p in probes.build(cases) if p.category in ("terse_correct", "confident_wrong", "padded_correct", "verbose_wrong")][:n_point]
    out = {}
    for t in temps:
        cfg = JudgeConfig(name=f"gemini-v2-t{t}", prompt="v2", temperature=t)
        pt = []
        for it in items:
            calls = [judge_pointwise(it, cfg, sample=s, tag=f"retest-t{t}") for s in range(samples)]
            ok = [c for c in calls if c.ok]
            verdicts = [passes(c.verdict) for c in ok]
            scores = [overall(c.verdict) for c in ok]
            pt.append({"id": it.id, "verdicts": verdicts, "flipped": len(set(verdicts)) > 1, "overall_std": float(np.std(scores)) if scores else None})
        pairs = []
        for it in mtbench.sample(n_pair):
            ws = []
            for s in range(samples):
                c = judge_pair(it["case"], it["a"], it["b"], cfg, sample=s, tag=f"retest-t{t}")
                if c.ok:
                    ws.append(c.verdict["winner"])
            pairs.append({"id": it["case"].id, "winners": ws, "flipped": len(set(ws)) > 1})
        out[f"temperature_{t}"] = {
            "samples": samples,
            "pointwise": {"items": len(pt), "verdict_flip_rate": round(float(np.mean([p["flipped"] for p in pt])), 4),
                          "mean_overall_std": round(float(np.mean([p["overall_std"] for p in pt if p["overall_std"] is not None])), 4), "rows": pt},
            "pairwise": {"pairs": len(pairs), "winner_flip_rate": round(float(np.mean([p["flipped"] for p in pairs])), 4), "rows": pairs},
        }
    return out


def exp_selfpref(cases, n: int = 12) -> dict:
    """Self-preference = P(judge of family F picks F's answer) - P(neutral judge picks F's answer)."""
    picked = [c for c in cases if c.category != "safety"][:n]
    llama = GenConfig(name="llama70b", provider="groq", model="llama-3.3-70b-versatile")
    gem = GenConfig(name="gemini", provider="gemini", model="gemini-3.5-flash-lite")
    judges = {
        "gemini": JudgeConfig(name="judge-gemini", provider="gemini", model="gemini-3.5-flash-lite", prompt="v2"),
        "llama": JudgeConfig(name="judge-llama70b", provider="groq", model="llama-3.3-70b-versatile", prompt="v2"),
        "gpt-oss (neutral)": JudgeConfig(name="judge-gptoss", provider="groq", model="openai/gpt-oss-120b", prompt="v2"),
    }
    rows = []
    for c in picked:
        x, y = generate(c, llama), generate(c, gem)  # x = Llama answer (label A), y = Gemini answer (label B)
        rec = {"id": c.id}
        for name, jc in judges.items():
            r = judge_pair_both_orders(c, x.model_output, y.model_output, jc, tag="selfpref")
            rec[name] = r["final"]
        rows.append(rec)

    def share(judge: str, label: str) -> float:
        v = [r[judge] for r in rows if r[judge]]
        return round(sum(w == label for w in v) / len(v), 4) if v else float("nan")

    res = {j: {"llama_wins": share(j, "A"), "gemini_wins": share(j, "B"), "ties": share(j, "tie")} for j in judges}
    res["self_preference"] = {
        "gemini_judge_for_gemini": round(res["gemini"]["gemini_wins"] - res["gpt-oss (neutral)"]["gemini_wins"], 4),
        "llama_judge_for_llama": round(res["llama"]["llama_wins"] - res["gpt-oss (neutral)"]["llama_wins"], 4),
    }
    res["rows"] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["probes", "mtbench", "retest", "selfpref"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bias_validation.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cases = load_suite("suites/core.yaml")
    res["judge"] = {"v2": asdict(V2), "naive": asdict(NAIVE)}
    for name, fn in [("probes", lambda: exp_probes(cases)), ("mtbench", exp_mtbench), ("retest", lambda: exp_retest(cases)),
                     ("selfpref", lambda: exp_selfpref(cases))]:
        if name in a.only:
            print(f"== {name}", flush=True)
            res[name] = fn()
            path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
