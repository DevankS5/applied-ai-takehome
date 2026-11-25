"""Per-case scores -> suite report; two configs -> a declared winner (or an honest 'no difference')."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from judge.schema import criteria_names, rubric


def overall(verdict: dict) -> float:
    """Weighted mean of the scored criteria (weights renormalised over what was scored)."""
    w = {n: c["weight"] for n, c in rubric()["criteria"].items()}
    scored = verdict["criteria"]
    total = sum(w[n] for n in scored)
    return round(sum(w[n] * scored[n]["score"] for n in scored) / total, 6)  # 0.3+0.2+... is not exactly 1.0 in floats


def passes(verdict: dict) -> bool:
    g, c = rubric()["gates"], verdict["criteria"]
    if overall(verdict) < g["pass_overall"]:
        return False
    if "correctness" in c and c["correctness"]["score"] < g["min_correctness"]:
        return False
    if "tone_safety" in c and c["tone_safety"]["score"] < g["min_tone_safety"]:
        return False
    return True


def bootstrap_ci(values: list[float], stat=np.mean, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [stat(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial test of wins vs losses (ties dropped), H0: p = 0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[label] * cb[label] for label in labels) / n**2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score_spread(scores: list[float], scale: int) -> dict:
    """How much of the scale a judge uses: std, distinct values, normalised entropy."""
    c = Counter(round(s) for s in scores)
    p = np.array(list(c.values()), dtype=float) / len(scores)
    return {"std": round(float(np.std(scores)), 3), "distinct": len(c),
            "entropy_norm": round(float(-(p * np.log(p)).sum() / math.log(scale)), 3), "histogram": dict(sorted(c.items()))}


def suite_report(results: list[dict]) -> dict:
    """results: [{"id", "category", "call": Call-as-dict}] from pointwise v2 judging."""
    ok = [r for r in results if r["call"]["ok"]]
    overalls = [overall(r["call"]["verdict"]) for r in ok]
    passed = [1.0 if passes(r["call"]["verdict"]) else 0.0 for r in ok]
    per_crit = {}
    for n in criteria_names():
        vals = [r["call"]["verdict"]["criteria"][n]["score"] for r in ok if n in r["call"]["verdict"]["criteria"]]
        if vals:
            per_crit[n] = round(float(np.mean(vals)), 3)
    by_cat = {}
    for r, p in zip(ok, passed):
        by_cat.setdefault(r.get("category", "general"), []).append(p)
    calls = [r["call"] for r in results]
    return {
        "cases": len(results), "parsed": len(ok), "parse_failed": len(results) - len(ok),
        "reasked": sum(c["reasked"] for c in calls),
        "parse_rungs": dict(Counter(c["parse_rung"] for c in calls if c["ok"])),
        "pass_rate": round(float(np.mean(passed)), 4) if passed else None,
        "pass_rate_ci95": [round(x, 4) for x in bootstrap_ci(passed)],
        "mean_overall": round(float(np.mean(overalls)), 3) if overalls else None,
        "mean_overall_ci95": [round(x, 3) for x in bootstrap_ci(overalls)],
        "mean_by_criterion": per_crit,
        "pass_rate_by_category": {k: round(float(np.mean(v)), 3) for k, v in by_cat.items()},
        "spread": score_spread(overalls, 5) if overalls else None,
        "judge_calls": sum(c["calls"] for c in calls),
        "judge_tokens": {"prompt": sum(c["prompt_tokens"] for c in calls), "completion": sum(c["completion_tokens"] for c in calls)},
        "judge_cost_usd_list_price": round(sum(c["cost_usd"] for c in calls), 5),
    }


def compare(a_report: dict, b_report: dict, pair_finals: list[str | None], a_overalls: list[float], b_overalls: list[float],
            alpha: float = 0.05) -> dict:
    """pair_finals: per case, both-order pairwise result with A = config A ('A' | 'B' | 'tie' | None).
    Winner is declared only if the sign test on decisive pairs is significant; otherwise 'no significant difference'."""
    d = [f for f in pair_finals if f]
    wins, losses, ties = d.count("A"), d.count("B"), d.count("tie")
    p = sign_test_p(wins, losses)
    win_rate_a = (wins + 0.5 * ties) / len(d) if d else float("nan")
    diffs = [x - y for x, y in zip(a_overalls, b_overalls)]
    ci = bootstrap_ci(diffs)
    if p < alpha:
        winner = "A" if wins > losses else "B"
        why = f"pairwise sign test p = {p:.4f} < {alpha}"
    else:
        winner = "no significant difference"
        why = f"pairwise sign test p = {p:.4f} >= {alpha}"
    return {
        "pairwise": {"decided": len(d), "a_wins": wins, "b_wins": losses, "ties": ties, "win_rate_a": round(win_rate_a, 4),
                     "win_rate_b": round(1 - win_rate_a, 4) if d else None, "sign_test_p": round(p, 5)},
        "pointwise": {"pass_rate_a": a_report["pass_rate"], "pass_rate_b": b_report["pass_rate"],
                      "mean_overall_a": a_report["mean_overall"], "mean_overall_b": b_report["mean_overall"],
                      "mean_diff_a_minus_b": round(float(np.mean(diffs)), 3) if diffs else None,
                      "mean_diff_ci95": [round(x, 3) for x in ci]},
        "winner": winner, "reason": why,
    }
