"""LLM-as-judge pipeline CLI.

  python -m judge run    --suite suites/core.yaml --gen configs/gen_b.yaml --judge configs/judge_v2.yaml
  python -m judge ab     --suite suites/core.yaml --gen-a configs/gen_a.yaml --gen-b configs/gen_b.yaml --judge configs/judge_v2.yaml
  python -m judge replay --log logs/judge_audit.jsonl

Suites are YAML or JSON: a list of cases, or {"cases": [...]}. Cases that already carry
model_output are judged as-is; otherwise the generator config fills it in first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

from judge.aggregate import compare, overall, suite_report
from judge.generate import GenConfig, generate
from judge.judge import JudgeConfig, judge_pair_both_orders, judge_pointwise
from judge.parse import ParseError, parse_verdict
from judge.schema import NaiveScore, PairVerdict, TestCase, Verdict


def load_suite(path: str) -> list[TestCase]:
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    items = data["cases"] if isinstance(data, dict) else data
    return [TestCase(**c) for c in items]


def run_suite(cases: list[TestCase], judge: JudgeConfig, gen: GenConfig | None) -> tuple[list[TestCase], list[dict]]:
    out_cases, results = [], []
    for c in cases:
        if c.model_output is None:
            if gen is None:
                raise SystemExit(f"case {c.id} has no model_output; pass --gen")
            c = generate(c, gen)
        call = judge_pointwise(c, judge, tag=gen.name if gen else "fixed")
        results.append({"id": c.id, "category": c.category, "model_output": c.model_output, "call": asdict(call),
                        "overall": round(overall(call.verdict), 3) if call.ok and judge.prompt == "v2" else None})
        out_cases.append(c)
        print(f"  {c.id:<12} {'ok' if call.ok else call.error:<10} overall={results[-1]['overall']}", file=sys.stderr)
    return out_cases, results


def cmd_run(a) -> dict:
    judge = JudgeConfig.load(a.judge)
    gen = GenConfig.load(a.gen) if a.gen else None
    _, results = run_suite(load_suite(a.suite), judge, gen)
    report = {"judge": asdict(judge), "generator": asdict(gen) if gen else None, "report": suite_report(results), "cases": results}
    return report


def cmd_ab(a) -> dict:
    judge = JudgeConfig.load(a.judge)
    ga, gb = GenConfig.load(a.gen_a), GenConfig.load(a.gen_b)
    if (ga.provider, ga.model.split("-")[0]) == (judge.provider, judge.model.split("-")[0]):
        print("warning: judge and generator A share a model family (self-enhancement risk)", file=sys.stderr)
    suite = load_suite(a.suite)
    print(f"config A = {ga.name}", file=sys.stderr)
    ca, ra = run_suite(suite, judge, ga)
    print(f"config B = {gb.name}", file=sys.stderr)
    cb, rb = run_suite(suite, judge, gb)
    pairs = []
    for x, y in zip(ca, cb):
        r = judge_pair_both_orders(x, x.model_output, y.model_output, judge, tag="ab")
        pairs.append({"id": x.id, **{k: r[k] for k in ("order_xy", "order_yx_mapped", "consistent", "flipped", "final")}})
    rep_a, rep_b = suite_report(ra), suite_report(rb)
    both = [(p, q) for p, q in zip(ra, rb) if p["overall"] is not None and q["overall"] is not None]
    cmp = compare(rep_a, rep_b, [p["final"] for p in pairs], [p["overall"] for p, _ in both], [q["overall"] for _, q in both])
    flips = [p["flipped"] for p in pairs if p["order_xy"] and p["order_yx_mapped"]]
    cmp["position_flip_rate"] = round(sum(flips) / len(flips), 4) if flips else None
    return {"judge": asdict(judge), "config_a": asdict(ga), "config_b": asdict(gb), "report_a": rep_a, "report_b": rep_b,
            "comparison": cmp, "pairs": pairs, "cases_a": ra, "cases_b": rb}


def cmd_replay(a) -> dict:
    """Re-parse every logged judge response offline: proves the log alone reproduces the verdicts."""
    stats, rungs = Counter(), Counter()
    for line in Path(a.log).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        kind = rec["tag"].split("|")[0]
        schema = {"pointwise_v2": Verdict, "pointwise_naive": NaiveScore}.get(kind, PairVerdict)
        try:
            _, rung = parse_verdict(rec["response_text"], schema)
            stats["parsed"] += 1
            rungs[rung] += 1
        except ParseError:
            stats["unparseable"] += 1
    return {"log": a.log, **stats, "rungs": dict(rungs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--judge", default="configs/judge_v2.yaml")
    r.add_argument("--gen")
    r.add_argument("--out", default="results/p2/run.json")
    ab = sub.add_parser("ab")
    ab.add_argument("--suite", required=True)
    ab.add_argument("--judge", default="configs/judge_v2.yaml")
    ab.add_argument("--gen-a", required=True)
    ab.add_argument("--gen-b", required=True)
    ab.add_argument("--out", default="results/p2/ab.json")
    rp = sub.add_parser("replay")
    rp.add_argument("--log", default="logs/judge_audit.jsonl")
    rp.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"run": cmd_run, "ab": cmd_ab, "replay": cmd_replay}[a.cmd](a)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = res.get("comparison") or res.get("report") or res
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
