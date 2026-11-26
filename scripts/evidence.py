"""Print raw evidence straight from the logs and result files (used for the submission screenshots).

    python scripts/evidence.py audit    --tag "pointwise_v2|fact-01|A"   # one raw judge prompt + raw response
    python scripts/evidence.py position --id fact-06 safety-04           # pairs judged in both orders + flip rates
    python scripts/evidence.py probe    --kind verbose_wrong --id fact-01 # adversarial probe verdicts, naive vs v2
    curl -s localhost:8000/query -d @docs/examples/query.json -H "content-type: application/json" | python scripts/evidence.py response
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def audit(a) -> None:
    for line in Path(a.log).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["tag"].startswith(a.tag):
            print(f"# logs/judge_audit.jsonl  tag={r['tag']}  model={r['provider']}/{r['model']}  temperature={r['temperature']}")
            print(f"# request_hash={r['request_hash'][:16]}…  tokens in/out={r['prompt_tokens']}/{r['completion_tokens']}  latency={r['latency_ms']} ms\n")
            print("=== SYSTEM PROMPT (raw) ===\n" + r["system"] + "\n")
            print("=== USER PROMPT (raw) ===\n" + r["user"] + "\n")
            print("=== RAW JUDGE RESPONSE ===\n" + r["response_text"])
            return
    raise SystemExit(f"no record with tag prefix {a.tag!r}")


def position(a) -> None:
    ab = json.loads(Path(a.ab).read_text(encoding="utf-8"))
    print(f"X = {ab['config_a']['name']}, Y = {ab['config_b']['name']}, judge {ab['judge']['model']}; "
          "a verdict counts only if both orders agree, otherwise it is a tie\n")
    for pid in a.id:
        p = next(x for x in ab["pairs"] if x["id"] == pid)
        print(f"pair {p['id']}")
        print(f"  order 1 (X shown as A, Y as B) -> winner {p['order_xy']}")
        print(f"  order 2 (Y shown as A, X as B) -> winner {p['order_yx_mapped']} (mapped back to X/Y labels)")
        print(f"  consistent={p['consistent']}  flipped={p['flipped']}  final verdict={p['final']}\n")
    print(f"A/B suite, {len(ab['pairs'])} pairs: position flip rate = {ab['comparison']['position_flip_rate']}")
    bias = json.loads(Path(a.bias).read_text(encoding="utf-8")) if Path(a.bias).exists() else {}
    for prompt in ("naive", "v2"):
        s = bias.get("mtbench", {}).get(prompt)
        if s:
            print(f"MT-Bench human-labelled pairs, {prompt} judge prompt: flip rate = {s['flip_rate']} over {s['pairs']} pairs")


def probe(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from judge.__main__ import load_suite
    from judge_eval import probes

    bias = json.loads(Path(a.bias).read_text(encoding="utf-8"))
    items = {p.id: p for p in probes.build(load_suite("suites/core.yaml"))}
    for kind in a.kind:
        r = next(x for x in bias["probes"]["rows"] if x["kind"] == kind and x["id"].startswith(a.id))
        it = items[r["id"]]
        print(f"probe {r['id']}  gold verdict = {r['gold'].upper()}  (label fixed by construction)")
        print(f"  system prompt: {it.system_prompt}   reference: {it.expected_output}")
        print(f"  answer judged: {it.model_output[:200]}{'…' if len(it.model_output) > 200 else ''}")
        print(f"  naive judge (bare 1-10 score):  score={r['naive_score']}  -> {'PASS' if r['naive_pass'] else 'FAIL'}")
        print(f"  v2 judge (rubric + evidence):   {r['v2_scores']}  overall={r['v2_overall']}"
              f"  -> {'PASS' if r['v2_pass'] else 'FAIL'}\n")
    if a.raw:
        print("=== raw v2 judge response for the first probe ===\n" + (next(
            x for x in bias["probes"]["rows"] if x["kind"] == a.kind[0] and x["id"].startswith(a.id))["v2_raw"] or ""))


def response(a) -> None:
    """Compact view of a /query response read from stdin (passage texts shortened to one line each)."""
    r = json.load(sys.stdin)
    for s in r.get("sources", []):
        s["text"] = s["text"][:44].replace("\n", " ") + "…"
    keep = ["answer", "abstained", "citations", "invalid_citations", "top_score", "chunks_retrieved", "chunks_cited",
            "latency_ms", "tokens", "model", "k", "mode", "filters"]
    for k in keep:
        if k in r:
            print(f"{k:>18}: {json.dumps(r[k], ensure_ascii=False)}")
    print(f"{'sources':>18}: (passage text cut to 44 chars here; full text is in the response)")
    for i, s in enumerate(r.get("sources", []), 1):
        print(f"{'':>18}  [{i}] cosine={s.get('score')} {s.get('doc_id')} / {s.get('heading') or ''} | {s['text']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("response")
    s = sub.add_parser("audit")
    s.add_argument("--tag", required=True)
    s.add_argument("--log", default="logs/judge_audit.jsonl")
    s = sub.add_parser("position")
    s.add_argument("--id", nargs="+", required=True)
    s.add_argument("--ab", default="results/p2/ab.json")
    s.add_argument("--bias", default="results/p2/bias_validation.json")
    s = sub.add_parser("probe")
    s.add_argument("--kind", nargs="+", default=["verbose_wrong"])
    s.add_argument("--raw", action="store_true")
    s.add_argument("--id", required=True)
    s.add_argument("--bias", default="results/p2/bias_validation.json")
    a = ap.parse_args()
    {"audit": audit, "position": position, "probe": probe, "response": response}[a.cmd](a)


if __name__ == "__main__":
    main()
