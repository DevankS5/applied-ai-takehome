"""Print raw evidence straight from the logs and result files (used for the submission screenshots).

    python scripts/evidence.py audit    --tag "pointwise_v2|fact-01|A"   # one raw judge prompt + raw response
    python scripts/evidence.py position --id fact-03                     # one pair judged in both orders + flip rates
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
    p = next(x for x in ab["pairs"] if x["id"] == a.id)
    print(f"pair {p['id']}: X = {ab['config_a']['name']}, Y = {ab['config_b']['name']}  (judge {ab['judge']['model']})")
    print(f"  order 1 (X shown as A, Y as B) -> winner {p['order_xy']}")
    print(f"  order 2 (Y shown as A, X as B) -> winner {p['order_yx_mapped']} (mapped back to X/Y labels)")
    print(f"  consistent={p['consistent']}  flipped={p['flipped']}  final verdict={p['final']}  (disagreement -> tie)\n")
    print(f"A/B suite, {len(ab['pairs'])} pairs: position flip rate = {ab['comparison']['position_flip_rate']}")
    bias = json.loads(Path(a.bias).read_text(encoding="utf-8")) if Path(a.bias).exists() else {}
    for prompt in ("naive", "v2"):
        s = bias.get("mtbench", {}).get(prompt)
        if s:
            print(f"MT-Bench human-labelled pairs, {prompt} judge prompt: flip rate = {s['flip_rate']} over {s['pairs']} pairs")


def probe(a) -> None:
    bias = json.loads(Path(a.bias).read_text(encoding="utf-8"))
    r = next(x for x in bias["probes"]["rows"] if x["kind"] == a.kind and x["id"].startswith(a.id))
    print(f"probe {r['id']}  kind={r['kind']}  gold verdict={r['gold']}")
    print(f"  naive judge (bare 1-10 score):  score={r['naive_score']}  -> {'PASS' if r['naive_pass'] else 'FAIL'}")
    print(f"  v2 judge (rubric + evidence):   per-criterion={r['v2_scores']}  overall={r['v2_overall']}  -> {'PASS' if r['v2_pass'] else 'FAIL'}")
    print("\n=== raw v2 judge response ===\n" + (r["v2_raw"] or ""))


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
    s.add_argument("--id", required=True)
    s.add_argument("--ab", default="results/p2/ab.json")
    s.add_argument("--bias", default="results/p2/bias_validation.json")
    s = sub.add_parser("probe")
    s.add_argument("--kind", default="verbose_wrong")
    s.add_argument("--id", required=True)
    s.add_argument("--bias", default="results/p2/bias_validation.json")
    a = ap.parse_args()
    {"audit": audit, "position": position, "probe": probe, "response": response}[a.cmd](a)


if __name__ == "__main__":
    main()
