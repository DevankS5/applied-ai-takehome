"""Answer-quality evaluation for the RAG service (needs GROQ_API_KEY).

    python -m rag_eval.run_answers

Generator: Llama 3.3 70B (Meta). Judge: gpt-oss-120b (OpenAI family), so the judge is not
grading its own family's writing. One judge call per answer returns:
  - the answer split into atomic claims, each marked supported / not supported by the passages
    the generator was given  -> faithfulness = supported / total claims (RAGAS definition)
  - a 1-5 relevance score: does the answer address the question  -> normalised to 0-1
Gold answers give EM, a contains-match variant (generative answers are sentences) and token F1.
Abstention is scored on both sides: answerable questions should be answered, the 5
unanswerable ones should be declined.
Writes results/p1/answers.json and results/p1/answers.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("RAG_LLM_CACHE_DIR", ".cache/llm")  # eval reruns are free and identical

from common.llm import chat  # noqa: E402
from judge.parse import ParseError, parse_json  # noqa: E402
from rag.generate import format_context  # noqa: E402
from rag.service import answer  # noqa: E402
from rag_eval import metrics as M  # noqa: E402
from rag_eval.run_retrieval import load_questions  # noqa: E402

OUT = Path("results/p1")
JUDGE = ("groq", "openai/gpt-oss-120b")
AUDIT = "logs/rag_judge_audit.jsonl"

JUDGE_SYSTEM = """You evaluate a retrieval-augmented answer. Be strict and literal.
1. Split the ANSWER into atomic factual claims (ignore citation markers like [1]).
2. For each claim decide if it is directly supported by the PASSAGES (not by your own knowledge).
3. Rate RELEVANCE 1-5: 5 = directly and fully answers the question; 3 = partly answers or adds unrequested material;
   1 = does not address the question.
Return ONLY JSON: {"claims": [{"claim": "...", "supported": true, "passage": 1}], "relevance": 5, "relevance_reason": "..."}"""


def judge_answer(question: str, ans: str, rows: list[dict]) -> dict:
    user = f"QUESTION: {question}\n\nPASSAGES:\n{format_context(rows)}\n\nANSWER:\n{ans}"
    res = chat(*JUDGE, user, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=1500, json_mode=True,
               cache_dir=".cache/llm", audit_path=AUDIT, tag="rag_answer_judge")
    try:
        obj, _ = parse_json(res.text)
    except ParseError as e:
        return {"ok": False, "error": str(e), "tokens": res.prompt_tokens + res.completion_tokens}
    claims = obj.get("claims") or []
    sup = sum(bool(c.get("supported")) for c in claims)
    return {"ok": True, "claims": claims, "n_claims": len(claims), "supported": sup,
            "faithfulness": sup / len(claims) if claims else None, "relevance": obj.get("relevance"),
            "relevance_reason": obj.get("relevance_reason"), "tokens": res.prompt_tokens + res.completion_tokens}


def main() -> None:
    qs = load_questions()
    rows = []
    for q in qs:
        out = answer(q["question"], k=5)
        rec = {"id": q["id"], "answerable": bool(q["evidence"]), "question": q["question"], "gold": q["answer"],
               "answer": out["answer"], "abstained": out["abstained"], "abstain_reason": out["abstain_reason"],
               "citations": out["citations"], "invalid_citations": out["invalid_citations"], "top_score": out["top_score"],
               "latency_ms": out["latency_ms"], "tokens": out["tokens"]}
        # End-to-end = embed + retrieve + the generator's measured API latency (kept even when replayed from cache).
        rec["e2e_ms"] = round(out["latency_ms"]["embed"] + out["latency_ms"]["retrieve"] + out["latency_ms"]["generate"], 1)
        if q["evidence"] and not out["abstained"]:
            rec["em"] = M.exact_match(out["answer"], q["answer"])
            rec["contains"] = M.contains_match(out["answer"], q["answer"])
            rec["f1"] = round(M.token_f1(out["answer"], q["answer"]), 4)
            rec["judge"] = judge_answer(q["question"], out["answer"], [{**s} for s in out["sources"]])
        elif not q["evidence"] and not out["abstained"]:
            # Answered an unanswerable question: still measure whether the answer stuck to the passages.
            rec["judge"] = judge_answer(q["question"], out["answer"], [{**s} for s in out["sources"]])
        rows.append(rec)
        print(f"{q['id']} abstained={out['abstained']!s:<5} {out['answer'][:90]!r}", flush=True)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    judged = [r for r in answered if r.get("judge", {}).get("ok")]
    faith = [r["judge"]["faithfulness"] for r in judged if r["judge"]["faithfulness"] is not None]
    rel = [(r["judge"]["relevance"] - 1) / 4 for r in judged if isinstance(r["judge"].get("relevance"), (int, float))]
    e2e = [r["e2e_ms"] for r in rows if r["latency_ms"]["generate"] > 0]
    summary = {
        "generator": "groq/llama-3.3-70b-versatile", "judge": "/".join(JUDGE), "k": 5,
        "answerable": len(ans), "answered": len(answered), "false_abstentions": len(ans) - len(answered),
        "unanswerable": len(una), "unanswerable_declined": sum(r["abstained"] for r in una),
        "unanswerable_declined_by_guard": sum(r["abstain_reason"] == "below_threshold" for r in una),
        "unanswerable_declined_by_model": sum(r["abstain_reason"] == "model_abstained" for r in una),
        "faithfulness_mean": round(float(np.mean(faith)), 4) if faith else None,
        "faithfulness_all_claims_supported": round(float(np.mean([f == 1.0 for f in faith])), 4) if faith else None,
        "answer_relevance_mean": round(float(np.mean(rel)), 4) if rel else None,
        "em": round(float(np.mean([r["em"] for r in answered])), 4) if answered else None,
        "contains_em": round(float(np.mean([r["contains"] for r in answered])), 4) if answered else None,
        "f1": round(float(np.mean([r["f1"] for r in answered])), 4) if answered else None,
        "answers_with_valid_citation": round(float(np.mean([bool(r["citations"]) for r in answered])), 4) if answered else None,
        "invalid_citations_total": sum(len(r["invalid_citations"]) for r in rows),
        "e2e_p50_ms": round(M.pct(e2e, 50), 1), "e2e_p95_ms": round(M.pct(e2e, 95), 1),
        "generator_tokens": {"prompt": sum(r["tokens"]["prompt"] for r in rows), "completion": sum(r["tokens"]["completion"] for r in rows)},
        "judge_tokens": sum(r.get("judge", {}).get("tokens", 0) for r in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    s = summary
    md = ["# Problem 1: answer quality", "", f"Generator `{s['generator']}`, judge `{s['judge']}` (different model family), k = 5, hybrid retrieval.", "",
          "| Metric | Value | Method |", "|---|---|---|",
          f"| Faithfulness | {s['faithfulness_mean']} | claim-level: supported claims / all claims (LLM judge), {len(faith)} answers |",
          f"| Answers with every claim supported | {s['faithfulness_all_claims_supported']} | same judge |",
          f"| Answer relevance | {s['answer_relevance_mean']} | judge 1-5 rescaled to 0-1 |",
          f"| Exact match | {s['em']} | SQuAD normalisation vs gold short answer |",
          f"| Contains-match | {s['contains_em']} | normalised gold answer appears in the response |",
          f"| Token F1 | {s['f1']} | SQuAD token F1 vs gold |",
          f"| Answered (answerable) | {s['answered']}/{s['answerable']} | false abstentions: {s['false_abstentions']} |",
          f"| Declined (unanswerable) | {s['unanswerable_declined']}/{s['unanswerable']} | guard: {s['unanswerable_declined_by_guard']}, model: {s['unanswerable_declined_by_model']} |",
          f"| Answers citing a valid passage | {s['answers_with_valid_citation']} | invalid citation markers: {s['invalid_citations_total']} |",
          f"| End-to-end p50 / p95 | {s['e2e_p50_ms']} / {s['e2e_p95_ms']} ms | embed + retrieve + generation API latency |"]
    (OUT / "answers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
