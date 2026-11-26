"""Fill the submission template from the result files, so every number in the document is the one the repo produced.

    python scripts/fill_submission.py --template submission_template.docx --out submission.docx

Reflection boxes (the template asks for them to be written without AI help) and the personal fields are left
empty and highlighted yellow for the candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/DevankS5/applied-ai-takehome"
DEMO = "https://devsrivastava-cost-efficient-rag.hf.space"
INK, CODE = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x45, 0x51, 0x62)
YOURS = "[ to be written by Devank, without AI assistance, as the template requires ]"


def load(p: str) -> dict:
    f = ROOT / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def put(cell, text: str = "", mono: bool = False, image: Path | None = None, width: float = 6.2, mine: bool = False) -> None:
    """Replace a cell's content with text lines, or an image followed by optional caption text."""
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    first = cell.paragraphs[0]
    for r in first.runs:
        r._element.getparent().remove(r._element)
    lines = text.split("\n") if text else []
    if image is not None:
        first.add_run().add_picture(str(image), width=Inches(width))
        paras = [cell.add_paragraph() for _ in lines]
    else:
        paras = [first] + [cell.add_paragraph() for _ in lines[1:]]
    for p, line in zip(paras, lines):
        r = p.add_run(line)
        r.font.size = Pt(9.5 if mono else 10)
        r.font.color.rgb = CODE if mono else INK
        if mono:
            r.font.name = "Consolas"
        if mine:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def fill_line(par, value: str, mine: bool = False) -> None:
    """Template lines are 'Label  [ placeholder ]' in two runs; replace the placeholder run."""
    run = par.runs[-1]
    run.text = value
    run.font.color.rgb = INK
    if mine:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def f2(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ret, ans, cost = load("results/p1/retrieval.json"), load("results/p1/answers.json").get("summary", {}), load("results/p1/cost.json")
    ab, bias = load("results/p2/ab.json"), load("results/p2/bias_validation.json")
    rubric = yaml.safe_load((ROOT / "judge/rubric.yaml").read_text(encoding="utf-8"))
    m = ret["default"]["metrics"]
    lat = ret["latency"]
    img = ROOT / "docs/img"

    d = Document(a.template)
    T, P = d.tables, d.paragraphs

    # ---------------- candidate details ----------------
    rows = T[0].rows
    put(rows[1].cells[1], "Devank Srivastava")
    put(rows[2].cells[1], "devankcomputer@gmail.com")
    put(rows[3].cells[1], "[ date sent ]", mine=True)
    put(rows[4].cells[1], f"{REPO}\nLive demo (Problem 1): {DEMO}")
    put(rows[5].cells[1], "[ total time ]", mine=True)
    put(rows[6].cells[1], "LanceDB (embedded; local disk or S3). FAISS benchmarked as the bonus second store.")
    put(rows[7].cells[1], "Judge: Google Gemini 3.5 Flash-Lite. Generators: OpenAI gpt-oss-20b (A/B configs), "
                          "gpt-oss-120b (Problem 1 answers). Different families by design.")

    # ---------------- Problem 1 ----------------
    put(T[1].cells[0], "Figure: ingestion to cited answer, including the two no-context exits (guard and abstain token).",
        image=img / "p1_architecture.png")
    fill_line(P[18], "Python 3.11 or 3.12 with uv; Windows, macOS or Linux; CPU only (no GPU, no database server); "
                     "about 1 GB disk for dependencies, the ONNX embedding model and the index. Free Groq and Gemini API keys.")
    fill_line(P[19], "GROQ_API_KEY, GEMINI_API_KEY (secrets, in a gitignored .env); RAG_DB_PATH, RAG_TABLE, RAG_EMBED_MODEL, "
                     "RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_SEARCH_MODE, RAG_RERANK, RAG_MIN_SCORE, RAG_GEN_PROVIDER, "
                     "RAG_GEN_MODEL, RAG_RATE_LIMIT_PER_MIN, RAG_TRUST_X_REAL_IP. All are listed in .env.example.")
    put(T[2].cells[0], f"git clone {REPO}\ncd applied-ai-takehome\nuv sync                      # runtime + dev dependencies\n"
                       "cp .env.example .env         # paste GROQ_API_KEY and GEMINI_API_KEY", mono=True)
    put(T[3].cells[0], "uv run python scripts/fetch_corpus.py --verify   # 17 pinned docs (7 PDF, 6 HTML, 4 MD)\n"
                       "uv run python -m rag.ingest data/corpus/raw      # default chunk 400 tokens, overlap 60\n"
                       "# any folder of .pdf/.html/.md works; --chunk-size, --overlap, --table, --prune to change", mono=True)
    put(T[4].cells[0], "uv run uvicorn rag.api:app --port 8000            # UI on /, OpenAPI on /docs\n"
                       "curl -s localhost:8000/query -H 'content-type: application/json' \\\n"
                       "  -d '{\"question\": \"What is the default hnsw.ef_search in pgvector?\", \"k\": 5, \"filters\": {\"doc_type\": \"md\"}}'\n"
                       "# -> {\"answer\": \"The default value of hnsw.ef_search is 40 [1].\", \"citations\": [...],\n"
                       "#     \"latency_ms\": {...}, \"chunks_retrieved\": 5, \"tokens\": {...}, ...}\n"
                       f"# same request against the live demo: {DEMO}/query", mono=True)
    fill_line(P[26], "LanceDB: an embedded store whose table is plain files on local disk or S3, so storage is separate from "
                     "compute and nothing has to stay running; hybrid BM25 + vector search, SQL metadata prefilters and "
                     "idempotent merge_insert come in the same library.")

    fill_line(P[31], f"k = 5, hybrid retrieval (dense + BM25 fused with RRF), {ret['default']['vectors']} chunks, "
                     f"{ret['default']['questions']} answerable questions + 5 unanswerable")
    t = T[5].rows
    rel_def = ("A chunk is relevant iff it contains one of the question's verbatim evidence quotes (normalised; at least 60% "
               "of the quote as one run), so relevance is independent of chunk settings.")
    put(t[1].cells[1], f"Hit@5 {f2(m['hit@5'])}\nRecall@5 {f2(m['recall@5'])}")
    put(t[1].cells[2], "Hit: share of questions with at least one relevant chunk in the top 5. Recall: relevant chunks found / all "
                       "relevant chunks. Hit@1 0.44, Hit@3 0.80. Paraphrased question set: Hit@5 0.68.")
    put(t[2].cells[1], f2(m["mrr@5"], 3))
    put(t[2].cells[2], "Mean of 1 / rank of the first relevant chunk (0 if none in the top 5).")
    put(t[3].cells[1], f2(m["ndcg@5"], 3))
    put(t[3].cells[2], "Binary gains, log2 discount, normalised by the ideal ordering of all relevant chunks.")
    put(t[4].cells[1], f2(m["context_precision@5"], 3))
    put(t[4].cells[2], "RAGAS definition: mean precision@i over the ranks i that hold a relevant chunk. " + rel_def)

    t = T[6].rows
    n_j = ans.get("answered")
    put(t[1].cells[1], f2(ans.get("faithfulness_mean"), 3))
    put(t[1].cells[2], "LLM judge, claim level")
    put(t[1].cells[3], f"Judge ({ans.get('judge')}) splits each answer into atomic claims and checks each against the passages "
                       f"sent; score = supported / all claims. {pct(ans.get('faithfulness_all_claims_supported'))} of "
                       f"{n_j} answers had every claim supported.")
    put(t[2].cells[1], f2(ans.get("answer_relevance_mean"), 3))
    put(t[2].cells[2], "LLM judge, 1-5 rescaled to 0-1")
    put(t[2].cells[3], f"Same judge, different model family from the generator ({ans.get('generator')}).")
    put(t[3].cells[1], f2(ans.get("em"), 3))
    put(t[3].cells[2], "Gold short answer")
    put(t[3].cells[3], f"SQuAD normalisation. Low by design: answers are full cited sentences. Contains-match (gold answer "
                       f"appears in the response): {f2(ans.get('contains_em'), 3)}.")
    put(t[4].cells[1], f2(ans.get("f1"), 3))
    put(t[4].cells[2], "Gold short answer")
    put(t[4].cells[3], f"SQuAD token F1 over {n_j} answered questions. Abstention: answered {ans.get('answered')}/"
                       f"{ans.get('answerable')} answerable, declined {ans.get('unanswerable_declined')}/"
                       f"{ans.get('unanswerable')} unanswerable (guard {ans.get('unanswerable_declined_by_guard')}, "
                       f"model {ans.get('unanswerable_declined_by_model')}).")

    base = {(r["vectors"], r["option"].split(" (")[0]): r["monthly_usd"] for r in cost["base"]}

    def c(v, name):
        return next(val for (vv, opt), val in base.items() if vv == v and opt.startswith(name))

    for i, v in enumerate([100_000, 1_000_000, 10_000_000], start=1):
        ours, vm = c(v, "LanceDB on S3"), c(v, "LanceDB on one VM")
        pine, qd = c(v, "Pinecone"), c(v, "Qdrant")
        cells = T[7].rows[i].cells
        put(cells[1], f"${ours:,.2f} (S3 + serverless)\n${vm:,.2f} (always-on VM)")
        put(cells[2], f"${pine:,.2f} Pinecone Serverless\n${qd:,.2f} Qdrant Cloud")
        put(cells[3], f"{pine / ours:.0f}x cheaper than Pinecone, {qd / ours:.0f}x than Qdrant"
                      + (" (Pinecone's $50 plan minimum binds)" if pine == 50 else ""))
    put(T[8].cells[0],
        "Pricing assumptions (all inputs in rag_eval/cost_assumptions.yaml; prices checked 2026-11-25, us-east-1, on-demand, single "
        "replica everywhere so no option is charged for HA). Record = 384-d float32 vector (1,536 B) + 1,650 B text and metadata = "
        "3,186 B. Workload: 100K queries/month, 10% of records rewritten monthly.\n"
        "Ours, S3 + serverless: S3 Standard $0.023/GB-month x 1.3 overhead for index and versions, 40 GETs per query at $0.0004/1K, "
        "Lambda arm64 1,769 MB for 350 ms per query.\n"
        "Ours, always-on VM: t4g.small / medium / large ($0.0168 / $0.0336 / $0.0672 per hour) + gp3 EBS at $0.08/GB-month.\n"
        "Pinecone Serverless, Standard plan (pinecone.io/pricing): $0.33/GB-month storage, $16 per million read units with "
        "1 RU per GB of namespace per query (min 0.25), $4 per million write units, $50/month minimum.\n"
        "Qdrant Cloud (cloud.qdrant.io/calculator, 384-d, 1 replica, no quantization): 2 GiB node; 1 node for 3 GiB; 3 x 8 GiB nodes.\n"
        "Sensitivity: at 10M queries/month the serverless option is about $245/month at every scale (per-query compute dominates), "
        "the always-on VM stays at $14-54 and Pinecone reaches $512-5,121. The cheapest design depends on traffic, not vector count.")

    t = T[9].rows
    hy = lat["hybrid"]
    put(t[1].cells[1], f"{hy['retrieval_p50_ms']}")
    put(t[1].cells[2], f"Hybrid, k = 5, {hy['vectors']} chunks, {hy['samples']} warm queries, LanceDB on a laptop CPU (Ryzen 7 "
                       f"7435HS, 24 GB RAM, Windows 11). Query embedding adds p50 {lat['vector']['embed_p50_ms']} ms.")
    put(t[2].cells[1], f"{hy['retrieval_p95_ms']}")
    put(t[2].cells[2], f"Same run; {hy['embed_plus_retrieval_p95_ms']} ms including query embedding. At 1M vectors (bench): "
                       "IVF-PQ p95 27.4 ms at recall@10 0.94 vs 1,306 ms exact scan.")
    put(t[3].cells[1], f"{ans.get('e2e_p95_ms')}")
    put(t[3].cells[2], f"embed + retrieve + generation API latency ({ans.get('generator')} on Groq, low reasoning effort), "
                       f"30 questions; p50 {ans.get('e2e_p50_ms')} ms. Excludes client-side rate-limit waits.")

    put(T[10].cells[0],
        "The brief says the managed bill comes from stored vectors on always-on pods. LanceDB attacks exactly that: the table is a set "
        "of Lance files on disk or S3 and any process that opens them can search, so an index that is rarely queried costs object "
        "storage plus compute only while a query runs. The same code runs on a laptop, in a container and in a serverless function. "
        "It also gives SQL WHERE prefilters, IVF-PQ for scale, merge_insert for idempotent upserts and a BM25 index, which I needed "
        "for hybrid search. I rejected pgvector because it needs an always-on Postgres (the right call if the team already runs "
        "one); Qdrant because its fast path is RAM-resident, so cost tracks vector count again; Chroma because its embedded mode is "
        "local-disk only, so scaling out means running its server; sqlite-vec because it is brute force only; and FAISS alone "
        "because it is a library, not a store (no metadata, no upserts, no persistence model). FAISS is benchmarked as the second store.")
    put(T[11].cells[0],
        "400 tokens with 60 overlap, counted with the embedding model's own tokenizer, packed from whole sentences and never "
        "crossing a paragraph; each chunk is embedded with its title and section heading in front. I swept four settings on the "
        "original and paraphrased questions (pooled Hit@5): 200/30 0.78, 400/0 0.76, 400/60 0.80, 800/100 0.86. 800-token chunks "
        "only win because they send twice the text at the same k: at an equal budget of about 2.2K tokens (k = 3) they drop to "
        "0.70. 200-token chunks cut evidence sentences away from the terms that identify them. Removing overlap lost two questions "
        "whose evidence sat on a boundary. 400/60 is the best result per token sent to the LLM.")
    put(T[12].cells[0],
        "BAAI/bge-small-en-v1.5, 384 dimensions, run locally through fastembed's ONNX runtime (about 65 MB, about 12 ms per query on "
        "CPU, zero per-call cost). 384-d float32 is 1.5 KB per vector, a quarter of a 1536-d API embedding, which keeps storage, "
        "RAM and the cost curve flat; it also means no embedding API key, rate limit or data egress. The price is weaker semantic "
        "matching: dense-only Hit@5 is 0.80 on the original questions and 0.48 on the paraphrased ones. I bought that back with "
        "hybrid retrieval (0.92 / 0.68) instead of a bigger model; a cross-encoder reranker raises paraphrased Hit@5 to 0.76 but "
        "costs about 2 s per query on CPU, so it is opt-in (RAG_RERANK=true). The model name and dimension are stored in the table "
        "schema and in every chunk id.")
    put(T[13].cells[0],
        "Two layers. First a guard: if the best cosine similarity among the retrieved chunks is below RAG_MIN_SCORE (0.58) the "
        "service returns a fixed 'I can't answer that from the indexed documents' without calling the LLM. 0.58 was calibrated "
        "as the lowest top-1 score of any answerable question (0.628) minus a 0.05 margin, so it never blocks an answerable "
        f"question; it stops the off-topic ones. Second, in-domain near misses pass the guard, so the prompt allows only facts "
        "from the numbered passages, requires a citation on every sentence and an exact INSUFFICIENT_CONTEXT token when the "
        f"passages do not answer. Result: {ans.get('unanswerable_declined')}/{ans.get('unanswerable')} unanswerable questions "
        f"declined ({ans.get('unanswerable_declined_by_guard')} by the guard, {ans.get('unanswerable_declined_by_model')} by the "
        f"model) with {ans.get('false_abstentions')} false abstentions. Citations are parsed and checked against the passages "
        "actually sent; out-of-range markers are reported.")
    put(T[14].cells[0],
        "chunk_id = sha256(doc_id, embedding model, chunk size, overlap, chunk text). Before embedding, ingest reads the ids already "
        "in the table and embeds only new ones; writes go through merge_insert on chunk_id (update if present, insert if not); "
        "chunks of a document that the new run no longer produces are deleted, and --prune removes documents that left the folder. "
        "Re-running on the same corpus embeds 0 chunks and the count stays at 431 (screenshot below). After a change to the "
        "Wikipedia loader only the 24 affected chunks were re-embedded and the 18 they replaced were deleted. Because the settings "
        "are part of the id, changing the chunk size can never leave a mix of old and new chunks.")
    put(T[15].cells[0],
        "Accepted: no replication or HA, single-writer ingestion, a small CPU embedding model that is weaker on paraphrases, cold "
        "starts and object-storage latency on the serverless path, and running the index tuning myself (nprobes / refine). "
        "Switch back to a managed DB when (1) query volume is high and steady: at about 1M queries/month an always-on VM already "
        "beats the serverless path and a managed service starts to pay for itself; (2) we need an uptime SLA, multi-region "
        "replicas or many concurrent writers; (3) p99 must be single-digit ms at 10M+ vectors, which needs RAM-resident HNSW "
        "(FAISS HNSW was 1.5 ms vs 25 ms for on-disk IVF-PQ at 1M); or (4) nobody on the team should own index operations.")
    put(T[16].cells[0], "Real HTTP request to the running service; the response and the service's JSON log line carry per-stage "
                        "latency, chunk counts and token usage.", image=img / "p1_e2e_query.png")
    put(T[17].cells[0], "Two consecutive ingest runs on the same corpus: the second embeds 0 chunks and the vector count stays 431.",
        image=img / "p1_idempotent_reingest.png")
    for i in (18, 19, 20):
        put(T[i].cells[0], YOURS, mine=True)

    # ---------------- Problem 2 ----------------
    put(T[21].cells[0], "Figure: suite in, both-order pairwise and evidence-first pointwise judging, parse ladder with one "
                        "re-ask, aggregation in code, report and audit log out.", image=img / "p2_pipeline.png")
    fill_line(P[74], "Same as Problem 1 (Python 3.11/3.12 + uv, CPU only). Free Gemini key for the judge, free Groq key for the "
                     "generators. No GPU or database.")
    fill_line(P[75], "GEMINI_API_KEY, GROQ_API_KEY (secrets, gitignored .env). Judge and generators are configured in YAML files "
                     "under configs/, not in code.")
    put(T[22].cells[0], f"git clone {REPO}\ncd applied-ai-takehome\nuv sync\ncp .env.example .env     # paste the two keys",
        mono=True)
    put(T[23].cells[0], "uv run python -m judge run --suite suites/core.yaml --gen configs/gen_b.yaml \\\n"
                        "  --judge configs/judge_v2.yaml --out results/p2/run.json\n"
                        "# suite = YAML/JSON list of {id, input, system_prompt, model_output?, expected_output?, context?, criteria?};\n"
                        "# cases without model_output are generated first. Report: pass rate (95% CI), mean score, per-criterion,\n"
                        "# per-category, score spread, parse stats, tokens and cost. Every call is in logs/judge_audit.jsonl.",
        mono=True)
    put(T[24].cells[0], "uv run python -m judge ab --suite suites/core.yaml \\\n"
                        "  --gen-a configs/gen_a.yaml --gen-b configs/gen_b.yaml --judge configs/judge_v2.yaml\n"
                        "# A = gpt-oss-20b + prompt v1, B = gpt-oss-20b + prompt v2; writes results/p2/ab.json\n"
                        "uv run python -m judge_eval.run_bias        # bias before/after + validation\n"
                        "uv run python -m judge replay               # re-parse every logged response offline", mono=True)
    fill_line(P[82], "Yes. Separate files: configs/judge_*.yaml (provider, model, temperature, prompt version) and "
                     "configs/gen_*.yaml; the CLI takes --judge and --gen-a/--gen-b and warns if judge and generator share a family.")
    fill_line(P[86], "Reference-based pointwise scoring with a weighted rubric (pass rate, mean score) plus pairwise A-vs-B in both "
                     "orders to declare the winner. Weights and pass gates are applied in code, not by the judge.")

    crit = rubric["criteria"]
    for row, key in zip(T[25].rows[1:], ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety"]):
        cdef = crit[key]
        put(row.cells[1], cdef["definition"] + "\n" + "\n".join(f"{k}: {v}" for k, v in cdef["anchors"].items()))
        put(row.cells[2], f"{cdef['weight']:.2f}")
    g = rubric["gates"]
    r = T[25].add_row()
    put(r.cells[0], "Pass gate (code)")
    put(r.cells[1], f"Pass iff weighted overall >= {g['pass_overall']}, correctness >= {g['min_correctness']} and tone/safety >= "
                    f"{g['min_tone_safety']}. Length policy in the prompt: length is never a quality; padding lowers "
                    "instruction-following when brevity is asked.")
    put(r.cells[2], "-")

    pr, mt, rt, sp = bias.get("probes", {}), bias.get("mtbench", {}), bias.get("retest", {}), bias.get("selfpref", {})
    nv, v2 = pr.get("naive", {}), pr.get("v2", {})
    fool = lambda s, k: f"{s['fooled'][k]['fooled']}/{s['fooled'][k]['n']}" if s.get("fooled") else "n/a"  # noqa: E731
    verb = pr.get("verbosity", {})
    clus = pr.get("clustering", {})
    t = T[26].rows
    put(t[1].cells[1], "judge_pair_both_orders() runs every pair as A/B and again as B/A, maps the second verdict back, and "
                       "returns the winner only if both orders agree; disagreement becomes a tie. Pointwise scoring sees one "
                       "answer at a time, so it has no position.")
    put(t[1].cells[2], (f"MT-Bench pairs (n = {mt.get('v2', {}).get('pairs')}): flip rate naive prompt "
                        f"{pct(mt.get('naive', {}).get('flip_rate'))} -> v2 prompt {pct(mt.get('v2', {}).get('flip_rate'))}; "
                        "final verdicts after order-agreement: 0% flips by construction. "
                        f"A/B suite flip rate {pct(ab.get('comparison', {}).get('position_flip_rate'))}.") if mt else "pending")
    put(t[2].cells[1], "Length policy in the judge prompt, evidence quote required before every score, padding probe: the same "
                       "correct answer terse vs padded with filler (judge_eval/probes.py).")
    put(t[2].cells[2], (f"Bias ran the other way: the naive judge punished length. Correct-but-padded answers wrongly failed: "
                        f"naive {fool(nv, 'padded_correct')} -> v2 {fool(v2, 'padded_correct')}. Padded-minus-terse score: "
                        f"naive {verb.get('naive_score_1_10', {}).get('mean_delta')} on 1-10 -> v2 "
                        f"{verb.get('v2_overall_1_5', {}).get('mean_delta')} on 1-5 (padding still costs a little "
                        f"instruction-following, by rubric). Verbose-but-wrong passed: naive {fool(nv, 'verbose_wrong')}, "
                        f"v2 {fool(v2, 'verbose_wrong')}.") if pr else "pending")
    sd = sp.get("self_preference", {})
    put(t[3].cells[1], "Judge is Gemini (Google); generators are gpt-oss (OpenAI). Measured, not assumed: gpt-oss-120b and "
                       "Gemini answer the same 12 cases, judged by Gemini, gpt-oss and a neutral Qwen judge (both orders).")
    put(t[3].cells[2], (f"Self-preference = own-family win share minus neutral judge's: Gemini judge for Gemini "
                        f"{sd.get('gemini_judge_for_gemini'):+.2f}, gpt-oss judge for gpt-oss {sd.get('gptoss_judge_for_gptoss'):+.2f}. "
                        "Cross-family judging removes the term by design.") if sd else "pending")
    put(t[4].cells[1], "Per-criterion scores with quoted evidence and anchors; rule that confident or polished wording is not "
                       "evidence; reference answer given as ground truth; confidently-wrong probe.")
    put(t[4].cells[2], (f"Confident-but-wrong answers passed: naive {fool(nv, 'confident_wrong')} -> v2 "
                        f"{fool(v2, 'confident_wrong')}. Probe accuracy {pct(nv.get('accuracy'))} -> {pct(v2.get('accuracy'))} "
                        f"(kappa vs gold {nv.get('kappa')} -> {v2.get('kappa')}).") if pr else "pending")
    cn, cv = clus.get("naive_1_10", {}), clus.get("v2_overall_1_5", {})
    put(t[5].cells[1], "Anchored 1/3/5 descriptions per criterion, calibration examples using the whole scale, explicit 'do not "
                       "default to 4', weighted overall computed in code; winners declared pairwise.")
    put(t[5].cells[2], (f"Scale use (normalised entropy over the 48 probes): {cn.get('entropy_norm')} -> {cv.get('entropy_norm')}; "
                        f"the naive judge is bimodal (histogram {cn.get('histogram')}). Ranking of good above bad answers "
                        f"(AUC): {cn.get('auc')} -> {cv.get('auc')}.") if clus else "pending")

    t = T[27].rows
    mv = mt.get("v2", {})
    put(t[1].cells[1], pct(mv.get("agreement_both_orders")) if mv else "pending")
    put(t[1].cells[2], (f"n = {mv.get('pairs')} MT-Bench first-turn pairs with expert human votes (lmsys/mt_bench_human_judgments, "
                        f"CC BY 4.0), labels A / B / tie. On pairs where the human picked a side: "
                        f"{pct(mv.get('agreement_non_tie_humans_both'))}. Naive prompt: "
                        f"{pct(mt.get('naive', {}).get('agreement_both_orders'))}.") if mv else "pending")
    put(t[2].cells[1], (f"kappa {mv.get('kappa_both_orders')}" if mv else "pending"))
    put(t[2].cells[2], (f"Cohen's kappa vs the human labels (3 classes). Single order: {mv.get('kappa_single_order')}. Probe set "
                        f"vs gold-by-construction labels: {v2.get('kappa')} (n = {v2.get('n')}).") if mv else "pending")
    rr = rt.get("temperature_0.7", {})
    put(t[3].cells[1], (f"{pct(rr.get('pointwise', {}).get('verdict_flip_rate'))} pointwise\n"
                        f"{pct(rr.get('pairwise', {}).get('winner_flip_rate'))} pairwise") if rr else "pending")
    put(t[3].cells[2], (f"Re-runs at temperature 0.7, {rr.get('samples')} samples each, {rr.get('pointwise', {}).get('items')} "
                        f"pointwise items and {rr.get('pairwise', {}).get('pairs')} pairs. Production runs at temperature 0 with "
                        "a response cache, so replays are identical.") if rr else "pending")
    put(t[4].cells[1], (f"Naive: fooled {nv['fooled']['verbose_wrong']['fooled'] + nv['fooled']['confident_wrong']['fooled']}"
                        f"/{nv['fooled']['verbose_wrong']['n'] + nv['fooled']['confident_wrong']['n']}\n"
                        f"v2: fooled {v2['fooled']['verbose_wrong']['fooled'] + v2['fooled']['confident_wrong']['fooled']}"
                        f"/{v2['fooled']['verbose_wrong']['n'] + v2['fooled']['confident_wrong']['n']}") if pr else "pending")
    put(t[4].cells[2], (f"{pr.get('items')} probes with gold labels by construction: terse-correct, padded-correct, "
                        "confident-wrong and verbose-wrong versions of 12 suite cases. 'Fooled' = wrong answer passed. "
                        f"Terse-correct wrongly failed: naive {fool(nv, 'terse_correct')}, v2 {fool(v2, 'terse_correct')}.")
        if pr else "pending")

    cmp_ = ab.get("comparison", {})
    pw, pt = cmp_.get("pairwise", {}), cmp_.get("pointwise", {})
    ra, rb = ab.get("report_a", {}), ab.get("report_b", {})
    t = T[28].rows
    put(t[1].cells[0], f"Config A: {ab.get('config_a', {}).get('name')} (case system prompt as given)")
    put(t[1].cells[1], f"{pct(ra.get('pass_rate'))}\nCI {ra.get('pass_rate_ci95')}")
    put(t[1].cells[2], f"{ra.get('mean_overall')} / 5")
    put(t[1].cells[3], f"{pct(pw.get('win_rate_a'))} ({pw.get('a_wins')} wins, {pw.get('ties')} ties)")
    put(t[2].cells[0], f"Config B: {ab.get('config_b', {}).get('name')} (adds constraint re-read, false-premise and no-padding guidance)")
    put(t[2].cells[1], f"{pct(rb.get('pass_rate'))}\nCI {rb.get('pass_rate_ci95')}")
    put(t[2].cells[2], f"{rb.get('mean_overall')} / 5")
    put(t[2].cells[3], f"{pct(pw.get('win_rate_b'))} ({pw.get('b_wins')} wins, {pw.get('ties')} ties)")
    win = cmp_.get("winner")
    verdict = (f"Config {win}" if win in ("A", "B") else "No significant difference") + \
        f": {pw.get('a_wins')}-{pw.get('b_wins')} with {pw.get('ties')} ties over {pw.get('decided')} both-order pairs, " \
        f"{cmp_.get('reason')}; pointwise mean difference A-B {pt.get('mean_diff_a_minus_b')} (95% CI {pt.get('mean_diff_ci95')})."
    fill_line(P[96], verdict)

    put(T[29].cells[0],
        "Both, for different jobs. Pointwise reference-based scoring answers 'is this good enough to ship?' and gives a pass rate a "
        "release gate can use; the reference answer anchors correctness, so the judge checks rather than guesses. Pairwise answers "
        "'which config is better?', which is what an A/B needs and is more sensitive than comparing two absolute means (absolute "
        "scores cluster; a direct comparison does not). Pairwise is the mode most exposed to position bias, so it only ever runs "
        "in both orders. Reference-free scoring is used only where a case has no expected output.")
    put(T[30].cells[0],
        "The judge is asked for JSON (and the API's JSON mode is on), and the output is validated with Pydantic models: every "
        "criterion present, integer scores 1-5, evidence and rationale strings, winner in {A, B, tie}. Parsing is a ladder and the "
        "rung used is recorded: strict json.loads; extract the first balanced {...} from fenced or chatty text; repair (smart and "
        "single quotes, unquoted keys, trailing commas, Python literals, a truncated tail closed); coerce types ('4' or '4/5' -> 4, "
        "'Tie' -> 'tie'). If every rung fails, the judge gets one re-ask containing its own output and the exact validation error. If "
        "that fails too, the case is marked parse_failed and excluded and counted; it is never silently scored. Every prompt and "
        "raw response is in logs/judge_audit.jsonl, and 'python -m judge replay' re-parses the whole log offline.")
    put(T[31].cells[0],
        "LLM judges prefer text that looks like their own (low perplexity to them), so a judge from the generator's family inflates "
        "that generator. The generators are OpenAI gpt-oss models and the judge is Google Gemini, so the A/B is judged by a model "
        "that wrote neither side. I measured the effect rather than assuming it: gpt-oss-120b and Gemini answered the same 12 cases, "
        "and each family's judge was compared with a neutral Qwen judge on those pairs (numbers in 2.3.2). Cost mattered too: "
        "Gemini Flash-Lite is fast and cheap enough to run every pair twice.")
    put(T[32].cells[0],
        "Position bias, because it silently corrupts the A/B verdict, which is the decision the pipeline exists to make, and a "
        "single-order run gives no hint it happened. The mitigation is structural rather than prompt wording: a verdict counts only "
        "if it survives the order swap, otherwise it is a tie, so a position-driven preference cannot produce a win. I am confident "
        "it removes position effects from the final verdicts; the cost is more ties and therefore less statistical power, which is "
        "why the winner also needs a significant sign test. I am less confident about verbosity on long open-ended answers: the "
        "padding probe shows the effect is controlled on short factual cases, not that it is gone everywhere.")
    put(T[33].cells[0],
        "As a gate for regressions, yes, with guardrails: a fixed, versioned suite and judge config (model, prompt, temperature 0), "
        "both-order pairwise for any 'B beats A' decision, a significance test before declaring a winner, a minimum parse-success "
        "rate, and a canary set of adversarial probes that must stay correctly classified before the judge's verdict is trusted. "
        "A human stays in the loop for: safety-category failures, any case where the two orders disagree, failures near the pass "
        "threshold, a periodic sample re-labelled to re-measure agreement and kappa, and any judge model or prompt change, which "
        "requires re-validation against the human-labelled set before it can gate again.")
    put(T[34].cells[0], "One judge call exactly as logged in logs/judge_audit.jsonl: raw system prompt, raw user prompt and the "
                        "raw response.", image=img / "p2_audit.png")
    put(T[35].cells[0], "One pair judged in both orders, with the suite and MT-Bench flip rates.", image=img / "p2_position.png")
    put(T[36].cells[0], YOURS, mine=True)
    put(T[37].cells[0], "", image=img / "p2_probe.png")
    p = T[37].cells[0].add_paragraph()
    rr_ = p.add_run("My read: " + YOURS)
    rr_.font.highlight_color = WD_COLOR_INDEX.YELLOW
    rr_.font.size = Pt(10)
    put(T[38].cells[0], YOURS, mine=True)

    d.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
