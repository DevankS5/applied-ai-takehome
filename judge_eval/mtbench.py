"""A fixed, seeded sample of MT-Bench expert human pairwise judgments (lmsys/mt_bench_human_judgments,
CC-BY-4.0, Zheng et al. 2023) used as gold labels for judge validation. First turns only, so each
pair is one question and two single answers. Cached to data/hf/ after the first download."""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

from judge.schema import TestCase

CACHE = Path("data/hf/mt_bench_human_turn1.json")
API = "https://datasets-server.huggingface.co/rows?dataset=lmsys/mt_bench_human_judgments&config=default&split=human"


def _download() -> list[dict]:
    rows, offset = [], 0
    with httpx.Client(timeout=60) as c:
        while True:
            r = c.get(f"{API}&offset={offset}&length=100")
            r.raise_for_status()
            batch = [x["row"] for x in r.json()["rows"]]
            rows += [x for x in batch if x["turn"] == 1]
            if len(batch) < 100:
                break
            offset += 100
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def sample(n: int = 40, seed: int = 13) -> list[dict]:
    """n distinct (question, model pair) items; one human vote each; label in {'A','B','tie'}."""
    rows = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else _download()
    seen, items = set(), []
    rng = random.Random(seed)
    rng.shuffle(rows)
    for r in rows:
        key = (r["question_id"], r["model_a"], r["model_b"])
        if key in seen:
            continue
        seen.add(key)
        label = {"model_a": "A", "model_b": "B"}.get(r["winner"], "tie")
        case = TestCase(id=f"mtb-{r['question_id']}-{r['model_a']}-vs-{r['model_b']}", input=r["conversation_a"][0]["content"],
                        category=f"mtbench-q{r['question_id']}")
        items.append({"case": case, "a": r["conversation_a"][1]["content"], "b": r["conversation_b"][1]["content"],
                      "human": label, "model_a": r["model_a"], "model_b": r["model_b"], "expert": r["judge"]})
        if len(items) == n:
            break
    return items
