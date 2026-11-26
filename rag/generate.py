"""Grounded answer generation with numbered citations and an explicit abstain token."""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.llm import chat
from rag.config import get_settings

ABSTAIN = "INSUFFICIENT_CONTEXT"
NO_ANSWER_TEXT = "I can't answer that from the indexed documents."

SYSTEM = f"""You answer questions using ONLY the numbered context passages provided.
Rules:
1. Use only facts stated in the passages. Do not use outside knowledge, even if you are sure.
2. End every sentence with the passage number(s) that support it, like [1] or [2][3].
3. If the passages do not contain the answer, reply with exactly {ABSTAIN} and nothing else.
4. Be concise: at most 4 sentences."""


@dataclass
class Answer:
    text: str
    abstained: bool
    cited: list[int]  # 1-based passage numbers actually cited
    invalid_citations: list[int]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


def format_context(rows: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(rows, start=1):
        where = f"p. {r['page']}" if r.get("page") else (r.get("heading") or "")
        blocks.append(f"[{i}] ({r['title']}{', ' + where if where else ''})\n{r['text']}")
    return "\n\n".join(blocks)


def normalize(text: str) -> str:
    """gpt-oss cites as 【1】 or 【1†L2-L4】 and uses narrow no-break spaces; map both to plain [1] and spaces."""
    text = re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", text)
    return re.sub(r"[   ]", " ", text).strip()


def generate(question: str, rows: list[dict], provider: str | None = None, model: str | None = None) -> Answer:
    s = get_settings()
    provider, model = provider or s.gen_provider, model or s.gen_model
    user = f"Context passages:\n\n{format_context(rows)}\n\nQuestion: {question}\nAnswer:"
    res = chat(provider, model, user, system=SYSTEM, temperature=0.0, max_tokens=s.gen_max_tokens, cache_dir=s.llm_cache_dir)
    text = normalize(res.text)
    abstained = ABSTAIN in text or not text
    nums = [int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.split(r"\s*,\s*", group)]
    cited = sorted({n for n in nums if 1 <= n <= len(rows)})
    invalid = sorted({n for n in nums if not 1 <= n <= len(rows)})
    return Answer(
        NO_ANSWER_TEXT if abstained else text, abstained, [] if abstained else cited, invalid,
        res.prompt_tokens, res.completion_tokens, res.latency_ms, model,
    )
