"""IR and QA metrics, written out so they can be checked by hand (see tests/test_metrics.py).

Relevance is binary: a retrieved chunk is relevant to a question iff it contains one of
the question's verbatim evidence quotes (after normalisation), or at least 60% of a quote
as one contiguous run when a chunk boundary cuts through it.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter

import numpy as np


def norm_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z]+", " ", s).split()


def _longest_run(needle: list[str], hay: list[str]) -> int:
    """Length of the longest contiguous run of `needle` tokens that appears in `hay`."""
    best = 0
    hay_s = " " + " ".join(hay) + " "
    n = len(needle)
    for size in range(n, 0, -1):
        if size <= best:
            break
        for i in range(0, n - size + 1):
            if " " + " ".join(needle[i : i + size]) + " " in hay_s:
                return size
    return best


def is_relevant(chunk_text: str, evidence: list[str], min_frac: float = 0.6) -> bool:
    hay = norm_tokens(chunk_text)
    for q in evidence:
        need = norm_tokens(q)
        if need and _longest_run(need, hay) >= max(1, math.ceil(min_frac * len(need))):
            return True
    return False


# ---- ranking metrics on a binary relevance list for one query (rank order) ----
def hit_at_k(rels: list[bool], k: int) -> float:
    return float(any(rels[:k]))


def recall_at_k(rels: list[bool], k: int, n_relevant: int) -> float:
    return sum(rels[:k]) / n_relevant if n_relevant else 0.0


def reciprocal_rank(rels: list[bool], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def precision_at_k(rels: list[bool], k: int) -> float:
    return sum(rels[:k]) / k


def context_precision(rels: list[bool], k: int) -> float:
    """RAGAS-style: mean of precision@i over the ranks i that hold a relevant chunk.
    Rewards putting relevant chunks first; 0 when nothing relevant was retrieved."""
    hits, total = 0, 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


# ---- SQuAD-style answer metrics ----
def _squad_norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) | {"–", "—", "×"})
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def exact_match(pred: str, gold: str) -> float:
    return float(_squad_norm(pred) == _squad_norm(gold))


def contains_match(pred: str, gold: str) -> float:
    """Lenient EM for generative answers: the normalised gold answer appears in the prediction."""
    g, p = " ".join(_squad_norm(gold)), " ".join(_squad_norm(pred))
    return float(bool(g) and f" {g} " in f" {p} ")


def token_f1(pred: str, gold: str) -> float:
    p, g = _squad_norm(pred), _squad_norm(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")
