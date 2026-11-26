"""Calling the judge: structured verdicts, one repair re-ask, both-order pairwise, full audit trail."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import yaml

from common.llm import LLMError, chat
from judge import prompts
from judge.parse import ParseError, parse_verdict
from judge.schema import NaiveScore, PairVerdict, TestCase, Verdict

AUDIT = "logs/judge_audit.jsonl"


@dataclass
class JudgeConfig:
    """Judge settings, independent of any generator config."""

    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    prompt: str = "v2"  # v2 | naive
    max_tokens: int = 1500
    name: str = ""

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(os.environ.get("JUDGE_PROVIDER", cls.provider), os.environ.get("JUDGE_MODEL", cls.model),
                   float(os.environ.get("JUDGE_TEMPERATURE", cls.temperature)))

    @classmethod
    def load(cls, path: str) -> "JudgeConfig":
        return cls(**yaml.safe_load(open(path, encoding="utf-8")))

    @property
    def label(self) -> str:
        return self.name or f"{self.provider}/{self.model}/{self.prompt}"


@dataclass
class Call:
    """Result of one judging attempt, kept whole for the report and replay."""

    ok: bool
    verdict: dict | None
    parse_rung: str | None
    reasked: bool
    error: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    raw: list[str] = field(default_factory=list)


def _run(cfg: JudgeConfig, system: str, user: str, schema, tag: str, sample: int = 0) -> Call:
    call = Call(False, None, None, False, None)
    sys_, usr = system, user
    for attempt in range(2):  # first try + one repair re-ask
        try:
            res = chat(cfg.provider, cfg.model, usr, system=sys_, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                       json_mode=True, sample=sample, audit_path=AUDIT, tag=f"{tag}|attempt{attempt}")
        except LLMError as e:
            call.error = f"llm_error: {e}"
            return call
        call.calls += 1
        call.prompt_tokens += res.prompt_tokens
        call.completion_tokens += res.completion_tokens
        call.cost_usd += res.cost_usd
        call.raw.append(res.text)
        try:
            v, rung = parse_verdict(res.text, schema)
            call.ok, call.verdict, call.parse_rung, call.error = True, v.model_dump(), rung, None
            return call
        except ParseError as e:
            call.error = f"parse_failed: {e}"
            call.reasked = True
            sys_, usr = prompts.reask(res.text, str(e), system, user)
    return call


def judge_pointwise(case: TestCase, cfg: JudgeConfig, sample: int = 0, tag: str = "") -> Call:
    if case.model_output is None:
        raise ValueError(f"case {case.id} has no model_output")
    if cfg.prompt == "naive":
        system, user = prompts.pointwise_naive(case)
        return _run(cfg, system, user, NaiveScore, f"pointwise_naive|{case.id}|{tag}", sample)
    system, user = prompts.pointwise_v2(case)
    return _run(cfg, system, user, Verdict, f"pointwise_v2|{case.id}|{tag}", sample)


def judge_pair(case: TestCase, a: str, b: str, cfg: JudgeConfig, sample: int = 0, tag: str = "") -> Call:
    build = prompts.pairwise_naive if cfg.prompt == "naive" else prompts.pairwise_v2
    system, user = build(case, a, b)
    return _run(cfg, system, user, PairVerdict, f"pairwise_{cfg.prompt}|{case.id}|{tag}", sample)


SWAP = {"A": "B", "B": "A", "tie": "tie"}


def judge_pair_both_orders(case: TestCase, x: str, y: str, cfg: JudgeConfig, sample: int = 0, tag: str = "") -> dict:
    """Position-bias mitigation: judge (x, y) and (y, x); map the second back to x/y labels.
    Final = the shared winner when both orders agree, else 'tie' (agreement-or-tie)."""
    first = judge_pair(case, x, y, cfg, sample, f"{tag}|xy")
    second = judge_pair(case, y, x, cfg, sample, f"{tag}|yx")
    w1 = first.verdict["winner"] if first.ok else None
    w2 = SWAP[second.verdict["winner"]] if second.ok else None  # back to x=A, y=B labels
    consistent = w1 is not None and w1 == w2
    return {
        "order_xy": w1, "order_yx_mapped": w2, "consistent": consistent,
        "flipped": (w1 is not None and w2 is not None and w1 != w2),
        "final": (w1 if consistent else "tie") if (w1 and w2) else None,
        "calls": [asdict(first), asdict(second)],
    }
