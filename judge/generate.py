"""Generator configs: produce model_output for a suite. Configured separately from the judge."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from common.llm import chat
from judge.schema import TestCase

# Prompt v2 adds guidance on top of each case's own system prompt; v1 sends the case prompt as-is.
V2_PREAMBLE = """Before answering: re-read every constraint in these instructions (format, length, language) and obey it exactly.
Be accurate; if the question rests on a false premise or the provided context does not contain the answer, say so plainly.
Do not pad the answer. Decline only requests that are actually harmful, and offer a safe alternative when you do."""


@dataclass
class GenConfig:
    name: str
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    prompt: str = "v1"  # v1 | v2
    temperature: float = 0.0
    max_tokens: int = 600

    @classmethod
    def load(cls, path: str) -> "GenConfig":
        return cls(**yaml.safe_load(open(path, encoding="utf-8")))


def system_for(case: TestCase, cfg: GenConfig) -> str:
    if cfg.prompt == "v2":
        return f"{case.system_prompt}\n\n{V2_PREAMBLE}".strip()
    return case.system_prompt


def user_for(case: TestCase) -> str:
    return f"Context:\n{case.context}\n\nQuestion: {case.input}" if case.context else case.input


def generate(case: TestCase, cfg: GenConfig) -> TestCase:
    res = chat(cfg.provider, cfg.model, user_for(case), system=system_for(case, cfg), temperature=cfg.temperature,
               max_tokens=cfg.max_tokens, audit_path="logs/generator_audit.jsonl", tag=f"gen|{cfg.name}|{case.id}")
    return case.model_copy(update={"model_output": res.text.strip()})
