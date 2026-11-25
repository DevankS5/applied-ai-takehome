"""Data model: test cases in, structured verdicts out."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

RUBRIC_PATH = Path(__file__).with_name("rubric.yaml")


@lru_cache
def rubric() -> dict:
    return yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8"))


def criteria_names() -> list[str]:
    return list(rubric()["criteria"])


class TestCase(BaseModel):
    """One item of a suite: {input, system_prompt, model_output, expected_output?, criteria?}."""

    __test__ = False  # not a pytest class
    id: str
    input: str
    system_prompt: str = ""
    model_output: str | None = None  # filled by a generator config when absent
    expected_output: str | None = None  # reference answer -> reference-based judging
    context: str | None = None  # optional grounding material the model was given
    criteria: list[str] | None = None  # subset of rubric criteria to score (default: all)
    category: str = "general"
    gold: Literal["pass", "fail"] | None = None  # label known by construction or from humans

    @field_validator("input", "system_prompt", "model_output", "expected_output", "context", mode="before")
    @classmethod
    def scalars_to_text(cls, v):
        """YAML reads `206.` or `no` as a number/bool; a test case field is always text."""
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v) if isinstance(v, (int, float)) else v


class CriterionScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    evidence: str = ""  # quote from the response / reference that justifies the score
    rationale: str = ""


class Verdict(BaseModel):
    criteria: dict[str, CriterionScore]
    summary: str = ""

    @field_validator("criteria")
    @classmethod
    def known_criteria(cls, v: dict) -> dict:
        unknown = set(v) - set(criteria_names())
        if unknown:
            raise ValueError(f"unknown criteria {sorted(unknown)}")
        return v


class PairVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]
    per_criterion: dict[str, Literal["A", "B", "tie"]] = {}
    rationale: str = ""


class NaiveScore(BaseModel):
    """Baseline verdict shape (bare number), kept only to measure what the rubric fixes."""

    score: int = Field(..., ge=1, le=10)
    reason: str = ""
