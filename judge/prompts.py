"""Judge prompts. Two versions of each so every mitigation can be measured against a baseline:

- naive: what a first attempt looks like (bare 1-10 number / "which is better?").
- v2:    explicit rubric with anchors, evidence-before-score, length and confidence policy,
         few-shot calibration anchors, and (pairwise) an order-is-random statement. Position
         bias is also handled in code by judging both orders (judge.py), not by prompt alone.
"""

from __future__ import annotations

import json

from judge.schema import TestCase, criteria_names, rubric

CALIBRATION = """Calibration examples (scores for `correctness` only):
- Q: "What is 12 x 12?" Reference: "144". Response: "144." -> 5 (exactly right; brevity is fine).
- Q: "What is 12 x 12?" Reference: "144". Response: "Great question! Multiplication is repeated addition, and after careful
  thought I am completely certain the answer is 124." -> 1 (wrong; the confident, friendly wording earns nothing).
- Q: "Name the capital of Japan and its population." Reference: "Tokyo, about 14 million." Response: "Tokyo." -> 5 for
  correctness, 3 for completeness (the second part is missing)."""


def _rubric_text(names: list[str]) -> str:
    out = []
    for n in names:
        c = rubric()["criteria"][n]
        anchors = "; ".join(f"{k} = {v}" for k, v in c["anchors"].items())
        out.append(f"- {n}: {c['definition']} Anchors: {anchors}")
    return "\n".join(out)


def _task_block(case: TestCase, response_label: str = "RESPONSE") -> str:
    parts = [f"<system_prompt_given_to_model>\n{case.system_prompt or '(none)'}\n</system_prompt_given_to_model>",
             f"<user_input>\n{case.input}\n</user_input>"]
    if case.context:
        parts.append(f"<context_given_to_model>\n{case.context}\n</context_given_to_model>")
    if case.expected_output:
        parts.append(f"<reference_answer>\n{case.expected_output}\n</reference_answer>")
    return "\n".join(parts)


# ---------------- pointwise ----------------
POINTWISE_V2_SYSTEM = f"""You are a strict, calibrated evaluation judge. You grade one model RESPONSE against a rubric.

Rules:
1. Judge only against the user input, the system prompt the model was given, the reference answer and any context provided.
   The reference answer is ground truth for correctness.
2. For every criterion, FIRST copy the exact span (at most 25 words) from the response or reference that decides the score
   into "evidence", THEN explain in "rationale", THEN give "score". If you cannot quote evidence, the score cannot exceed 3.
3. Confident, polished, friendly or expert-sounding wording is not evidence. A fluent wrong answer scores 1 on correctness.
4. {rubric()['length_policy']}
5. Use the whole 1-5 scale. 5 means nothing to improve on that criterion; do not default to 4.

{CALIBRATION}

Return ONLY a JSON object, no prose, in this shape:
{{"criteria": {{"<criterion>": {{"evidence": "...", "rationale": "...", "score": 1}}}}, "summary": "one sentence"}}"""


def pointwise_v2(case: TestCase) -> tuple[str, str]:
    names = case.criteria or criteria_names()
    user = (f"{_task_block(case)}\n<response>\n{case.model_output}\n</response>\n\n"
            f"Score these criteria: {', '.join(names)}.\nRubric:\n{_rubric_text(names)}")
    return POINTWISE_V2_SYSTEM, user


POINTWISE_NAIVE_SYSTEM = 'You are an AI judge. Rate the quality of the response from 1 to 10. Return JSON: {"score": <1-10>, "reason": "..."}'


def pointwise_naive(case: TestCase) -> tuple[str, str]:
    return POINTWISE_NAIVE_SYSTEM, f"{_task_block(case)}\n<response>\n{case.model_output}\n</response>"


# ---------------- pairwise ----------------
PAIR_V2_SYSTEM = f"""You are a strict, impartial judge comparing two responses, A and B, to the same user input.

Rules:
1. The order of A and B is random and carries no information. Do not favour the first or the second.
2. Do not favour the longer response. {rubric()['length_policy']}
3. Confidence and style are not evidence; check the substance against the reference answer when one is given.
4. Decide each criterion separately (A, B or tie), quoting what decides it in your head, then give the overall winner.
   Say "tie" when the responses are equally good or equally bad on what matters; do not force a winner.

Return ONLY a JSON object:
{{"per_criterion": {{"<criterion>": "A" | "B" | "tie"}}, "rationale": "two sentences", "winner": "A" | "B" | "tie"}}"""


def pairwise_v2(case: TestCase, a: str, b: str) -> tuple[str, str]:
    names = case.criteria or criteria_names()
    user = (f"{_task_block(case)}\n<response_A>\n{a}\n</response_A>\n<response_B>\n{b}\n</response_B>\n\n"
            f"Criteria: {', '.join(names)}.\nRubric:\n{_rubric_text(names)}")
    return PAIR_V2_SYSTEM, user


PAIR_NAIVE_SYSTEM = 'Which response is better, A or B? Return JSON: {"winner": "A" | "B" | "tie", "rationale": "..."}'


def pairwise_naive(case: TestCase, a: str, b: str) -> tuple[str, str]:
    return PAIR_NAIVE_SYSTEM, f"{_task_block(case)}\n<response_A>\n{a}\n</response_A>\n<response_B>\n{b}\n</response_B>"


def reask(previous: str, error: str, system: str, user: str) -> tuple[str, str]:
    """One repair turn: the same task plus the broken output and the parser's error."""
    return system, (f"{user}\n\nYour previous reply could not be parsed ({error}). Previous reply:\n"
                    f"{json.dumps(previous[:2000])}\nReturn ONLY the corrected JSON object.")
