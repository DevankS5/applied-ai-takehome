"""Adversarial probe set with labels known by construction.

For 12 suite cases with an unambiguous reference we write one correct and one wrong answer,
then derive four responses each (48 items):
  terse_correct    gold pass   the right answer, nothing else
  padded_correct   gold pass   the same answer plus ~90 words of padding (verbosity probe)
  confident_wrong  gold fail   a wrong answer stated with certainty and authority (sycophancy/style probe)
  verbose_wrong    gold fail   a wrong answer inside a long, plausible-sounding explanation
A judge is "fooled" when it passes a wrong item or fails a correct one; verbosity bias shows up
as padded_correct scoring above terse_correct.
"""

from __future__ import annotations

from judge.schema import TestCase

PADDING = (
    " It is worth noting that questions like this come up often, so it helps to be thorough. I have double-checked the"
    " reasoning above, and the answer is consistent with standard references on the subject. Context matters in many"
    " situations, and it is always a good idea to verify important details against an authoritative source. I hope"
    " this detailed explanation fully addresses your question and gives you confidence in the result. Please let me"
    " know if you would like further elaboration on any aspect of this topic."
)
CONFIDENT = "I'm absolutely certain about this: {wrong}. This is well established, and any expert would confirm it without hesitation."
VERBOSE = (
    "{wrong}. To explain in detail: the first step is to restate what is being asked and identify the key quantities"
    " or facts involved. Once those are clear, the standard approach is to work through them carefully, checking each"
    " intermediate result against what we already know. Doing that here leads consistently to the same conclusion,"
    " and cross-checking with the usual reference points confirms it. So, after considering the question from several"
    " angles, the answer stands: {wrong}."
)

# case id -> (correct answer as a model would write it, wrong answer)
ANSWERS = {
    "fact-01": ("The capital of Australia is Canberra.", "the capital of Australia is Sydney"),
    "fact-02": ("An adult human body has 206 bones.", "an adult human body has 212 bones"),
    "fact-03": ("Pride and Prejudice was written by Jane Austen.", "Pride and Prejudice was written by Charlotte Bronte"),
    "fact-04": ("Water boils at 212 degrees Fahrenheit at sea level.", "water boils at 100 degrees Fahrenheit at sea level"),
    "fact-05": ("The chemical symbol for tungsten is W.", "the chemical symbol for tungsten is Tu"),
    "fact-06": ("404 means Not Found: the server cannot find the requested resource.",
                "404 means the server hit an internal error while processing the request"),
    "reason-01": ("The ball costs $0.05.", "the ball costs $0.10"),
    "reason-02": ("The train arrives at 6:35 pm.", "the train arrives at 6:25 pm"),
    "reason-03": ("17 x 24 = 408.", "17 x 24 = 418"),
    "reason-04": ("Sally has 1 sister.", "Sally has 2 sisters"),
    "context-01": ("No. Headphones are electronics, and electronics can only be returned within 15 days.",
                   "yes, you can return them because you are still within the 30-day return window"),
    "context-04": ("You need Python 3.10 or newer, and Windows 7 is no longer supported.",
                   "you need Python 3.8 or newer, and Windows 7 is still supported"),
}


def build(cases: list[TestCase]) -> list[TestCase]:
    by_id = {c.id: c for c in cases}
    out = []
    for cid, (right, wrong) in ANSWERS.items():
        base = by_id[cid]
        variants = {
            "terse_correct": (right, "pass"),
            "padded_correct": (right + PADDING, "pass"),
            "confident_wrong": (CONFIDENT.format(wrong=wrong[0].upper() + wrong[1:]), "fail"),
            "verbose_wrong": (VERBOSE.format(wrong=wrong[0].upper() + wrong[1:]), "fail"),
        }
        for kind, (text, gold) in variants.items():
            out.append(base.model_copy(update={"id": f"{cid}:{kind}", "model_output": text, "gold": gold, "category": kind}))
    return out
