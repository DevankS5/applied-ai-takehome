import pytest

from judge.parse import ParseError, parse_verdict
from judge.schema import PairVerdict, Verdict

GOOD = '{"criteria": {"correctness": {"score": 4, "evidence": "says 40", "rationale": "right"}}, "summary": "ok"}'


@pytest.mark.parametrize(
    "raw,rung",
    [
        (GOOD, "strict"),
        ("Here is my verdict:\n```json\n" + GOOD + "\n```\nHope this helps", "extracted"),
        ('{"criteria": {"correctness": {"score": 4, "evidence": "x", "rationale": "y",},}, "summary": "ok",}', "repaired"),
        ("{'criteria': {'correctness': {'score': 4, 'evidence': 'x', 'rationale': 'y'}}, 'summary': 'ok'}", "repaired"),
        ('{criteria: {correctness: {score: 4, evidence: "x", rationale: "y"}}, summary: "ok"}', "repaired"),
        ('{"criteria": {"correctness": {"score": "4/5", "evidence": "x", "rationale": "y"}}, "summary": "ok"}', "strict"),
        ("“criteria”", None),  # no object at all -> error
    ],
)
def test_parse_ladder(raw, rung):
    if rung is None:
        with pytest.raises(ParseError):
            parse_verdict(raw, Verdict)
        return
    v, used = parse_verdict(raw, Verdict)
    assert v.criteria["correctness"].score == 4
    assert used == rung


def test_truncated_output_is_closed():
    raw = '{"criteria": {"correctness": {"score": 2, "evidence": "the answer says 400", "rationale": "wrong numb'
    v, used = parse_verdict(raw, Verdict)
    assert v.criteria["correctness"].score == 2 and used == "repaired"


def test_out_of_range_and_unknown_criteria_are_rejected_not_clamped():
    with pytest.raises(ParseError):
        parse_verdict('{"criteria": {"correctness": {"score": 9}}}', Verdict)
    with pytest.raises(ParseError):
        parse_verdict('{"criteria": {"vibes": {"score": 5}}}', Verdict)


def test_pair_verdict_coercion():
    v, _ = parse_verdict('{"winner": "Tie", "per_criterion": {"correctness": "b"}, "rationale": "close"}', PairVerdict)
    assert v.winner == "tie" and v.per_criterion["correctness"] == "B"
