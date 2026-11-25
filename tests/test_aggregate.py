import math

from judge.aggregate import cohens_kappa, compare, overall, passes, sign_test_p, suite_report


def v(**scores):
    return {"criteria": {k: {"score": s, "evidence": "", "rationale": ""} for k, s in scores.items()}}


def test_overall_renormalises_weights():
    assert overall(v(correctness=5, faithfulness=5, completeness=5, instruction_following=5, tone_safety=5)) == 5
    # only correctness (0.3) and completeness (0.2) scored: (0.3*4 + 0.2*2) / 0.5
    assert math.isclose(overall(v(correctness=4, completeness=2)), 3.2)


def test_gates_override_a_high_average():
    high_but_wrong = v(correctness=2, faithfulness=5, completeness=5, instruction_following=5, tone_safety=5)
    assert overall(high_but_wrong) >= 3.5 and not passes(high_but_wrong)
    unsafe = v(correctness=5, faithfulness=5, completeness=5, instruction_following=5, tone_safety=1)
    assert not passes(unsafe)


def test_sign_test_and_kappa_by_hand():
    assert sign_test_p(5, 5) == 1.0
    assert math.isclose(sign_test_p(9, 1), 2 * (1 + 10) / 2**10)
    assert cohens_kappa(list("AABB"), list("AABB")) == 1.0
    # po = 0.5, pe = 0.5 -> kappa 0
    assert cohens_kappa(list("AABB"), list("ABAB")) == 0.0


def test_suite_report_counts_parse_failures_and_never_scores_them():
    ok = {"ok": True, "verdict": v(correctness=5, faithfulness=5, completeness=5, instruction_following=5, tone_safety=5),
          "parse_rung": "strict", "reasked": False, "calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0}
    bad = {"ok": False, "verdict": None, "parse_rung": None, "reasked": True, "calls": 2, "prompt_tokens": 10,
           "completion_tokens": 5, "cost_usd": 0.0}
    rep = suite_report([{"id": "a", "call": ok}, {"id": "b", "call": bad}])
    assert rep["parse_failed"] == 1 and rep["pass_rate"] == 1.0 and rep["judge_calls"] == 3


def test_compare_declares_no_winner_without_significance():
    r = {"pass_rate": 0.5, "mean_overall": 3.0}
    res = compare(r, r, ["A", "B", "A", "tie"], [3, 3], [3, 3])
    assert res["winner"] == "no significant difference"
    res = compare(r, r, ["B"] * 12, [3] * 12, [4] * 12)
    assert res["winner"] == "B"
