import math

from rag_eval import metrics as m

R = [False, True, False, True, False]  # relevant at ranks 2 and 4


def test_hit_and_recall():
    assert m.hit_at_k(R, 1) == 0.0
    assert m.hit_at_k(R, 2) == 1.0
    assert m.recall_at_k(R, 5, n_relevant=3) == 2 / 3
    assert m.recall_at_k(R, 5, n_relevant=0) == 0.0


def test_mrr():
    assert m.reciprocal_rank(R, 5) == 0.5
    assert m.reciprocal_rank(R, 1) == 0.0


def test_ndcg_by_hand():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(m.ndcg_at_k(R, 5, n_relevant=2), dcg / idcg)
    assert m.ndcg_at_k([True, True], 2, 2) == 1.0


def test_precision_and_context_precision():
    assert m.precision_at_k(R, 5) == 0.4
    # relevant at 2 and 4: (1/2 + 2/4) / 2
    assert math.isclose(m.context_precision(R, 5), 0.5)
    assert m.context_precision([False] * 5, 5) == 0.0


def test_relevance_matching_tolerates_formatting_and_boundaries():
    ev = ["Specify the size of the dynamic candidate list for search (40 by default)"]
    assert m.is_relevant("### Query Options\n\nSpecify the size of the dynamic candidate list for search (40 by default)", ev)
    # chunk cut after 70% of the quote still counts; 30% does not
    assert m.is_relevant("blah blah Specify the size of the dynamic candidate list", ev)
    assert not m.is_relevant("for search (40 by default) and other text", ev)


def test_squad_answer_metrics():
    assert m.exact_match("The 40.", "40") == 1.0
    assert m.contains_match("The default is 40 [1].", "40") == 1.0
    assert m.contains_match("The default is 400.", "40") == 0.0
    # pred tokens: m is 16 and efconstruction 64 (6); gold: m 16 and efconstruction 64 (5); 5 shared
    # P = 5/6, R = 1, F1 = 10/11
    assert math.isclose(m.token_f1("m is 16 and ef_construction 64", "m = 16 and ef_construction = 64"), 10 / 11)
