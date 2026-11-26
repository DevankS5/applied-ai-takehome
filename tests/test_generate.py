from rag.generate import normalize


def test_normalize_maps_gpt_oss_citations_and_spaces():
    assert normalize("k1 is 1.2 to 2.0【2】.") == "k1 is 1.2 to 2.0[2]."
    assert normalize("Yes【1†L3-L5】【3】") == "Yes[1][3]"
    assert normalize("  plain [1] text  ") == "plain [1] text"
