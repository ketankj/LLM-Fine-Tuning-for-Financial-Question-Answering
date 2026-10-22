import pytest

from finqa_ft.metrics import exact_match, extract_numbers, numeric_match, score, token_f1


def test_exact_match_ignores_case_punctuation_articles():
    assert exact_match("The revenue was $4.2 billion.", "revenue was $4.2 billion") == 1.0
    assert exact_match("4.2 billion", "4.3 billion") == 0.0


def test_token_f1_partial_overlap():
    assert token_f1("revenue was 4.2 billion", "revenue was 4.2 billion") == 1.0
    assert 0 < token_f1("revenue grew to 4.2 billion", "total revenue was 4.2 billion") < 1
    assert token_f1("", "") == 1.0
    assert token_f1("abc", "") == 0.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$4.2 billion", [4.2e9]),
        ("$1,240 million", [1.24e9]),
        ("3.1%", [3.1]),
        ("a loss of (12.5)", [-12.5]),
        ("grew from 3.9 to 4.2", [3.9, 4.2]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_numeric_match_handles_scale_words():
    assert numeric_match("Revenue was $4,200 million.", "$4.2 billion") == 1.0
    assert numeric_match("Revenue was $4.5 billion.", "$4.2 billion") == 0.0


def test_numeric_match_tolerance():
    assert numeric_match("about 4.21 billion", "4.2 billion", rel_tol=0.01) == 1.0
    assert numeric_match("about 4.3 billion", "4.2 billion", rel_tol=0.01) == 0.0


def test_numeric_match_not_applicable_without_numbers():
    assert numeric_match("anything", "The context does not provide this information.") is None


def test_score_aggregates_and_skips_non_numeric():
    preds = ["$4.2 billion", "no idea", "The context does not provide this information."]
    refs = ["$4.2 billion", "8%", "The context does not provide this information."]
    s = score(preds, refs)
    assert s["n"] == 3
    assert s["exact_match"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["numeric_n"] == 2
    assert s["numeric_match"] == 0.5


def test_score_length_mismatch():
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])
