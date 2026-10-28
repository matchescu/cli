import pytest

from matchescu.cli.commands.ambiguity._similarity import (
    ngram_sim,
    tok_sim,
    weighted_similarity,
)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param("abcd", "abcd", 1.0, id="identical"),
        pytest.param("abcd", "bcde", 1 / 3, id="one-shared-trigram-of-three"),
        pytest.param("ab", "ba", 1.0, id="short-falls-back-to-char-sets"),
        pytest.param("ABCD", "abcd", 1.0, id="case-insensitive"),
    ],
)
def test_ngram_sim_returns_trigram_jaccard(a, b, expected):
    assert ngram_sim(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param("new york", "York New", 1.0, id="order-and-case-free"),
        pytest.param("a b", "b c", 1 / 3, id="one-shared-token-of-three"),
        pytest.param("", "", 0.0, id="empty"),
    ],
)
def test_tok_sim_returns_token_jaccard(a, b, expected):
    assert tok_sim(a, b) == pytest.approx(expected)


def test_weighted_similarity_of_identical_strings_is_one():
    assert weighted_similarity("main street", "Main Street") == pytest.approx(1.0)


def test_weighted_similarity_of_disjoint_strings_is_zero():
    assert weighted_similarity("abc", "xyz") == pytest.approx(0.0)
