from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

NGRAM_SIZE: int = 3
NGRAM_WEIGHT: float = 0.4
TOKEN_WEIGHT: float = 0.3
SEQUENCE_WEIGHT: float = 0.3


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def _ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_sim(a: str, b: str, n: int = NGRAM_SIZE) -> float:
    """Return the case-insensitive Jaccard similarity of character n-grams.

    Strings shorter than ``n`` have no n-grams, so their non-space character
    sets are compared instead.

    :param a: first string.
    :param b: second string.
    :param n: n-gram length.
    :returns: similarity in [0, 1].
    """
    a, b = a.lower(), b.lower()
    if len(a) < n or len(b) < n:
        return _jaccard(set(a.replace(" ", "")), set(b.replace(" ", "")))
    return _jaccard(_ngrams(a, n), _ngrams(b, n))


def tok_sim(a: str, b: str) -> float:
    """Return the case-insensitive Jaccard similarity of whitespace tokens.

    :param a: first string.
    :param b: second string.
    :returns: similarity in [0, 1]; two empty strings score 0.
    """
    return _jaccard(set(a.lower().split()), set(b.lower().split()))


def weighted_similarity(a: str, b: str) -> float:
    """Return the blended string similarity the ambiguity tooling relies on.

    Character n-grams, tokens and the longest-matching-blocks ratio each
    capture a different notion of closeness; their weighted sum is the score
    bridge candidates are picked by and the score dataset profiles report.

    :param a: first string.
    :param b: second string.
    :returns: similarity in [0, 1].
    """
    return (
        NGRAM_WEIGHT * ngram_sim(a, b)
        + TOKEN_WEIGHT * tok_sim(a, b)
        + SEQUENCE_WEIGHT * SequenceMatcher(None, a.lower(), b.lower()).ratio()
    )


class RecordSimilarity:
    """Score two records by their mean weighted similarity over chosen fields.

    :param fields: names of the compared fields, typically the ambiguity
        targets of a dataset.
    :raises ValueError: when `fields` is empty.
    """

    def __init__(self, fields: Sequence[str]) -> None:
        if not fields:
            raise ValueError("at least one field is required")
        self.__fields = tuple(fields)

    def __call__(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        """Return the mean weighted similarity of the two records.

        :param left: first record, as a field-to-value mapping.
        :param right: second record, as a field-to-value mapping.
        :returns: similarity in [0, 1]; missing or ``None`` values compare
            as empty strings.
        """
        total = sum(
            weighted_similarity(self.__text(left, f), self.__text(right, f))
            for f in self.__fields
        )
        return total / len(self.__fields)

    @staticmethod
    def __text(record: Mapping[str, Any], field: str) -> str:
        value = record.get(field)
        return "" if value is None else str(value)
