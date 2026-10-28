from collections.abc import Mapping, Sequence
from typing import Any

from ._similarity import weighted_similarity


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
