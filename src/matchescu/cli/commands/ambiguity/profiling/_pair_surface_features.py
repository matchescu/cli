from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ._input import ProfileInput, RecordPair
from ._token_overlap import TokenOverlap
from ._tokens import field_tokens, record_tokens

type FeatureSet = Callable[[TokenOverlap], tuple[float, ...]]


class PairSurfaceFeatures:
    """Token overlap features of record pairs, per field and per record.

    Each pair yields one overlap per compared field plus one for all
    compared fields together; a feature row concatenates the features of
    every overlap.

    :param profile_input: dataset supplying records and compared fields.
    :param pairs: ordered record pairs, one feature row each.
    """

    def __init__(self, profile_input: ProfileInput, pairs: Sequence[RecordPair]):
        records, fields = profile_input.records, profile_input.compared_fields
        self.__overlaps = [
            self.__overlaps_of(records[left], records[right], fields)
            for left, right in pairs
        ]

    def full(self) -> np.ndarray:
        """Order-aware feature matrix, one row per pair."""
        return self.__matrix(TokenOverlap.full)

    def symmetric(self) -> np.ndarray:
        """Swap-invariant feature matrix, one row per pair."""
        return self.__matrix(TokenOverlap.symmetric)

    def __matrix(self, feature_set: FeatureSet) -> np.ndarray:
        rows = [
            [x for overlap in overlaps for x in feature_set(overlap)]
            for overlaps in self.__overlaps
        ]
        return np.array(rows, dtype=float).reshape(len(rows), -1)

    @staticmethod
    def __overlaps_of(
        left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]
    ) -> list[TokenOverlap]:
        per_field = [
            TokenOverlap.of(field_tokens(left, f), field_tokens(right, f))
            for f in fields
        ]
        whole = TokenOverlap.of(
            record_tokens(left, fields), record_tokens(right, fields)
        )
        return [*per_field, whole]
