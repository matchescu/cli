from collections import Counter

import numpy as np
from matchescu.typing import EntityReferenceIdentifier as RefId
from scipy.optimize import linear_sum_assignment

from ._layout import sorted_clusters

Cluster = frozenset[RefId]


def _overlaps(
    true_clusters: list[list[RefId]], predicted: list[list[RefId]]
) -> np.ndarray:
    owner = {ref: idx for idx, members in enumerate(true_clusters) for ref in members}
    matrix = np.zeros((len(predicted), len(true_clusters)), dtype=np.int32)
    for row, members in enumerate(predicted):
        counts = Counter(owner[ref] for ref in members if ref in owner)
        for col, count in counts.items():
            matrix[row, col] = count
    return matrix


def match_clusters(
    true_clusters: frozenset[Cluster], predicted: frozenset[Cluster]
) -> dict[Cluster, Cluster | None]:
    """Pair predicted clusters with true clusters by maximum total overlap.

    The pairing is one-to-one (Hungarian assignment) and ignores singletons.
    Predicted clusters left without a partner map to ``None``.
    """
    truth = [m for m in sorted_clusters(true_clusters) if len(m) > 1]
    guesses = [m for m in sorted_clusters(predicted) if len(m) > 1]
    result: dict[Cluster, Cluster | None] = {c: None for c in predicted}
    if not truth or not guesses:
        return result
    overlaps = _overlaps(truth, guesses)
    for row, col in zip(*linear_sum_assignment(overlaps, maximize=True)):
        if overlaps[row, col] > 0:
            result[frozenset(guesses[row])] = frozenset(truth[col])
    return result
