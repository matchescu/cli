from collections.abc import Mapping

from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

BINARY_CLASS_COUNT = 2


def _one_hot(label: int, n_classes: int) -> list[float]:
    return [1.0 if idx == label else 0.0 for idx in range(n_classes)]


def build_ground_truth_graph(
    labels: Mapping[tuple[RefId, RefId], int],
) -> ReferenceGraph:
    """Build the reference graph that a perfect matcher would output.

    The graph is built the same way a matcher's output is: one ``MatchResult``
    per comparison, added to a directed ``ReferenceGraph``. Labels ``0`` and
    ``1`` give a single edge from left to right. When the labels include ``2``,
    label ``1`` gives edges in both directions and label ``2`` gives one edge
    from left to right.

    :param labels: true label of every comparison in the comparison space.
    """
    n_classes = max(max(labels.values(), default=0) + 1, BINARY_CLASS_COUNT)
    graph = ReferenceGraph(directed=True)
    for (left, right), label in labels.items():
        graph.add(MatchResult(left, right, label, _one_hot(label, n_classes)))
    return graph
