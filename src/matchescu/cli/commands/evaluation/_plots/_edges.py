from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from matchescu.similarity import ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

Edge = tuple[RefId, RefId]


class EdgeStatus(StrEnum):
    TRUTH = "truth"
    TP = "tp"
    FP = "fp"
    FN = "fn"


@dataclass(frozen=True)
class DrawnEdge:
    """An edge to draw. ``bidirectional`` edges stand for both directions."""

    source: RefId
    target: RefId
    status: EdgeStatus
    bidirectional: bool = False


def truth_edges(truth: ReferenceGraph) -> list[DrawnEdge]:
    return [DrawnEdge(u, v, EdgeStatus.TRUTH) for u, v in truth.matches()]


def _classify(
    predicted: ReferenceGraph, truth: ReferenceGraph, key: Callable[[Edge], object]
) -> list[DrawnEdge]:
    truth_keys = {key(edge) for edge in truth.matches()}
    predicted_keys = {key(edge) for edge in predicted.matches()}
    found = [
        DrawnEdge(u, v, EdgeStatus.TP if key((u, v)) in truth_keys else EdgeStatus.FP)
        for u, v in predicted.matches()
    ]
    missed = [
        DrawnEdge(u, v, EdgeStatus.FN)
        for u, v in truth.matches()
        if key((u, v)) not in predicted_keys
    ]
    return found + missed


def classify_edges(
    predicted: ReferenceGraph, truth: ReferenceGraph, ordered: bool
) -> list[DrawnEdge]:
    """Label every edge of ``predicted`` and every missed edge of ``truth``.

    ``ordered=False`` ignores edge direction, like the binary precision and
    recall metrics. ``ordered=True`` compares exact directed edges, like the
    multiclass metric.
    """
    return _classify(predicted, truth, tuple if ordered else frozenset)


def merge_reciprocal(edges: list[DrawnEdge]) -> list[DrawnEdge]:
    """Replace opposite edges of equal status by one bidirectional edge."""
    available = set(edges)
    merged, consumed = [], set()
    for edge in edges:
        if edge in consumed:
            continue
        reverse = DrawnEdge(edge.target, edge.source, edge.status)
        if reverse not in available:
            merged.append(edge)
            continue
        consumed.add(reverse)
        merged.append(replace(edge, bidirectional=True))
    return merged


def count_edges(edges: list[DrawnEdge], ordered: bool) -> Counter[EdgeStatus]:
    """Count merged edges per status; a bidirectional edge is one pair."""
    counts: Counter[EdgeStatus] = Counter()
    for edge in edges:
        counts[edge.status] += 2 if ordered and edge.bidirectional else 1
    return counts
