from collections.abc import Iterable
from os import PathLike

from matchescu.similarity import GmlGraphPersistence, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


class StaleReferenceGraphError(Exception):
    """A saved reference graph was built on a different comparison space."""

    def __init__(
        self,
        path: str | PathLike,
        unexpected_nodes: int,
        missing_nodes: int,
        unexpected_edges: int,
    ) -> None:
        super().__init__(path, unexpected_nodes, missing_nodes, unexpected_edges)
        self.path = path
        self.unexpected_nodes = unexpected_nodes
        self.missing_nodes = missing_nodes
        self.unexpected_edges = unexpected_edges

    def __str__(self) -> str:
        return (
            "These REALLY are not the droids you're looking for.\n"
            f"{self.path} was built on a different comparison space than the "
            "one in your config "
            f"({_plural(self.unexpected_nodes, 'unexpected node')}, "
            f"{_plural(self.missing_nodes, 'missing node')}, "
            f"{_plural(self.unexpected_edges, 'unexpected edge')})."
        )


def _unordered(pairs: Iterable[tuple[RefId, RefId]]) -> set[frozenset[RefId]]:
    return {frozenset(pair) for pair in pairs}


def load_reference_graph(
    path: str | PathLike, comparison_space: Iterable[tuple[RefId, RefId]]
) -> ReferenceGraph:
    """Load a saved reference graph, making sure it fits a comparison space.

    A graph built on a comparison space contains exactly the references that
    occur in it, and its edges are comparisons of that space. Anything else
    means the file was produced for a different comparison space. The graph is
    never repaired: a stale graph must be recomputed.

    :param path: GML file holding the reference graph.
    :param comparison_space: the comparisons the graph is expected to cover.
        Direction is ignored.
    :raises StaleReferenceGraphError: the graph doesn't fit the comparison space.
    """
    pairs = list(comparison_space)
    graph = ReferenceGraph.load(GmlGraphPersistence(path))
    expected_nodes = {ref for pair in pairs for ref in pair}
    nodes = set(graph.nodes)
    allowed_edges = _unordered(pairs)
    unexpected_edges = _unordered((u, v) for u, v, _ in graph.edges) - allowed_edges
    stale = (
        len(nodes - expected_nodes),
        len(expected_nodes - nodes),
        len(unexpected_edges),
    )
    if any(stale):
        raise StaleReferenceGraphError(path, *stale)
    return graph
