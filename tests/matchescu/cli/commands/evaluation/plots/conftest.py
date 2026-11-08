import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId


@pytest.fixture
def refs():
    return [RefId(i, "s") for i in range(6)]


@pytest.fixture
def make_graph():
    def _make(edges: list[tuple[RefId, RefId]]) -> ReferenceGraph:
        graph = ReferenceGraph(directed=True)
        for left, right in edges:
            graph.add(MatchResult(left, right, 1, [0.1, 0.9]))
        return graph

    return _make
