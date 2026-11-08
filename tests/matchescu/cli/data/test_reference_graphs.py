import pickle

import pytest
from matchescu.similarity import GmlGraphPersistence, MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.data import StaleReferenceGraphError, load_reference_graph


@pytest.fixture
def refs():
    return [RefId(i, "s") for i in range(5)]


@pytest.fixture
def comparison_space(refs):
    return [(refs[0], refs[1]), (refs[1], refs[2]), (refs[3], refs[2])]


@pytest.fixture
def save_graph(tmp_path):
    def _save(results: list[tuple[RefId, RefId, int]]) -> str:
        graph = ReferenceGraph(directed=True)
        for left, right, label in results:
            weights = [0.1, 0.9] if label else [0.9, 0.1]
            graph.add(MatchResult(left, right, label, weights))
        path = tmp_path / "graph.gml"
        graph.save(GmlGraphPersistence(path))
        return path

    return _save


@pytest.fixture
def matching_results(comparison_space):
    labels = [1, 0, 1]
    return [(a, b, label) for (a, b), label in zip(comparison_space, labels)]


def test_load_reference_graph_returns_graph_when_built_on_comparison_space(
    save_graph, matching_results, comparison_space
):
    graph = load_reference_graph(save_graph(matching_results), comparison_space)

    assert isinstance(graph, ReferenceGraph)


def test_load_reference_graph_keeps_stored_edges(
    save_graph, matching_results, comparison_space, refs
):
    graph = load_reference_graph(save_graph(matching_results), comparison_space)

    assert graph.has_edge(refs[0], refs[1])


def test_load_reference_graph_ignores_direction_of_comparisons(
    save_graph, matching_results, comparison_space
):
    reversed_space = [(b, a) for a, b in comparison_space]

    load_reference_graph(save_graph(matching_results), reversed_space)


def test_load_reference_graph_rejects_node_outside_comparison_space(
    save_graph, matching_results, comparison_space, refs
):
    path = save_graph([*matching_results, (refs[3], refs[4], 0)])

    with pytest.raises(StaleReferenceGraphError):
        load_reference_graph(path, comparison_space)


def test_load_reference_graph_rejects_comparison_space_node_missing_from_graph(
    save_graph, matching_results, comparison_space, refs
):
    path = save_graph(matching_results)

    with pytest.raises(StaleReferenceGraphError):
        load_reference_graph(path, [*comparison_space, (refs[3], refs[4])])


def test_load_reference_graph_rejects_edge_between_unpaired_known_nodes(
    save_graph, matching_results, comparison_space, refs
):
    path = save_graph([*matching_results, (refs[0], refs[3], 1)])

    with pytest.raises(StaleReferenceGraphError):
        load_reference_graph(path, comparison_space)


def test_stale_error_reports_graph_path(
    save_graph, matching_results, comparison_space, refs
):
    path = save_graph([*matching_results, (refs[3], refs[4], 0)])

    with pytest.raises(StaleReferenceGraphError) as error:
        load_reference_graph(path, comparison_space)

    assert str(path) in str(error.value)


def test_stale_error_reports_counts(
    save_graph, matching_results, comparison_space, refs
):
    path = save_graph([*matching_results, (refs[0], refs[3], 1)])

    with pytest.raises(StaleReferenceGraphError) as error:
        load_reference_graph(path, comparison_space)

    assert "1 unexpected edge" in str(error.value)


def test_stale_error_survives_pickling():
    error = StaleReferenceGraphError("g.gml", 1, 2, 3)

    assert str(pickle.loads(pickle.dumps(error))) == str(error)


def test_load_reference_graph_raises_when_file_is_missing(tmp_path, comparison_space):
    with pytest.raises(FileNotFoundError):
        load_reference_graph(tmp_path / "missing.gml", comparison_space)
