import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation._plots import build_ground_truth_graph

A, B, C = (RefId(i, "s") for i in range(3))


@pytest.fixture
def binary_labels():
    return {(A, B): 1, (B, C): 0}


@pytest.fixture
def directional_labels():
    return {(A, B): 1, (A, C): 2, (B, C): 0}


def test_binary_ground_truth_has_single_edge_for_match(binary_labels):
    graph = build_ground_truth_graph(binary_labels)

    assert sorted(graph.matches()) == [(A, B)]


def test_binary_ground_truth_contains_non_matching_nodes(binary_labels):
    graph = build_ground_truth_graph(binary_labels)

    assert set(graph.nodes) == {A, B, C}


def test_directional_ground_truth_is_bidirectional_for_match(directional_labels):
    graph = build_ground_truth_graph(directional_labels)

    assert graph.has_edge(A, B) and graph.has_edge(B, A)


def test_directional_ground_truth_is_one_way_for_direction_label(directional_labels):
    graph = build_ground_truth_graph(directional_labels)

    assert graph.has_edge(A, C) and not graph.has_edge(C, A)


def test_ground_truth_graph_is_directed(binary_labels):
    assert build_ground_truth_graph(binary_labels).directed
