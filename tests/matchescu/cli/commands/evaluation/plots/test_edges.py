import pytest
from pyresolvemetrics import precision, recall

from matchescu.cli.commands.evaluation._plots._edges import (
    DrawnEdge,
    EdgeStatus,
    classify_edges,
    count_edges,
    merge_reciprocal,
    truth_edges,
)


def _statuses(edges: list[DrawnEdge]) -> dict[tuple, EdgeStatus]:
    return {(e.source, e.target): e.status for e in edges}


def test_unordered_classification_accepts_reversed_edge_as_true_positive(
    refs, make_graph
):
    a, b = refs[:2]

    edges = classify_edges(make_graph([(b, a)]), make_graph([(a, b)]), ordered=False)

    assert _statuses(edges) == {(b, a): EdgeStatus.TP}


def test_ordered_classification_marks_reversed_edge_as_false_positive(refs, make_graph):
    a, b = refs[:2]

    edges = classify_edges(make_graph([(b, a)]), make_graph([(a, b)]), ordered=True)

    assert _statuses(edges)[(b, a)] == EdgeStatus.FP


def test_ordered_classification_marks_missed_edge_as_false_negative(refs, make_graph):
    a, b = refs[:2]

    edges = classify_edges(make_graph([(b, a)]), make_graph([(a, b)]), ordered=True)

    assert _statuses(edges)[(a, b)] == EdgeStatus.FN


def test_unordered_classification_marks_unpredicted_pair_as_false_negative(
    refs, make_graph
):
    a, b = refs[:2]

    edges = classify_edges(make_graph([]), make_graph([(a, b)]), ordered=False)

    assert _statuses(edges) == {(a, b): EdgeStatus.FN}


def test_unordered_classification_marks_wrong_pair_as_false_positive(refs, make_graph):
    a, b, c = refs[:3]

    edges = classify_edges(make_graph([(a, c)]), make_graph([(a, b)]), ordered=False)

    assert _statuses(edges)[(a, c)] == EdgeStatus.FP


def test_merge_reciprocal_combines_same_status_opposite_edges(refs):
    a, b = refs[:2]
    edges = [
        DrawnEdge(a, b, EdgeStatus.TP),
        DrawnEdge(b, a, EdgeStatus.TP),
    ]

    assert merge_reciprocal(edges) == [DrawnEdge(a, b, EdgeStatus.TP, True)]


def test_merge_reciprocal_keeps_opposite_edges_with_different_status(refs):
    a, b = refs[:2]
    edges = [DrawnEdge(a, b, EdgeStatus.TP), DrawnEdge(b, a, EdgeStatus.FP)]

    assert merge_reciprocal(edges) == edges


def test_truth_edges_are_merged_when_bidirectional(refs, make_graph):
    a, b, c = refs[:3]

    edges = merge_reciprocal(truth_edges(make_graph([(a, b), (b, a), (a, c)])))

    assert {e.bidirectional for e in edges} == {True, False}


def test_unordered_count_counts_each_pair_once(refs, make_graph):
    a, b = refs[:2]
    edges = classify_edges(
        make_graph([(a, b), (b, a)]), make_graph([(a, b)]), ordered=False
    )

    assert count_edges(merge_reciprocal(edges), ordered=False)[EdgeStatus.TP] == 1


def test_ordered_count_counts_each_direction(refs, make_graph):
    a, b = refs[:2]
    both = make_graph([(a, b), (b, a)])

    edges = merge_reciprocal(classify_edges(both, both, ordered=True))

    assert count_edges(edges, ordered=True)[EdgeStatus.TP] == 2


@pytest.mark.parametrize("predicted", [[(0, 1), (3, 2)], [(1, 0), (2, 3), (4, 5)]])
def test_unordered_counts_reproduce_pair_metrics(refs, make_graph, predicted):
    truth_pairs = [(refs[0], refs[1]), (refs[2], refs[3])]
    pred_pairs = [(refs[i], refs[j]) for i, j in predicted]
    truth, pred = make_graph(truth_pairs), make_graph(pred_pairs)

    counts = count_edges(
        merge_reciprocal(classify_edges(pred, truth, ordered=False)), ordered=False
    )

    tp, fp, fn = (counts[s] for s in (EdgeStatus.TP, EdgeStatus.FP, EdgeStatus.FN))
    assert (tp / (tp + fp), tp / (tp + fn)) == (
        precision(set(truth_pairs), set(pred_pairs)),
        recall(set(truth_pairs), set(pred_pairs)),
    )
