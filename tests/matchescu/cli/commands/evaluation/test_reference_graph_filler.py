import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation._input_order import InputOrder
from matchescu.cli.commands.evaluation._matching import ReferenceGraphFiller

A, B = RefId(1, "s"), RefId(2, "s")

LABEL_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
LABEL_PAIR_IDS = [f"{normal}-{reverse}" for normal, reverse in LABEL_PAIRS]
RESOLVED = {
    (0, 0): (0, 0),
    (0, 1): (1, 1),
    (0, 2): (0, 2),
    (1, 0): (1, 1),
    (1, 1): (1, 1),
    (1, 2): (1, 1),
    (2, 0): (2, 0),
    (2, 1): (1, 1),
    (2, 2): (1, 1),
}
RESOLVED_WHEN_DEGRADED = {
    (0, 0): (0, 0),
    (0, 1): (1, 1),
    (1, 0): (1, 1),
    (2, 0): (1, 1),
    (0, 2): (1, 1),
}
DEGRADED_IDS = ["0-0", "0-1", "1-0", "2-0", "0-2"]
COLUMN_ENDS = {"normal": (A, B), "reverse": (B, A)}
FILLER_SETTINGS = [
    (order, degrade) for degrade in (False, True) for order in InputOrder
]
FILLER_IDS = [
    f"{order}-{'degraded' if degrade else 'kept'}" for order, degrade in FILLER_SETTINGS
]


@pytest.fixture
def graph():
    return ReferenceGraph(directed=True)


@pytest.fixture
def make_matcher():
    def _make(forward, backward, weights=(0.1, 0.2, 0.7)):
        def match(left, right):
            label = forward if left == A else backward
            return MatchResult(left, right, label, list(weights))

        return match

    return _make


@pytest.fixture(params=FILLER_SETTINGS, ids=FILLER_IDS)
def reference_graph_filler(request):
    order, degrade = request.param
    return ReferenceGraphFiller(order, degrade)


@pytest.mark.parametrize("labels", LABEL_PAIRS, ids=LABEL_PAIR_IDS)
def test_recorded_labels_match_the_graph(
    labels, graph, make_matcher, reference_graph_filler
):
    record = reference_graph_filler(graph, make_matcher(*labels), A, B, {})

    for column, (left, right) in COLUMN_ENDS.items():
        if column in record:
            assert record[column] == graph.label(left, right)


def test_filler_writes_into_the_given_record(
    graph, make_matcher, reference_graph_filler
):
    record = {"dataset": "d"}

    assert reference_graph_filler(graph, make_matcher(1, 0), A, B, record) is record


def test_normal_order_records_the_forward_label(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.normal)(
        graph, make_matcher(2, 0), A, B, {}
    )

    assert record == {"normal": 2}


def test_reverse_order_records_the_backward_label(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.reverse)(
        graph, make_matcher(0, 2), A, B, {}
    )

    assert record == {"reverse": 2}


def test_reverse_order_adds_the_backward_edge(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.reverse)(graph, make_matcher(0, 2), A, B, {})

    assert graph.has_edge(B, A)


def test_both_order_matches_in_both_orders(graph):
    calls = []

    def matcher(left, right):
        calls.append((left, right))
        return MatchResult(left, right, 0, [1.0, 0.0, 0.0])

    ReferenceGraphFiller(InputOrder.both)(graph, matcher, A, B, {})

    assert calls == [(A, B), (B, A)]


@pytest.mark.parametrize("labels", LABEL_PAIRS, ids=LABEL_PAIR_IDS)
def test_both_order_resolves_conflicting_labels(labels, graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.both)(
        graph, make_matcher(*labels), A, B, {}
    )
    expected_normal, expected_reverse = RESOLVED[labels]

    assert record["normal"] == expected_normal
    assert record["reverse"] == expected_reverse


@pytest.mark.parametrize("labels", RESOLVED_WHEN_DEGRADED, ids=DEGRADED_IDS)
def test_both_order_resolves_conflicting_labels_when_degraded(
    labels, graph, make_matcher
):
    record = ReferenceGraphFiller(InputOrder.both, degrade_directional=True)(
        graph, make_matcher(*labels), A, B, {}
    )
    expected_normal, expected_reverse = RESOLVED_WHEN_DEGRADED[labels]

    assert record["normal"] == expected_normal
    assert record["reverse"] == expected_reverse


def test_both_order_full_match_uses_the_confidence_of_the_matching_call(graph):
    def matcher(left, right):
        if left == A:
            return MatchResult(left, right, 2, [0.1, 0.2, 0.7])
        return MatchResult(left, right, 1, [0.3, 0.6, 0.1])

    ReferenceGraphFiller(InputOrder.both)(graph, matcher, A, B, {})

    assert graph.weight(A, B) == 0.6
    assert graph.weight(B, A) == 0.6


def test_degraded_directional_match_is_recorded_as_the_graphed_label(
    graph, make_matcher
):
    record = ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(2, 0), A, B, {}
    )

    assert record["normal"] == 1
    assert graph.label(A, B) == 1


def test_degraded_full_match_adds_both_edges(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(1, 0), A, B, {}
    )

    assert graph.has_edge(A, B)
    assert graph.has_edge(B, A)


@pytest.mark.parametrize(
    "weights,expected",
    [([0.1, 0.2, 0.7], 0.7 / 0.8), ([0.34, 0.33, 0.33], 0.34 / 0.67)],
    ids=["median-discarded", "extremes-renormalised"],
)
def test_degraded_directional_match_carries_the_collapsed_weight(
    weights, expected, graph, make_matcher
):
    ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(2, 0, weights=weights), A, B, {}
    )

    assert graph.weight(A, B) == pytest.approx(expected)


def test_degraded_binary_match_keeps_its_weight(graph):
    def matcher(left, right):
        return MatchResult(left, right, 1, [0.2, 0.8])

    ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, matcher, A, B, {}
    )

    assert graph.weight(A, B) == 0.8


def test_binary_matcher_adds_one_directed_edge(graph):
    def matcher(left, right):
        return MatchResult(left, right, 1, [0.2, 0.8])

    ReferenceGraphFiller(InputOrder.normal)(graph, matcher, A, B, {})

    assert not graph.has_edge(B, A)


def test_filler_creates_a_record_when_none_is_given(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.normal)(
        graph, make_matcher(1, 0), A, B, None
    )

    assert record == {"normal": 1}


def test_filler_rejects_an_unknown_input_order(graph):
    with pytest.raises(ValueError):
        ReferenceGraphFiller("sideways")(graph, lambda left, right: None, A, B, {})


def test_both_order_rejects_an_unknown_match_label(graph, make_matcher):
    with pytest.raises(ValueError):
        ReferenceGraphFiller(InputOrder.both)(graph, make_matcher(3, 0), A, B, {})
