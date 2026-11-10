import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation._input_order import InputOrder
from matchescu.cli.commands.evaluation._matching import ReferenceGraphFiller

LEFT = EntityReference(RefId(1, "s"), {})
RIGHT = EntityReference(RefId(2, "s"), {})
A, B = LEFT.id, RIGHT.id

LABEL_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
LABEL_PAIR_IDS = [f"{normal}-{reverse}" for normal, reverse in LABEL_PAIRS]
GRAPH_LABELS = {
    (0, 0): (0, 0),
    (0, 1): (1, 1),
    (0, 2): (0, 2),
    (1, 0): (1, 1),
    (1, 1): (1, 1),
    (1, 2): (1, 2),
    (2, 0): (2, 0),
    (2, 1): (1, 1),
    (2, 2): (2, 2),
}
DEGRADED_LABELS = {
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
BINARY_WEIGHTS = (0.2, 0.8)


@pytest.fixture
def graph():
    return ReferenceGraph(directed=True)


@pytest.fixture
def make_matcher():
    def _make(forward, backward, weights=(0.1, 0.2, 0.7)):
        def match(left, right):
            label = forward if left == LEFT else backward
            return MatchResult(left.id, right.id, label, list(weights))

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
    record = reference_graph_filler(graph, make_matcher(*labels), LEFT, RIGHT, {})

    for column, (left, right) in COLUMN_ENDS.items():
        if column in record:
            assert record[column] == graph.label(left, right)


def test_filler_writes_into_the_given_record(
    graph, make_matcher, reference_graph_filler
):
    record = {"dataset": "d"}

    assert (
        reference_graph_filler(graph, make_matcher(1, 0), LEFT, RIGHT, record) is record
    )


def test_normal_order_records_the_forward_label(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.normal)(
        graph, make_matcher(2, 0), LEFT, RIGHT, {}
    )

    assert record == {"normal": 2}


def test_reverse_order_records_the_backward_label(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.reverse)(
        graph, make_matcher(0, 2), LEFT, RIGHT, {}
    )

    assert record == {"reverse": 2}


def test_reverse_order_adds_the_backward_edge(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.reverse)(graph, make_matcher(0, 2), LEFT, RIGHT, {})

    assert graph.has_edge(B, A)


def test_both_order_matches_in_both_orders(graph):
    calls = []

    def matcher(left, right):
        calls.append((left, right))
        return MatchResult(left.id, right.id, 0, [1.0, 0.0, 0.0])

    ReferenceGraphFiller(InputOrder.both)(graph, matcher, LEFT, RIGHT, {})

    assert calls == [(LEFT, RIGHT), (RIGHT, LEFT)]


@pytest.mark.parametrize("labels", LABEL_PAIRS, ids=LABEL_PAIR_IDS)
def test_both_order_leaves_conflicts_to_the_reference_graph(
    labels, graph, make_matcher
):
    record = ReferenceGraphFiller(InputOrder.both)(
        graph, make_matcher(*labels), LEFT, RIGHT, {}
    )
    expected_normal, expected_reverse = GRAPH_LABELS[labels]

    assert record["normal"] == expected_normal
    assert record["reverse"] == expected_reverse


@pytest.mark.parametrize("labels", DEGRADED_LABELS, ids=DEGRADED_IDS)
def test_both_order_resolves_conflicts_when_degraded(labels, graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.both, degrade_directional=True)(
        graph, make_matcher(*labels), LEFT, RIGHT, {}
    )
    expected_normal, expected_reverse = DEGRADED_LABELS[labels]

    assert record["normal"] == expected_normal
    assert record["reverse"] == expected_reverse


def test_degraded_both_order_uses_the_confidence_of_the_matching_direction(graph):
    def matcher(left, right):
        if left == LEFT:
            return MatchResult(left.id, right.id, 0, [0.9, 0.1, 0.0])
        return MatchResult(left.id, right.id, 2, [0.1, 0.2, 0.7])

    ReferenceGraphFiller(InputOrder.both, degrade_directional=True)(
        graph, matcher, LEFT, RIGHT, {}
    )

    assert graph.weight(A, B) == pytest.approx(0.875)
    assert graph.weight(B, A) == pytest.approx(0.875)


def test_degraded_directional_match_is_recorded_as_the_graphed_label(
    graph, make_matcher
):
    record = ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(2, 0), LEFT, RIGHT, {}
    )

    assert record["normal"] == 1
    assert graph.label(A, B) == 1


def test_degraded_full_match_adds_both_edges(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(1, 0), LEFT, RIGHT, {}
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
        graph, make_matcher(2, 0, weights=weights), LEFT, RIGHT, {}
    )

    assert graph.weight(A, B) == pytest.approx(expected)


def test_degraded_binary_match_keeps_its_weight(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.normal, degrade_directional=True)(
        graph, make_matcher(1, 0, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert graph.weight(A, B) == 0.8


def test_binary_matcher_adds_one_directed_edge(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.normal)(
        graph, make_matcher(1, 0, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert not graph.has_edge(B, A)


def test_binary_matcher_both_order_keeps_the_forward_edge(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.both)(
        graph, make_matcher(1, 0, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert graph.has_edge(A, B)
    assert not graph.has_edge(B, A)


def test_binary_matcher_both_order_keeps_the_backward_edge(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.both)(
        graph, make_matcher(0, 1, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert graph.has_edge(B, A)
    assert not graph.has_edge(A, B)


def test_binary_matcher_both_order_records_the_raw_labels(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.both)(
        graph, make_matcher(1, 0, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert record["normal"] == 1
    assert record["reverse"] == 0


def test_binary_matcher_ignores_degradation(graph, make_matcher):
    ReferenceGraphFiller(InputOrder.both, degrade_directional=True)(
        graph, make_matcher(1, 0, weights=BINARY_WEIGHTS), LEFT, RIGHT, {}
    )

    assert not graph.has_edge(B, A)


def test_filler_creates_a_record_when_none_is_given(graph, make_matcher):
    record = ReferenceGraphFiller(InputOrder.normal)(
        graph, make_matcher(1, 0), LEFT, RIGHT, None
    )

    assert record == {"normal": 1}


def test_filler_rejects_an_unknown_input_order(graph):
    with pytest.raises(ValueError):
        ReferenceGraphFiller("sideways")(
            graph, lambda left, right: None, LEFT, RIGHT, {}
        )
