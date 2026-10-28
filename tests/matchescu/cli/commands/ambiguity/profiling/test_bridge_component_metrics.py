import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions, SplitRatio
from matchescu.cli.commands.ambiguity.profiling._bridge_component_metrics import (
    BridgeComponentMetrics,
)


@pytest.mark.parametrize(
    "column, expected",
    [
        ("bridge_component_count", 2),
        ("bridge_component_size_avg", 3.0),
        ("bridge_component_size_stdev", 1.0),
        ("largest_bridge_component_clusters", 4),
        ("largest_bridge_component_records", 8),
        ("largest_bridge_component_label2", 4),
        ("largest_bridge_component_split_load", 4.0),
    ],
)
def test_bridge_component_metrics_value(context, column, expected):
    assert BridgeComponentMetrics()(context)[column] == pytest.approx(expected)


def test_split_load_uses_smallest_split_share(make_context):
    context = make_context(options=ProfileOptions(split_ratio=SplitRatio.parse("1,1")))

    metrics = BridgeComponentMetrics()(context)

    assert metrics["largest_bridge_component_split_load"] == pytest.approx(1.6)


def test_component_count_is_zero_without_bridges(unbridged_context):
    assert BridgeComponentMetrics()(unbridged_context)["bridge_component_count"] == 0


@pytest.mark.parametrize(
    "column",
    [
        "bridge_component_size_avg",
        "largest_bridge_component_clusters",
        "largest_bridge_component_split_load",
    ],
)
def test_component_sizes_are_empty_without_bridges(unbridged_context, column):
    assert BridgeComponentMetrics()(unbridged_context)[column] is None
