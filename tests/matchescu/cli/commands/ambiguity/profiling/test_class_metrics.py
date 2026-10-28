import pytest

from matchescu.cli.commands.ambiguity.profiling._class_metrics import ClassMetrics


@pytest.mark.parametrize(
    "column, expected",
    [
        ("n_label1", 8),
        ("n_label2", 5),
        ("n_undirected", 13),
        ("comparison_space_size", 78),
        ("n_label0_implicit", 65),
        ("positive_rate", 1 / 6),
        ("neg_pos_imbalance", 5.0),
        ("match_bridge_ratio", 1.6),
    ],
)
def test_class_metrics_value(context, column, expected):
    assert ClassMetrics()(context)[column] == pytest.approx(expected)


def test_match_bridge_ratio_is_empty_without_bridges(unbridged_context):
    assert ClassMetrics()(unbridged_context)["match_bridge_ratio"] is None
