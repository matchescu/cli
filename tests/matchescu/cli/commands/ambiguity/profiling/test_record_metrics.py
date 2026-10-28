from math import sqrt

import pytest

from matchescu.cli.commands.ambiguity.profiling._record_metrics import RecordMetrics


@pytest.mark.parametrize(
    "column, expected",
    [
        ("n_records", 13),
        ("n_original_records", 10),
        ("n_bridges", 3),
        ("bridge_share", 3 / 13),
        ("n_clusters", 7),
        ("n_singleton_clusters", 3),
        ("cluster_size_avg", 13 / 7),
        ("cluster_size_stdev", sqrt(34) / 7),
        ("n_home_clusters", 3),
        ("n_target_clusters", 3),
        ("n_unbridged_clusters", 1),
        ("home_clusters_per_bridge_avg", 1.0),
        ("home_clusters_per_bridge_stdev", 0.0),
        ("target_clusters_per_bridge_avg", 4 / 3),
        ("target_clusters_per_bridge_stdev", sqrt(2) / 3),
        ("bridges_per_home_cluster_avg", 3 / 7),
        ("bridges_per_home_cluster_stdev", sqrt(12) / 7),
        ("bridges_per_target_cluster_avg", 4 / 7),
        ("bridges_per_target_cluster_stdev", sqrt(26) / 7),
    ],
)
def test_record_metrics_value(context, column, expected):
    assert RecordMetrics()(context)[column] == pytest.approx(expected)


@pytest.mark.parametrize(
    "column", ["target_clusters_per_bridge_avg", "home_clusters_per_bridge_stdev"]
)
def test_per_bridge_spread_is_empty_without_bridges(unbridged_context, column):
    assert RecordMetrics()(unbridged_context)[column] is None


def test_bridges_per_cluster_counts_every_cluster_without_bridges(
    unbridged_context,
):
    metrics = RecordMetrics()(unbridged_context)

    assert metrics["bridges_per_home_cluster_avg"] == pytest.approx(0.0)
