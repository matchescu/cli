import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions, Profiler

CORE_COLUMNS = [
    "n_records",
    "n_original_records",
    "n_bridges",
    "bridge_share",
    "n_clusters",
    "n_singleton_clusters",
    "cluster_size_avg",
    "cluster_size_stdev",
    "n_home_clusters",
    "n_target_clusters",
    "n_unbridged_clusters",
    "home_clusters_per_bridge_avg",
    "home_clusters_per_bridge_stdev",
    "target_clusters_per_bridge_avg",
    "target_clusters_per_bridge_stdev",
    "bridges_per_home_cluster_avg",
    "bridges_per_home_cluster_stdev",
    "bridges_per_target_cluster_avg",
    "bridges_per_target_cluster_stdev",
    "n_label1",
    "n_label2",
    "n_undirected",
    "comparison_space_size",
    "n_label0_implicit",
    "positive_rate",
    "neg_pos_imbalance",
    "match_bridge_ratio",
    "bridge_component_count",
    "bridge_component_size_avg",
    "bridge_component_size_stdev",
    "largest_bridge_component_clusters",
    "largest_bridge_component_records",
    "largest_bridge_component_label2",
    "largest_bridge_component_split_load",
    "field_share_home_median",
    "field_share_home_iqr",
    "field_sim_home_median",
    "field_sim_home_iqr",
    "field_sim_target_median",
    "field_sim_target_iqr",
    "n_fallback_values",
    "floor_changed_share",
    "link_ambiguity_avg",
    "link_ambiguity_stdev",
    "content_auc_source_vs_target",
    "content_separability",
    "content_auc_extra_vs_target",
    "content_winrate_source_vs_target",
    "sim_source_avg",
    "sim_source_stdev",
    "sim_target_avg",
    "sim_target_stdev",
    "sim_extra_avg",
    "sim_extra_stdev",
    "margin_pi_resolvable",
    "margin_theta",
    "margin_boundary_type",
]


def test_profile_columns_follow_core_schema(profile_input):
    row = Profiler().profile(profile_input, ProfileOptions())

    assert list(row) == CORE_COLUMNS


@pytest.mark.parametrize(
    "column, expected",
    [
        pytest.param("n_bridges", 3, id="records"),
        pytest.param("match_bridge_ratio", 1.6, id="classes"),
        pytest.param("largest_bridge_component_split_load", 4.0, id="components"),
        pytest.param("content_auc_source_vs_target", 0.56, id="direction"),
    ],
)
def test_profile_holds_calculator_values(profile_input, column, expected):
    row = Profiler().profile(profile_input, ProfileOptions())

    assert row[column] == pytest.approx(expected)
