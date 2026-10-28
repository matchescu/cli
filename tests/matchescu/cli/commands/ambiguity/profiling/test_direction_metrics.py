from statistics import fmean, pstdev

import pytest

from matchescu.cli.commands.ambiguity.profiling._direction_metrics import (
    DirectionMetrics,
)

SOURCE_SIMS = [0.9, 0.7, 0.6, 0.2, 0.4]
TARGET_SIMS = [0.3, 0.5, 0.8, 0.6, 0.4]
EXTRA_SIMS = [1.0, 0.1, 0.5]


@pytest.mark.parametrize(
    "column, expected",
    [
        ("content_auc_source_vs_target", 0.56),
        ("content_separability", 0.12),
        ("content_auc_extra_vs_target", 0.5),
        ("content_winrate_source_vs_target", 1 / 3),
        ("sim_source_avg", fmean(SOURCE_SIMS)),
        ("sim_source_stdev", pstdev(SOURCE_SIMS)),
        ("sim_target_avg", fmean(TARGET_SIMS)),
        ("sim_target_stdev", pstdev(TARGET_SIMS)),
        ("sim_extra_avg", fmean(EXTRA_SIMS)),
        ("sim_extra_stdev", pstdev(EXTRA_SIMS)),
        ("margin_pi_resolvable", 1 / 3),
    ],
)
def test_direction_metrics_value(context, column, expected):
    assert DirectionMetrics()(context)[column] == pytest.approx(expected, abs=1e-3)


def test_margin_theta_separates_margin_populations(context):
    theta = DirectionMetrics()(context)["margin_theta"]

    assert 0.1 < theta < 0.4


def test_margin_boundary_type_reports_fit_kind(context):
    assert DirectionMetrics()(context)["margin_boundary_type"] == "bayes_optimal"


def test_winrate_counts_tied_bridge_as_half(make_context):
    context = make_context(
        clusters={1: ("a1", "b1"), 2: ("e1", "b2"), 3: ("t1",), 4: ("u1",)},
        target_links=(("t1", "b1"), ("u1", "b2")),
    )

    assert DirectionMetrics()(context)[
        "content_winrate_source_vs_target"
    ] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "column", ["margin_pi_resolvable", "margin_theta", "margin_boundary_type"]
)
def test_margin_columns_are_empty_with_too_few_bridges(single_bridge_context, column):
    with pytest.warns(UserWarning, match="margin"):
        metrics = DirectionMetrics()(single_bridge_context)

    assert metrics[column] is None


@pytest.mark.parametrize(
    "column",
    [
        "content_auc_source_vs_target",
        "content_separability",
        "content_winrate_source_vs_target",
        "sim_target_avg",
    ],
)
def test_direction_columns_are_empty_without_bridges(unbridged_context, column):
    with pytest.warns(UserWarning, match="margin"):
        metrics = DirectionMetrics()(unbridged_context)

    assert metrics[column] is None
