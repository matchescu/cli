from math import sqrt

import pytest

from matchescu.cli.commands.ambiguity.profiling._bridge_text_metrics import (
    BridgeTextMetrics,
)


def _entry(share_home, sim_home, sim_target, value, value_no_floor):
    return {
        "share_home": share_home,
        "sim_home": sim_home,
        "sim_target": sim_target,
        "value": value,
        "value_no_floor": value_no_floor,
        "fallback": share_home is None,
    }


BRIDGE_LOG = (
    _entry(0.6, 0.6, 0.4, "x", "x"),
    _entry(0.5, 0.5, 0.5, "y", "z"),
    _entry(None, None, None, "w", "w"),
    _entry(0.8, 0.8, 0.2, "v", "v"),
)
LINK_LOG = ({"ambiguity": 2.0}, {"ambiguity": 4.0}, {"ambiguity": 6.0})


@pytest.fixture
def logged_context(make_context):
    return make_context(bridge_log=BRIDGE_LOG, link_log=LINK_LOG)


@pytest.mark.parametrize(
    "column, expected",
    [
        ("field_share_home_median", 0.6),
        ("field_share_home_iqr", 0.15),
        ("field_sim_home_median", 0.6),
        ("field_sim_home_iqr", 0.15),
        ("field_sim_target_median", 0.4),
        ("field_sim_target_iqr", 0.15),
        ("n_fallback_values", 1),
        ("floor_changed_share", 0.25),
        ("link_ambiguity_avg", 4.0),
        ("link_ambiguity_stdev", sqrt(8 / 3)),
    ],
)
def test_bridge_text_metrics_value(logged_context, column, expected):
    assert BridgeTextMetrics()(logged_context)[column] == pytest.approx(expected)


@pytest.mark.parametrize(
    "column",
    ["field_share_home_median", "n_fallback_values", "link_ambiguity_avg"],
)
def test_bridge_text_metrics_are_empty_without_logs(context, column):
    assert BridgeTextMetrics()(context)[column] is None
