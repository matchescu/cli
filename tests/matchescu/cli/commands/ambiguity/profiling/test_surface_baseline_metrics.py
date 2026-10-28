"""Surface baseline on a dataset whose direction only signed features reveal.

Each home record extends its bridge by one token; each target record drops
tokens from its bridge. Bridges alternate between 4 and 5 tokens, so the
source pair of a 4-token bridge and the target pair of a 5-token bridge have
identical symmetric features (min containment 0.8, max 1, size gap 1). The
two bridges share a target cluster, hence a component, hence a fold.
"""

import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.profiling._surface_baseline_metrics import (
    SurfaceBaselineMetrics,
)

TEXT = "text"
COMPONENTS = 4
SHORT_BRIDGE, LONG_BRIDGE = 4, 5
TARGET_CLUSTER_OFFSET = 100


def _tokens(bridge, count):
    return " ".join(f"w{bridge}_{i}" for i in range(count))


def _bridge_size(bridge):
    return SHORT_BRIDGE if bridge % 2 == 0 else LONG_BRIDGE


def _texts(bridge):
    size = _bridge_size(bridge)
    return {
        f"b{bridge}": {TEXT: _tokens(bridge, size)},
        f"h{bridge}": {TEXT: _tokens(bridge, size + 1)},
        f"t{bridge}": {TEXT: _tokens(bridge, size - 1)},
    }


def _clusters():
    homes = {k: (f"h{k}", f"b{k}") for k in range(2 * COMPONENTS)}
    targets = {
        TARGET_CLUSTER_OFFSET + j: (f"t{2 * j}", f"t{2 * j + 1}")
        for j in range(COMPONENTS)
    }
    return homes | targets


@pytest.fixture
def directional_context(make_context):
    bridges = range(2 * COMPONENTS)
    return make_context(
        ProfileOptions(surface_baseline=True, cv_folds=COMPONENTS),
        clusters=_clusters(),
        target_links=tuple((f"t{k}", f"b{k}") for k in bridges),
        texts={label: text for k in bridges for label, text in _texts(k).items()},
        fields=(TEXT,),
    )


@pytest.fixture
def metrics(directional_context):
    return SurfaceBaselineMetrics()(directional_context)


def test_full_features_recover_direction(metrics):
    assert metrics["surface_auc_full"] == pytest.approx(1.0)


def test_symmetric_features_miss_direction(metrics):
    assert metrics["surface_auc_symmetric"] < metrics["surface_auc_full"]


def test_gap_is_full_minus_symmetric(metrics):
    expected = metrics["surface_auc_full"] - metrics["surface_auc_symmetric"]

    assert metrics["surface_auc_gap"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "column", ["surface_auc_symmetric", "surface_auc_full", "surface_auc_gap"]
)
def test_columns_are_empty_with_fewer_components_than_folds(make_context, column):
    context = make_context(ProfileOptions(surface_baseline=True, cv_folds=5))

    with pytest.warns(UserWarning, match="surface baseline"):
        metrics = SurfaceBaselineMetrics()(context)

    assert metrics[column] is None
