import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.profiling._detectability_metrics import (
    DetectabilityMetrics,
)

from .conftest import CLUSTERS

TEXT = "text"


def _texts(original_text, bridge_text):
    return {
        label: {TEXT: bridge_text if label.startswith("b") else original_text}
        for members in CLUSTERS.values()
        for label in members
    }


@pytest.fixture
def make_metrics(make_context):
    def _make(original_text, bridge_text, cv_folds=2, **input_kwargs):
        context = make_context(
            ProfileOptions(synthetic_detectability=True, cv_folds=cv_folds),
            texts=_texts(original_text, bridge_text),
            fields=(TEXT,),
            **input_kwargs,
        )
        return DetectabilityMetrics()(context)

    return _make


def test_empty_bridge_fields_give_bridges_away(make_metrics):
    metrics = make_metrics("x y", "")

    assert (metrics["detect_best_feature"], metrics["detect_auc_best_single"]) == (
        "n_empty_fields",
        pytest.approx(1.0),
    )


def test_best_single_auc_reads_shorter_bridges_as_detectable(make_metrics):
    metrics = make_metrics("x y", "z")

    assert (metrics["detect_best_feature"], metrics["detect_auc_best_single"]) == (
        "n_tokens",
        pytest.approx(1.0),
    )


def test_cross_validated_model_detects_empty_bridges(make_metrics):
    assert make_metrics("x y", "")["detect_auc_lr"] == pytest.approx(1.0)


def test_model_auc_is_empty_with_fewer_components_than_folds(make_metrics):
    with pytest.warns(UserWarning, match="detectability"):
        metrics = make_metrics("x y", "", cv_folds=5)

    assert metrics["detect_auc_lr"] is None


@pytest.mark.parametrize(
    "column", ["detect_auc_best_single", "detect_best_feature", "detect_auc_lr"]
)
def test_columns_are_empty_without_bridges(make_metrics, column):
    with pytest.warns(UserWarning, match="detectability"):
        metrics = make_metrics(
            "x y", "", clusters={1: ("a1", "a2"), 2: ("t1",)}, target_links=()
        )

    assert metrics[column] is None
