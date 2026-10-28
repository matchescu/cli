from math import sqrt

import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.profiling._negative_similarity_metrics import (
    NegativeSimilarityMetrics,
)

POSITIVES = 13
IMPLICIT_NEGATIVES = 65
EXHAUSTIVE_RATIO = 10.0


@pytest.fixture
def make_metrics(make_context):
    def _make(neg_pos_ratio, seed=42, **input_kwargs):
        options = ProfileOptions(
            negative_similarity=True, neg_pos_ratio=neg_pos_ratio, seed=seed
        )
        return NegativeSimilarityMetrics()(make_context(options, **input_kwargs))

    return _make


@pytest.mark.parametrize(
    "column, expected",
    [
        ("neg_sample_size", IMPLICIT_NEGATIVES),
        ("sim_label0_avg", 0.95 / IMPLICIT_NEGATIVES),
        ("sim_label0_stdev", 0.95 * sqrt(IMPLICIT_NEGATIVES - 1) / IMPLICIT_NEGATIVES),
        ("content_auc_label1_vs_label0", 513 / 520),
        ("content_auc_label2_vs_label0", 320 / 325),
    ],
)
def test_ratio_beyond_population_covers_every_implicit_negative(
    make_metrics, column, expected
):
    metrics = make_metrics(EXHAUSTIVE_RATIO)

    assert metrics[column] == pytest.approx(expected)


@pytest.mark.parametrize(
    "neg_pos_ratio, expected",
    [
        pytest.param(1.0, POSITIVES, id="one-to-one"),
        pytest.param(4.0, 4 * POSITIVES, id="four-to-one"),
        pytest.param(0.5, round(0.5 * POSITIVES), id="fraction-rounds"),
    ],
)
def test_sample_size_is_ratio_times_positives(make_metrics, neg_pos_ratio, expected):
    assert make_metrics(neg_pos_ratio)["neg_sample_size"] == expected


def test_same_seed_draws_same_sample(make_metrics):
    assert make_metrics(1.0, seed=7) == make_metrics(1.0, seed=7)


def test_label2_auc_is_empty_without_bridges(make_metrics):
    metrics = make_metrics(
        EXHAUSTIVE_RATIO, clusters={1: ("a1", "a2"), 2: ("t2",)}, target_links=()
    )

    assert metrics["content_auc_label2_vs_label0"] is None
