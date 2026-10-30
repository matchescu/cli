"""Alpha sweep over the single-bridge fixture.

With one source and one target pair, the content AUC is 0, 0.5 or 1, so the
separability of every alpha is exactly 0 or 1: the band [0, 1] holds every
alpha, the band [0.1, 0.8] none.
"""

import csv

import pytest

from matchescu.cli.commands.ambiguity._alpha_sweep import AlphaSweep, AlphaSweepResult
from matchescu.cli.commands.ambiguity._dataset_generation import DatasetGeneration
from matchescu.cli.commands.ambiguity._generate import AmbiguityGenerator
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.selection import (
    AlphaChoice,
    AlphaSelector,
    SelectionSettings,
    SeparabilityBounds,
)

pytestmark = pytest.mark.filterwarnings("ignore:margin mixture skipped")

LABEL = "ds-label"
EVERY_ALPHA = SeparabilityBounds(0.0, 1.0)
NO_ALPHA = SeparabilityBounds(0.1, 0.9)


@pytest.fixture
def make_generator(id_table, clusters):
    def _make(alphas=(0.5, 0.8)):
        return AmbiguityGenerator(
            id_table,
            clusters,
            ambiguity_target_properties=["name"],
            alphas=alphas,
            dataset_name="ds",
        )

    return _make


@pytest.fixture
def make_result(make_generator, id_table):
    def _make(bounds=EVERY_ALPHA, alphas=(0.5, 0.8), criteria=("smallest",)):
        selectors = [
            AlphaSelector(SelectionSettings(bounds, criterion))
            for criterion in criteria
        ]
        sweep = AlphaSweep(selectors, ProfileOptions())
        return sweep.run(make_generator(alphas), id_table, LABEL)

    return _make


def test_rows_hold_one_row_per_alpha(make_result):
    assert [row["alpha"] for row in make_result().rows] == [0.5, 0.8]


def test_rows_carry_dataset_label(make_result):
    assert {row["dataset"] for row in make_result().rows} == {LABEL}


def test_rows_carry_profile_columns(make_result):
    assert [row["n_bridges"] for row in make_result().rows] == [1, 1]


@pytest.mark.parametrize(
    "bounds, expected",
    [
        pytest.param(EVERY_ALPHA, [True, True], id="every-alpha"),
        pytest.param(NO_ALPHA, [False, False], id="no-alpha"),
    ],
)
def test_rows_flag_feasible_alphas(make_result, bounds, expected):
    assert [row["feasible"] for row in make_result(bounds).rows] == expected


def test_rows_flag_alpha_selected_per_criterion(make_result):
    result = make_result(criteria=("smallest", "gmm-balance"))

    assert [row["selected_smallest"] for row in result.rows] == [True, False]
    assert [row["selected_gmm-balance"] for row in result.rows] == [False, False]


def test_choices_hold_one_choice_per_criterion(make_result):
    result = make_result(criteria=("smallest", "gmm-balance"))

    assert result.choices["smallest"] == AlphaChoice(0.5, AlphaChoice.SELECTED, 2)
    assert result.choices["gmm-balance"] == AlphaChoice(
        None, AlphaChoice.NO_BAYES_OPTIMAL_FIT, 2
    )


def test_choice_without_feasible_alpha_is_reported(make_result):
    result = make_result(NO_ALPHA)

    assert result.choices["smallest"] == AlphaChoice(
        None, AlphaChoice.NO_FEASIBLE_ALPHA, 0
    )


def test_summary_reports_choice_per_criterion_and_bridge_components(make_result):
    summary = make_result(criteria=("smallest", "gmm-balance")).summary()

    assert summary == {
        "alpha_star_smallest": 0.5,
        "alpha_star_gmm-balance": None,
        "alpha_star_smallest_reason": AlphaChoice.SELECTED,
        "alpha_star_gmm-balance_reason": AlphaChoice.NO_BAYES_OPTIMAL_FIT,
        "n_feasible_alphas": 2,
        "bridge_component_count": 1,
        "bridge_component_size_avg": 2.0,
        "bridge_component_size_stdev": 0.0,
        "largest_bridge_component_clusters": 2,
        "largest_bridge_component_records": 3,
        "largest_bridge_component_label2": 1,
        "largest_bridge_component_split_load": 5.0,
    }


def test_summary_of_empty_sweep_leaves_component_columns_empty():
    result = AlphaSweepResult(
        (), {"smallest": AlphaChoice(None, AlphaChoice.NO_FEASIBLE_ALPHA, 0)}
    )

    assert result.summary()["largest_bridge_component_clusters"] is None


def test_sweep_rejects_criterion_lacking_profile_option():
    selector = AlphaSelector(SelectionSettings(EVERY_ALPHA, "surface-gap"))

    with pytest.raises(ValueError, match="surface_baseline"):
        AlphaSweep([selector], ProfileOptions())


@pytest.mark.parametrize(
    "column",
    [
        "n_bridges",
        "content_auc_source_vs_target",
        "sim_source_avg",
        "sim_target_avg",
        "field_share_home_median",
    ],
)
def test_sweep_profiles_alpha_like_generate(
    make_result, make_generator, id_table, tmp_path, column
):
    swept = make_result(alphas=(0.5, 0.65, 0.95)).rows[1]
    DatasetGeneration(tmp_path, ProfileOptions()).run(
        make_generator(alphas=(0.65,)), id_table
    )
    with (tmp_path / "metadata.csv").open(newline="") as handle:
        (generated,) = csv.DictReader(handle)

    assert swept[column] == pytest.approx(float(generated[column]))
