import polars as pl
import pytest

from matchescu.cli.commands.ambiguity._alpha_sweep import AlphaSweepResult
from matchescu.cli.commands.ambiguity._analyse import (
    _alpha_sweeps,
    build_methodology_rows,
    write_report,
)
from matchescu.cli.commands.ambiguity._config import AmbiguityConfig
from matchescu.cli.commands.ambiguity._gmm_fit import GmmFit
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.selection import (
    AlphaChoice,
    SelectionSettings,
    SeparabilityBounds,
)

SETTINGS = SelectionSettings(SeparabilityBounds(0.3, 0.85), "gmm-balance", 0.4)

DATASET = {
    "type": "csv",
    "directory": "./some-dataset",
    "sources": [{"fileName": "ids.csv", "idCol": "id"}],
    "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
}


def test_alpha_sweeps_raise_when_dataset_criterion_option_is_off():
    config = AmbiguityConfig.model_validate(
        {
            "inputData": [
                {
                    "dataset": DATASET,
                    "alphas": [0.5],
                    "analysis": {"selection": {"criterion": "surface-gap"}},
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="surface_baseline"):
        _alpha_sweeps(config, ProfileOptions())


@pytest.fixture
def threshold_frame():
    return pl.DataFrame(
        {
            "dataset": ["ds"],
            "threshold": [1.0],
            "bridge_count": [3],
            "cluster_count": [10],
        }
    )


@pytest.fixture
def fit():
    return GmmFit(1.234, 2.0, 0.5, 0.4, 0.5, 0.25, 0.6, GmmFit.BAYES_OPTIMAL)


@pytest.fixture
def sweep():
    rows = (
        {"dataset": "ds", "alpha": 0.5, "feasible": False, "selected": False,
         "bridge_component_count": 4, "bridge_component_size_avg": 2.3456,
         "largest_bridge_component_split_load": 1.23456},
        {"dataset": "ds", "alpha": 0.65, "feasible": True, "selected": True,
         "bridge_component_count": 4, "bridge_component_size_avg": 2.3456,
         "largest_bridge_component_split_load": 1.23456},
    )  # fmt: skip
    return AlphaSweepResult(rows, AlphaChoice(0.65, AlphaChoice.SELECTED, 1))


@pytest.fixture
def row(threshold_frame, fit, sweep):
    (methodology_row,) = build_methodology_rows(
        [threshold_frame], [fit], [3], [sweep], [SETTINGS]
    )
    return methodology_row


@pytest.mark.parametrize(
    "column, expected",
    [
        pytest.param("theta", 1.23, id="threshold-columns-kept"),
        pytest.param("alpha_criterion", "gmm-balance", id="selection-settings"),
        pytest.param("max_auc", 0.85, id="bounds"),
        pytest.param("alpha_star", 0.65, id="choice"),
        pytest.param("alpha_star_reason", "selected", id="reason"),
        pytest.param("n_feasible_alphas", 1, id="feasible-count"),
        pytest.param("bridge_component_count", 4, id="component-count"),
        pytest.param("bridge_component_size_avg", 2.35, id="rounded-float"),
        pytest.param("largest_bridge_component_split_load", 1.23, id="split-load"),
    ],
)
def test_methodology_row_value(row, column, expected):
    assert row[column] == expected


def test_methodology_row_keeps_missing_component_values_empty(row):
    assert row["largest_bridge_component_clusters"] is None


def test_methodology_rows_use_each_dataset_selection(threshold_frame, fit, sweep):
    other_settings = SelectionSettings(SeparabilityBounds(0.3, 0.85), "smallest", 0.4)
    other_frame = threshold_frame.with_columns(pl.lit("other").alias("dataset"))

    rows = build_methodology_rows(
        [threshold_frame, other_frame],
        [fit, fit],
        [3, 3],
        [sweep, sweep],
        [SETTINGS, other_settings],
    )

    assert [row["alpha_criterion"] for row in rows] == ["gmm-balance", "smallest"]


def test_report_writes_alpha_sweep_rows(threshold_frame, row, sweep, tmp_path):
    write_report([threshold_frame], [row], list(sweep.rows), tmp_path)

    written = pl.read_csv(tmp_path / "ambiguity_alpha_sweep.csv")

    assert written["alpha"].to_list() == [0.5, 0.65]


def test_report_writes_methodology_with_choice(threshold_frame, row, sweep, tmp_path):
    write_report([threshold_frame], [row], list(sweep.rows), tmp_path)

    written = pl.read_csv(tmp_path / "ambiguity_methodology.csv")

    assert written["alpha_star"].to_list() == [0.65]
