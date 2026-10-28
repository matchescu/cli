import pytest
from pydantic import ValidationError

from matchescu.cli.commands.ambiguity._config import (
    AmbiguityConfig,
    AmbiguityDataConfig,
    ProfilingConfig,
    SelectionConfig,
    ThresholdAnalysisConfig,
)
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions, SplitRatio
from matchescu.cli.commands.ambiguity.selection import (
    SelectionSettings,
    SeparabilityBounds,
)
from matchescu.cli.config import JSONConfig

DATASET = {
    "type": "csv",
    "directory": "./some-dataset",
    "sources": [{"fileName": "ids.csv", "idCol": "id"}],
    "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
}


@pytest.fixture
def ambiguity_config_path(config_dir):
    return config_dir / "ambiguity-config.json"


@pytest.fixture
def fixture_config(ambiguity_config_path):
    return JSONConfig(ambiguity_config_path, AmbiguityConfig).load().config_obj


@pytest.fixture
def make_entry():
    def _make(**overrides):
        return AmbiguityDataConfig.model_validate(
            {"dataset": DATASET, "alphas": [0.65], **overrides}
        )

    return _make


def test_secondary_features_parses_from_camel_case_json_key(make_entry):
    sut = make_entry(secondaryFeatures=["year"])

    assert sut.secondary_features == ["year"]


def test_musicbrainz_entry_uses_year_as_secondary_feature(fixture_config):
    musicbrainz = next(
        i for i in fixture_config.input_data if "musicbrainz" in i.dataset.directory
    )

    assert musicbrainz.secondary_features == ["year"]


def test_other_dataset_entries_have_no_secondary_features(fixture_config):
    non_musicbrainz = [
        i for i in fixture_config.input_data if "musicbrainz" not in i.dataset.directory
    ]

    assert all(i.secondary_features is None for i in non_musicbrainz)


def test_alphas_parse_from_dataset_entry(make_entry):
    assert make_entry(alphas=[0.5, 0.8]).alphas == [0.5, 0.8]


def test_dataset_entry_without_alphas_raises():
    with pytest.raises(ValidationError, match="alphas"):
        AmbiguityDataConfig.model_validate({"dataset": DATASET})


@pytest.mark.parametrize(
    "alphas",
    [
        pytest.param([], id="empty"),
        pytest.param([0.0], id="zero"),
        pytest.param([0.5, 1.0], id="one"),
    ],
)
def test_invalid_alphas_raise(make_entry, alphas):
    with pytest.raises(ValidationError, match="alphas"):
        make_entry(alphas=alphas)


def test_target_floor_parses_from_camel_case_json_key(make_entry):
    assert make_entry(targetFloor=0.25).target_floor == 0.25


def test_target_links_parses_from_camel_case_json_key(make_entry):
    assert make_entry(targetLinks=2).target_links == 2


def test_dataset_entry_defaults_to_unfloored_single_link(make_entry):
    sut = make_entry()

    assert (sut.target_floor, sut.target_links) == (0.0, 1)


@pytest.mark.parametrize("floor", [-0.1, 1.1])
def test_target_floor_outside_unit_interval_raises(make_entry, floor):
    with pytest.raises(ValidationError, match="target_floor|targetFloor"):
        make_entry(targetFloor=floor)


def test_target_links_below_one_raises(make_entry):
    with pytest.raises(ValidationError, match="target_links|targetLinks"):
        make_entry(targetLinks=0)


def test_profiling_defaults_to_core_profile():
    sut = AmbiguityConfig.model_validate({"inputData": []})

    assert sut.profiling.to_profile_options() == ProfileOptions()


def test_profiling_parses_every_camel_case_key():
    sut = ProfilingConfig.model_validate(
        {
            "negativeSimilarity": True,
            "surfaceBaseline": True,
            "syntheticDetectability": True,
            "seed": 7,
            "cvFolds": 3,
            "negPosRatio": 2.5,
            "splitRatio": [8, 1, 1],
        }
    )

    assert sut.to_profile_options() == ProfileOptions(
        split_ratio=SplitRatio((8.0, 1.0, 1.0)),
        negative_similarity=True,
        neg_pos_ratio=2.5,
        surface_baseline=True,
        synthetic_detectability=True,
        seed=7,
        cv_folds=3,
    )


@pytest.mark.parametrize(
    "profiling",
    [
        pytest.param({"cvFolds": 1}, id="one-fold"),
        pytest.param({"negPosRatio": 0}, id="zero-neg-pos-ratio"),
        pytest.param({"splitRatio": [3, 0, 1]}, id="zero-split-part"),
    ],
)
def test_invalid_profiling_raises(profiling):
    with pytest.raises(ValidationError):
        ProfilingConfig.model_validate(profiling)


def test_selection_defaults_to_default_settings():
    sut = AmbiguityConfig.model_validate({"inputData": []})

    assert sut.analysis.selection.to_selection_settings() == SelectionSettings()


def test_selection_parses_every_camel_case_key():
    sut = SelectionConfig.model_validate(
        {
            "criterion": "gmm-balance",
            "minSeparability": 0.3,
            "maxAuc": 0.85,
            "targetResolvable": 0.4,
            "maxDetectAuc": 0.8,
        }
    )

    assert sut.to_selection_settings() == SelectionSettings(
        SeparabilityBounds(0.3, 0.85), "gmm-balance", 0.4, 0.8
    )


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param({"criterion": "largest"}, id="unknown-criterion"),
        pytest.param({"maxAuc": 0.5}, id="max-auc-at-chance"),
        pytest.param({"minSeparability": 0.7, "maxAuc": 0.8}, id="empty-band"),
        pytest.param({"targetResolvable": 1.5}, id="target-above-one"),
        pytest.param({"maxDetectAuc": 0.4}, id="detect-cap-below-chance"),
    ],
)
def test_invalid_selection_raises(selection):
    with pytest.raises(ValidationError):
        SelectionConfig.model_validate(selection)


def test_threshold_analysis_defaults():
    sut = AmbiguityConfig.model_validate({"inputData": []}).analysis.threshold

    assert sut == ThresholdAnalysisConfig(
        start_value=0.0, step_count=10, step_size=1.0, gmm_restarts=50
    )


def test_threshold_analysis_parses_every_camel_case_key():
    sut = ThresholdAnalysisConfig.model_validate(
        {"startValue": 2.0, "stepCount": 4, "stepSize": 0.5, "gmmRestarts": 5}
    )

    assert sut == ThresholdAnalysisConfig(
        start_value=2.0, step_count=4, step_size=0.5, gmm_restarts=5
    )


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param({"stepCount": 0}, id="no-steps"),
        pytest.param({"stepSize": 0.05}, id="step-too-small"),
        pytest.param({"gmmRestarts": 0}, id="no-restarts"),
    ],
)
def test_invalid_threshold_analysis_raises(threshold):
    with pytest.raises(ValidationError):
        ThresholdAnalysisConfig.model_validate(threshold)
