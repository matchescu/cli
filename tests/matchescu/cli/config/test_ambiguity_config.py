from pathlib import Path

import pytest

from matchescu.cli.config import AmbiguityConfig, JSONConfig
from matchescu.cli.config._config import AmbiguityDataConfig

CLI_ROOT = Path(__file__).parent.parent


@pytest.fixture
def ambiguity_config_path(config_dir):
    return config_dir / "ambiguity-config.json"


def test_secondary_features_parses_from_camel_case_json_key():
    sut = AmbiguityDataConfig.model_validate(
        {
            "dataset": {
                "type": "csv",
                "directory": "./some-dataset",
                "sources": [{"fileName": "ids.csv", "idCol": "id"}],
                "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
            },
            "ambiguityTargets": ["name"],
            "secondaryFeatures": ["year"],
        }
    )

    assert sut.secondary_features == ["year"]


def test_musicbrainz_entry_uses_year_as_secondary_feature(ambiguity_config_path):
    sut = JSONConfig(ambiguity_config_path, AmbiguityConfig).load().config_obj

    musicbrainz = next(
        i for i in sut.input_data if "musicbrainz" in i.dataset.directory
    )
    assert musicbrainz.secondary_features == ["year"]


def test_other_dataset_entries_have_no_secondary_features(ambiguity_config_path):
    sut = JSONConfig(ambiguity_config_path, AmbiguityConfig).load().config_obj

    non_musicbrainz = [
        i for i in sut.input_data if "musicbrainz" not in i.dataset.directory
    ]
    assert all(i.secondary_features is None for i in non_musicbrainz)
