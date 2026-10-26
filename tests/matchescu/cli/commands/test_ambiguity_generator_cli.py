import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from matchescu.cli.commands import _ambiguity_generator as amg

TEST_CONFIG = {
    "inputData": [
        {
            "dataset": {
                "type": "csv",
                "directory": "./some-dataset",
                "sources": [{"fileName": "ids.csv", "idCol": "id"}],
                "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
            },
            "ambiguityTargets": ["name"],
            "secondaryFeatures": ["album"],
        }
    ]
}


@pytest.fixture
def captured_generator(monkeypatch, id_table, clusters):
    captured = {}
    fake_data = SimpleNamespace(
        id_table=id_table, true_matches={}, compute_clusters=lambda: clusters
    )

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    def fake_factory(_params, _data_dir):
        return SimpleNamespace(
            load_data=lambda: SimpleNamespace(create=lambda: fake_data)
        )

    monkeypatch.setattr(amg, "get_options", lambda _ctx: SimpleNamespace(root_dir="."))
    monkeypatch.setattr(amg, "new_benchmark_data_factory", fake_factory)
    monkeypatch.setattr(amg, "AmbiguityGenerator", StubGenerator)
    monkeypatch.setattr(amg, "_generate_amb_datasets", lambda *_args: None)
    return captured


def _invoke(tmp_path, extra_args=()):
    config_path = tmp_path / "ambiguity.json"
    config_path.write_text(json.dumps(TEST_CONFIG), encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return CliRunner().invoke(
        amg.main,
        ["-c", str(config_path), "-O", str(output_dir), *extra_args],
    )


def test_explicit_option_overrides_configured_secondary_features(
    captured_generator, tmp_path
):
    result = _invoke(tmp_path, ("--secondary-features", "year, language ,"))

    assert result.exit_code == 0
    assert captured_generator["kwargs"]["secondary_features"] == ["year", "language"]


def test_absent_option_uses_configured_secondary_features(captured_generator, tmp_path):
    result = _invoke(tmp_path)

    assert result.exit_code == 0
    assert captured_generator["kwargs"]["secondary_features"] == ["album"]
