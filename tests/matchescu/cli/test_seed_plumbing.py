from unittest.mock import MagicMock

import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation import _clustering
from matchescu.cli.config import EvaluationConfig
from matchescu.cli.config._config import ComparisonSpaceConfig
from matchescu.cli.data import _comparison_space


def _evaluation_config(**overrides):
    raw = {
        "benchmarkData": [],
        "matching": [],
        "clustering": {"algorithms": []},
        **overrides,
    }
    return EvaluationConfig.model_validate(raw)


def test_evaluation_config_seed_defaults_to_42():
    assert _evaluation_config().seed == 42


def test_evaluation_config_accepts_explicit_seed():
    assert _evaluation_config(seed=7).seed == 7


def test_new_clusterer_passes_seed_to_leiden(monkeypatch):
    mock = MagicMock(spec=_clustering.LeidenPartitioning)
    monkeypatch.setitem(_clustering.CLUSTERING_ALGOS, "LEI", mock)

    _clustering._new_clusterer("LEI", [RefId(1, "s")], 17)

    assert mock.call_count == 1
    assert mock.call_args[1]["seed"] == 17


def test_new_clusterer_passes_seed_to_spectral(monkeypatch):
    mock = MagicMock(spec=_clustering.SpectralClustering)
    monkeypatch.setitem(_clustering.CLUSTERING_ALGOS, "SC", mock)

    _clustering._new_clusterer("SC", [RefId(1, "s")], 17)

    assert mock.call_count == 1
    assert mock.call_args[1]["k_means_random_state"] == 17


class _FakeData:
    def __init__(self):
        self.name = "fake"
        self.id_table = None
        self.true_matches = {}

    def compute_clusters(self):
        return frozenset()


def test_load_comparison_space_passes_seed(tmp_path, monkeypatch):
    mock = MagicMock(spec=_comparison_space.GroundTruthComparisonSpaceGenerator)

    monkeypatch.setattr(_comparison_space, "GroundTruthComparisonSpaceGenerator", mock)

    _comparison_space.load_comparison_space(
        _FakeData(),
        tmp_path,
        ComparisonSpaceConfig(file_name="cs.csv"),
        seed=13,
    )

    assert mock.call_count == 1
    assert mock.call_args[1]["seed"] == 13


class _FakeIdRow:
    def __init__(self, ref_id):
        self.id = ref_id


class _BridgeFakeData:
    def __init__(self, ref_ids, true_matches=None):
        self.name = "fake"
        self.id_table = [_FakeIdRow(ref_id) for ref_id in ref_ids]
        self.true_matches = true_matches or {}

    def compute_clusters(self):
        return frozenset()


def test_is_bridge_matches_configured_source():
    bridge = RefId(1, "ambiguity-generator")
    ordinary = RefId(2, "dbpedia")
    data = _BridgeFakeData([bridge, ordinary])

    assert _comparison_space._is_bridge(data, "ambiguity-generator", bridge)
    assert not _comparison_space._is_bridge(data, "ambiguity-generator", ordinary)


def test_is_bridge_falls_back_to_label_two_records():
    left, right, other = RefId(1, "a"), RefId(2, "b"), RefId(3, "a")
    data = _BridgeFakeData(
        [left, right, other],
        {(left, right): 2, (left, other): 1},
    )

    assert _comparison_space._is_bridge(data, None, right)
    assert not _comparison_space._is_bridge(data, None, left)
    assert not _comparison_space._is_bridge(data, None, other)


def test_load_comparison_space_passes_bridge_predicate(tmp_path, monkeypatch):
    mock = MagicMock(spec=_comparison_space.GroundTruthComparisonSpaceGenerator)
    monkeypatch.setattr(_comparison_space, "GroundTruthComparisonSpaceGenerator", mock)
    bridge = RefId(1, "ambiguity-generator")
    data = _BridgeFakeData([bridge])

    _comparison_space.load_comparison_space(
        data,
        tmp_path,
        ComparisonSpaceConfig(file_name="cs.csv", bridge_source="ambiguity-generator"),
    )

    is_bridge = mock.call_args[1]["is_bridge"]
    assert is_bridge(bridge)
    assert not is_bridge(RefId(2, "dbpedia"))


def test_load_comparison_space_uses_label_two_bridge_predicate(tmp_path, monkeypatch):
    mock = MagicMock(spec=_comparison_space.GroundTruthComparisonSpaceGenerator)
    monkeypatch.setattr(_comparison_space, "GroundTruthComparisonSpaceGenerator", mock)
    left, right = RefId(1, "a"), RefId(2, "b")
    data = _BridgeFakeData([left, right], {(left, right): 2})

    _comparison_space.load_comparison_space(
        data, tmp_path, ComparisonSpaceConfig(file_name="cs.csv")
    )

    is_bridge = mock.call_args[1]["is_bridge"]
    assert is_bridge(right)
    assert not is_bridge(left)


def test_load_comparison_space_rejects_unknown_bridge_source(tmp_path):
    data = _BridgeFakeData([RefId(1, "dbpedia")])

    with pytest.raises(ValueError, match="does not appear"):
        _comparison_space.load_comparison_space(
            data,
            tmp_path,
            ComparisonSpaceConfig(file_name="cs.csv", bridge_source="missing"),
        )


def test_comparison_space_config_parses_bridge_source_alias():
    config = ComparisonSpaceConfig.model_validate(
        {"fileName": "cs.csv", "bridgeSource": "ambiguity-generator"}
    )

    assert config.bridge_source == "ambiguity-generator"
    assert config.model_dump(by_alias=True)["bridgeSource"] == "ambiguity-generator"
