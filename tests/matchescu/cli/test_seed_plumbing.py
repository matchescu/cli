from unittest.mock import MagicMock

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
