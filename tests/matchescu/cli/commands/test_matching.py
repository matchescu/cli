import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.config import EvaluationConfig


def _evaluation_config(dataset_alias: str | None = None) -> EvaluationConfig:
    benchmark = {
        "dataset": {
            "type": "csv",
            "directory": "./data/fake",
            "sources": [{"fileName": "fake_ids.csv"}],
            "pairwiseMapping": {"fileName": "fake_mapping.csv"},
        },
        "comparisonSpace": {"fileName": "runtime-cs.csv"},
    }
    if dataset_alias is not None:
        benchmark["datasetAlias"] = dataset_alias
    return EvaluationConfig.model_validate({"benchmarkData": [benchmark]})


@pytest.fixture
def match_result() -> MatchResult:
    return MatchResult(
        left=RefId(1, "source"),
        right=RefId(2, "source"),
        label=2,
        label_weights=[0.1, 0.2, 0.7],
    )


def test_evaluation_config_reads_dataset_alias():
    cfg = _evaluation_config("amb_fake_a050_dir")

    assert cfg.benchmark_data[0].dataset_alias == "amb_fake_a050_dir"


def test_undegraded_label_adds_one_graph_direction(match_result):
    graph = ReferenceGraph(directed=True)

    graph.add(match_result)

    assert graph.has_edge(match_result.left, match_result.right)
    assert not graph.has_edge(match_result.right, match_result.left)
