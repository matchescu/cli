import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation._matching import (
    _add_normal,
    _degrade_directional_label,
    _matcher_dataset_name,
    _uses_binary_metrics,
)
from matchescu.cli.config import EvaluationConfig
from matchescu.cli.config._config import DittoConfig


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


def test_matcher_dataset_name_prefers_alias():
    cfg = _evaluation_config("amb_fake_a050_dir")

    assert _matcher_dataset_name(cfg.benchmark_data[0], "fake") == "amb_fake_a050_dir"


def test_matcher_dataset_name_falls_back_to_benchmark_name():
    cfg = _evaluation_config()

    assert _matcher_dataset_name(cfg.benchmark_data[0], "fake") == "fake"


def test_degrade_directional_label_converts_label_two(match_result):
    degraded = _degrade_directional_label(match_result)

    assert degraded.label == 1


def test_degrade_directional_label_preserves_other_labels(match_result):
    result = MatchResult(
        left=match_result.left,
        right=match_result.right,
        label=1,
        label_weights=match_result.label_weights,
    )

    assert _degrade_directional_label(result) is result


def test_degraded_label_adds_both_graph_directions(match_result):
    graph = ReferenceGraph(directed=True)

    graph.add(_degrade_directional_label(match_result))

    assert graph.has_edge(match_result.left, match_result.right)
    assert graph.has_edge(match_result.right, match_result.left)


def test_undegraded_label_adds_one_graph_direction(match_result):
    graph = ReferenceGraph(directed=True)

    graph.add(match_result)

    assert graph.has_edge(match_result.left, match_result.right)
    assert not graph.has_edge(match_result.right, match_result.left)


def test_add_normal_records_raw_label(match_result):
    left = EntityReference(match_result.left, {"name": "left"})
    right = EntityReference(match_result.right, {"name": "right"})
    graph = ReferenceGraph(directed=True)

    record = _add_normal(
        graph,
        lambda _, __: match_result,
        left,
        right,
        {},
        degrade_directional=True,
    )

    assert record["normal"] == 2


def test_multiclass_matcher_uses_binary_metrics_when_degraded():
    matcher = DittoConfig(name="bert", path="model.pt", type="multiclass")

    assert _uses_binary_metrics(matcher, True)


def test_multiclass_matcher_uses_multiclass_metrics_by_default():
    matcher = DittoConfig(name="bert", path="model.pt", type="multiclass")

    assert not _uses_binary_metrics(matcher, False)


def test_binary_matcher_uses_binary_metrics():
    matcher = DittoConfig(name="bert", path="model.pt", type="ditto")

    assert _uses_binary_metrics(matcher, False)
