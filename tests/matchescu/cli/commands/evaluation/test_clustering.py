from types import SimpleNamespace
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation import _clustering
from matchescu.cli.commands.evaluation._parallel import UnitReporter
from matchescu.cli.commands.evaluation._plots import PlotDir


@pytest.fixture
def patch_dataset(monkeypatch):
    def _patch(name: str, cluster_map: dict):
        data = SimpleNamespace(name=name, ref_id_cluster_map=cluster_map)
        builder = Mock()
        builder.load_data.return_value.create.return_value = data
        monkeypatch.setattr(
            _clustering, "new_benchmark_data_factory", lambda *_: builder
        )
        return data

    return _patch


def _ds_config():
    return SimpleNamespace(dataset=SimpleNamespace(directory="ds"))


def test_prepare_dataset_reports_missing_cluster_ground_truth(
    patch_dataset, monkeypatch, tmp_path
):
    patch_dataset("no-clusters", {})
    monkeypatch.setattr(
        _clustering, "load_comparison_space", Mock(side_effect=Exception)
    )

    status = _clustering._prepare_dataset(
        _ds_config(), 1, tmp_path, None, 120, UnitReporter("k")
    )

    assert status == _clustering.DatasetStatus("no-clusters", False)


def test_require_cluster_ground_truth_names_every_dataset_without_clusters():
    statuses = [
        _clustering.DatasetStatus("a", False),
        _clustering.DatasetStatus("b", True),
        _clustering.DatasetStatus("c", False),
    ]

    with pytest.raises(click.ClickException, match="a, c"):
        _clustering._require_cluster_ground_truth(statuses)


def test_require_cluster_ground_truth_accepts_complete_datasets():
    _clustering._require_cluster_ground_truth([_clustering.DatasetStatus("a", True)])


@pytest.fixture
def prepared_dataset(patch_dataset, monkeypatch, refs_pair):
    patch_dataset("ds", {refs_pair[0]: 1})
    clusters = frozenset({frozenset(refs_pair)})
    monkeypatch.setattr(
        _clustering,
        "_load_dataset_context",
        lambda *_: ("ds", list(refs_pair), clusters, []),
    )
    plot = Mock()
    monkeypatch.setattr(_clustering, "plot_ground_truth_clusters", plot)
    return plot, clusters


@pytest.fixture
def refs_pair():
    return RefId(1, "s"), RefId(2, "s")


def test_prepare_dataset_plots_ground_truth_clusters_when_plot_root_given(
    prepared_dataset, tmp_path
):
    plot, clusters = prepared_dataset

    _clustering._prepare_dataset(
        _ds_config(), 1, tmp_path, tmp_path, 90, UnitReporter("k")
    )

    plot.assert_called_once_with(PlotDir(tmp_path, "ds"), clusters, 90)


def test_prepare_dataset_skips_plots_without_plot_root(prepared_dataset, tmp_path):
    plot, _ = prepared_dataset

    _clustering._prepare_dataset(_ds_config(), 1, tmp_path, None, 90, UnitReporter("k"))

    plot.assert_not_called()


@pytest.fixture
def unit_env(monkeypatch, refs_pair):
    clusters = frozenset({frozenset(refs_pair)})
    monkeypatch.setattr(_clustering, "_load_benchmark_data", lambda *_: object())
    monkeypatch.setattr(
        _clustering,
        "_load_dataset_context",
        lambda *_: ("ds", list(refs_pair), clusters, []),
    )
    monkeypatch.setattr(_clustering, "load_reference_graph", lambda *_: "graph")
    monkeypatch.setattr(
        _clustering, "_new_clusterer", lambda *_: lambda _graph: clusters
    )
    plot = Mock()
    monkeypatch.setattr(_clustering, "plot_predicted_clusters", plot)
    return plot


def _run_unit(tmp_path, plot_root):
    return _clustering._eval_clustering_unit(
        "WCC",
        _ds_config(),
        ["m1", "m2"],
        ["normal", "reverse"],
        1,
        tmp_path,
        tmp_path,
        plot_root,
        120,
        UnitReporter("k"),
    )


def test_clustering_unit_plots_every_matcher_and_order(unit_env, tmp_path):
    _run_unit(tmp_path, tmp_path)

    assert unit_env.call_count == 4


def test_clustering_unit_passes_metrics_to_plot(unit_env, tmp_path):
    _run_unit(tmp_path, tmp_path)

    metrics = unit_env.call_args.args[6]
    assert set(metrics) == {key for key, _ in _clustering.CLUSTER_METRICS}


def test_clustering_unit_skips_plots_without_plot_root(unit_env, tmp_path):
    _run_unit(tmp_path, None)

    unit_env.assert_not_called()


def _invoke(eval_obj, tmp_path, *extra):
    return CliRunner().invoke(
        _clustering.evaluate_clustering,
        ["-R", str(tmp_path), "-s", str(tmp_path / "out.csv"), "-w", "1", *extra],
        obj=eval_obj,
    )


@pytest.fixture
def command_units(monkeypatch):
    prepare = Mock(
        side_effect=lambda ds, *_: _clustering.DatasetStatus(ds_name(ds), True)
    )
    unit = Mock(return_value=[{"dataset": "d1"}])
    monkeypatch.setattr(_clustering, "_prepare_dataset", prepare)
    monkeypatch.setattr(_clustering, "_eval_clustering_unit", unit)
    return prepare, unit


def ds_name(ds_config) -> str:
    return ds_config.dataset.directory


def test_command_refuses_datasets_without_cluster_ground_truth(
    eval_obj, tmp_path, command_units
):
    prepare, unit = command_units
    prepare.side_effect = lambda ds, *_: _clustering.DatasetStatus(
        ds_name(ds), ds_name(ds) != "d2"
    )

    result = _invoke(eval_obj, tmp_path)

    assert (result.exit_code, "d2" in result.output, unit.call_count) == (1, True, 0)


def test_command_passes_plot_options_to_every_unit(eval_obj, tmp_path, command_units):
    prepare, unit = command_units
    plots = tmp_path / "plots"

    _invoke(eval_obj, tmp_path, "--cluster-plots", str(plots), "--plot-dpi", "75")

    prepared = {call.args[3:5] for call in prepare.call_args_list}
    evaluated = {call.args[7:9] for call in unit.call_args_list}
    assert prepared == evaluated == {(plots, 75)}


def test_command_disables_plots_by_default(eval_obj, tmp_path, command_units):
    _, unit = command_units

    _invoke(eval_obj, tmp_path)

    assert {call.args[7] for call in unit.call_args_list} == {None}
