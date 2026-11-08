from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from matchescu.similarity import MatchResult
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.evaluation import _matching
from matchescu.cli.commands.evaluation._cmd_group import evaluate
from matchescu.cli.commands.evaluation._parallel import UnitReporter
from matchescu.cli.commands.evaluation._plots import PlotDir

A, B = RefId(1, "s"), RefId(2, "s")


@pytest.fixture
def dataset(monkeypatch):
    data = SimpleNamespace(
        name="ds",
        true_matches={(A, B): 1},
        id_table=SimpleNamespace(
            get_all=lambda cmp: tuple(EntityReference(ref, {}) for ref in cmp)
        ),
    )
    monkeypatch.setattr(_matching, "_load_dataset", lambda *_: (data, [(A, B)]))
    monkeypatch.setattr(
        _matching,
        "new_matcher",
        lambda *_: lambda x, y: MatchResult(x.id, y.id, 1, [0.1, 0.9]),
    )
    return data


@pytest.fixture
def plot_predicted(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(_matching, "plot_predicted_graph", mock)
    return mock


def _run_unit(tmp_path, plot_root, model_type="binary", orders=("normal", "reverse")):
    model = SimpleNamespace(name="m", type=model_type)
    return _matching._eval_matching_unit(
        SimpleNamespace(),
        model,
        list(orders),
        1,
        tmp_path,
        tmp_path / "graphs",
        False,
        plot_root,
        120,
        UnitReporter("k"),
    )


def test_matching_unit_plots_one_graph_per_input_order(
    dataset, plot_predicted, tmp_path
):
    _run_unit(tmp_path, tmp_path / "plots")

    assert plot_predicted.call_count == 2


def test_matching_unit_skips_plots_without_plot_root(dataset, plot_predicted, tmp_path):
    _run_unit(tmp_path, None)

    plot_predicted.assert_not_called()


@pytest.mark.parametrize(
    "model_type, ordered", [("binary", False), ("multiclass", True)]
)
def test_matching_unit_compares_directed_edges_for_multiclass_only(
    dataset, plot_predicted, tmp_path, model_type, ordered
):
    _run_unit(tmp_path, tmp_path / "plots", model_type, ("normal",))

    assert plot_predicted.call_args.kwargs["ordered"] is ordered


def test_matching_unit_plots_into_dataset_folder(dataset, plot_predicted, tmp_path):
    _run_unit(tmp_path, tmp_path / "plots", orders=("normal",))

    assert plot_predicted.call_args.args[0] == PlotDir(tmp_path / "plots", "ds")


def test_matching_unit_passes_only_metrics_to_plot(dataset, plot_predicted, tmp_path):
    _run_unit(tmp_path, tmp_path / "plots", orders=("normal",))

    assert set(plot_predicted.call_args.args[5]) == {"precision", "recall", "f1"}


def test_ground_truth_unit_plots_comparison_space_labels(
    dataset, monkeypatch, tmp_path
):
    plot = Mock()
    monkeypatch.setattr(_matching, "plot_ground_truth_graph", plot)

    name = _matching._plot_ground_truth_unit(
        SimpleNamespace(), 5, tmp_path, tmp_path, 90, UnitReporter("k")
    )

    assert (name, plot.call_args.args) == (
        "ds",
        (PlotDir(tmp_path, "ds"), {(A, B): 1}, 5, 90),
    )


@pytest.mark.parametrize(
    "command, option",
    [("matching", "--graph-plots"), ("clustering", "--cluster-plots")],
)
def test_evaluation_commands_expose_plot_options(command, option):
    result = CliRunner().invoke(
        evaluate.commands[command],
        ["--help"],
        obj={"verbose": False, "root_dir": None, "config": None},
    )

    assert option in result.output and "--plot-dpi" in result.output


def _invoke(eval_obj, tmp_path, *extra):
    return CliRunner().invoke(
        _matching.evaluate_matching,
        ["-R", str(tmp_path), "-s", str(tmp_path / "out.csv"), "-w", "1", *extra],
        obj=eval_obj,
    )


@pytest.fixture
def command_units(monkeypatch):
    monkeypatch.setattr(_matching, "count_comparison_space_rows", lambda *_: None)
    unit = Mock(return_value=[{"dataset": "d1"}])
    truth = Mock(return_value="ds")
    monkeypatch.setattr(_matching, "_eval_matching_unit", unit)
    monkeypatch.setattr(_matching, "_plot_ground_truth_unit", truth)
    return truth, unit


def test_command_plots_ground_truth_of_every_dataset_when_plots_requested(
    eval_obj, tmp_path, command_units
):
    truth, _ = command_units
    plots = tmp_path / "plots"

    _invoke(eval_obj, tmp_path, "--graph-plots", str(plots), "--plot-dpi", "75")

    assert {call.args[3:5] for call in truth.call_args_list} == {(plots, 75)}
    assert truth.call_count == 2


def test_command_passes_plot_options_to_every_unit(eval_obj, tmp_path, command_units):
    _, unit = command_units
    plots = tmp_path / "plots"

    _invoke(eval_obj, tmp_path, "--graph-plots", str(plots), "--plot-dpi", "75")

    assert {call.args[7:9] for call in unit.call_args_list} == {(plots, 75)}


def test_command_disables_plots_by_default(eval_obj, tmp_path, command_units):
    truth, unit = command_units

    _invoke(eval_obj, tmp_path)

    assert truth.call_count == 0
    assert {call.args[7] for call in unit.call_args_list} == {None}
