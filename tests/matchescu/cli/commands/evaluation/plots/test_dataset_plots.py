import matplotlib.image as mpimg
import pytest

from matchescu.cli.commands.evaluation._plots import (
    PlotDir,
    plot_ground_truth_clusters,
    plot_ground_truth_graph,
    plot_predicted_clusters,
    plot_predicted_graph,
)

DPI = 40
SEED = 3


@pytest.fixture
def plot_dir(tmp_path):
    return PlotDir(tmp_path, "ds")


@pytest.fixture
def labels(refs):
    return {(refs[0], refs[1]): 1, (refs[2], refs[3]): 1, (refs[1], refs[2]): 0}


def test_plot_dir_groups_files_under_dataset_folder(plot_dir, tmp_path):
    assert plot_dir.graph_prediction("m", "normal").parent == tmp_path / "ds"


def test_plot_dir_names_prediction_files_after_their_parameters(plot_dir):
    assert plot_dir.cluster_prediction("WCC", "m", "both").name == (
        "clusters-WCC-m-both.png"
    )


def test_ground_truth_graph_plot_writes_image_and_layout(plot_dir, labels):
    plot_ground_truth_graph(plot_dir, labels, SEED, DPI)

    assert plot_dir.graph_ground_truth.is_file() and plot_dir.graph_layout.is_file()


def test_predicted_graph_plot_reuses_saved_layout(plot_dir, labels, make_graph, refs):
    plot_ground_truth_graph(plot_dir, labels, SEED, DPI)
    graph = make_graph([(refs[0], refs[1]), (refs[1], refs[2])])

    plot_predicted_graph(
        plot_dir, graph, labels, "m", "normal", {"f1": 0.5}, ordered=False, dpi=DPI
    )

    assert plot_dir.graph_prediction("m", "normal").is_file()


def test_predicted_graph_plot_matches_ground_truth_image_size(
    plot_dir, labels, make_graph, refs
):
    plot_ground_truth_graph(plot_dir, labels, SEED, DPI)
    graph = make_graph([(refs[0], refs[1])])

    plot_predicted_graph(
        plot_dir, graph, labels, "m", "normal", {"f1": 0.5}, ordered=False, dpi=DPI
    )

    truth_shape = mpimg.imread(plot_dir.graph_ground_truth).shape
    assert mpimg.imread(plot_dir.graph_prediction("m", "normal")).shape == truth_shape


def test_ground_truth_cluster_plot_writes_image_and_layout(plot_dir, refs):
    clusters = frozenset({frozenset(refs[:2]), frozenset(refs[2:])})

    plot_ground_truth_clusters(plot_dir, clusters, DPI)

    assert plot_dir.cluster_ground_truth.is_file() and plot_dir.cluster_layout.is_file()


def test_predicted_cluster_plot_writes_image(plot_dir, refs):
    clusters = frozenset({frozenset(refs[:2]), frozenset(refs[2:])})
    plot_ground_truth_clusters(plot_dir, clusters, DPI)

    plot_predicted_clusters(
        plot_dir, clusters, clusters, "WCC", "m", "normal", {"ari": 1.0}, DPI
    )

    assert plot_dir.cluster_prediction("WCC", "m", "normal").is_file()
