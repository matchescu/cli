import matplotlib.image as mpimg
import pytest

from matchescu.cli.commands.evaluation._plots._canvas import FIG_WIDTH_IN, PlotText
from matchescu.cli.commands.evaluation._plots._cluster_layout import cluster_layout
from matchescu.cli.commands.evaluation._plots._cluster_render import (
    render_predicted_clusters,
    render_true_clusters,
)
from matchescu.cli.commands.evaluation._plots._edges import (
    classify_edges,
    merge_reciprocal,
    truth_edges,
)
from matchescu.cli.commands.evaluation._plots._graph_layout import graph_layout
from matchescu.cli.commands.evaluation._plots._graph_render import render_graph

DPI = 50
TEXT = PlotText("title", "footnote")


@pytest.fixture
def truth(refs, make_graph):
    return make_graph([(refs[0], refs[1]), (refs[1], refs[0]), (refs[2], refs[3])])


@pytest.fixture
def predicted(refs, make_graph):
    return make_graph([(refs[0], refs[1]), (refs[3], refs[4])])


def _width(path) -> int:
    return mpimg.imread(path).shape[1]


def test_graph_render_writes_png_of_requested_resolution(truth, tmp_path):
    path = tmp_path / "truth.png"
    edges = merge_reciprocal(truth_edges(truth))

    render_graph(edges, graph_layout(truth, 1), TEXT, path, DPI)

    assert _width(path) == FIG_WIDTH_IN * DPI


def test_graph_render_handles_prediction_with_every_edge_status(
    truth, predicted, tmp_path
):
    path = tmp_path / "pred.png"
    layout = graph_layout(truth, 1)
    layout.update(graph_layout(predicted, 1))
    edges = merge_reciprocal(classify_edges(predicted, truth, ordered=True))

    render_graph(edges, layout, TEXT, path, DPI)

    assert path.is_file()


@pytest.fixture
def true_clusters(refs):
    return frozenset({frozenset(refs[:3]), frozenset(refs[3:5]), frozenset(refs[5:])})


def test_true_cluster_render_writes_png_of_requested_resolution(
    true_clusters, tmp_path
):
    path = tmp_path / "clusters.png"

    render_true_clusters(true_clusters, cluster_layout(true_clusters), TEXT, path, DPI)

    assert _width(path) == FIG_WIDTH_IN * DPI


def test_predicted_cluster_render_writes_png(true_clusters, refs, tmp_path):
    path = tmp_path / "pred.png"
    predicted = frozenset({frozenset(refs[:2]), frozenset(refs[2:6])})

    render_predicted_clusters(
        true_clusters, predicted, cluster_layout(true_clusters), TEXT, path, DPI
    )

    assert path.is_file()


def test_cluster_render_handles_collinear_clusters(refs, tmp_path):
    path = tmp_path / "collinear.png"
    clusters = frozenset({frozenset(refs[:3])})
    layout = {ref: (float(i), 0.0) for i, ref in enumerate(refs[:3])}

    render_true_clusters(clusters, layout, TEXT, path, DPI)

    assert path.is_file()


def test_predicted_cluster_render_handles_unmatched_prediction(refs, tmp_path):
    path = tmp_path / "unmatched.png"
    truth = frozenset({frozenset(refs[:2]), frozenset(refs[2:3]), frozenset(refs[3:4])})
    predicted = frozenset({frozenset(refs[:2]), frozenset(refs[2:4])})

    render_predicted_clusters(truth, predicted, cluster_layout(truth), TEXT, path, DPI)

    assert path.is_file()
