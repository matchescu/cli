from collections.abc import Mapping

from matchescu.similarity import ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._canvas import PlotText
from ._edges import (
    EdgeStatus,
    classify_edges,
    count_edges,
    merge_reciprocal,
    truth_edges,
)
from ._graph_layout import graph_layout
from ._graph_render import render_graph
from ._ground_truth import build_ground_truth_graph
from ._layout import load_layout, save_layout
from ._paths import PlotDir

Labels = Mapping[tuple[RefId, RefId], int]
METRIC_PRECISION = 3


def format_metrics(metrics: Mapping[str, float]) -> str:
    return " · ".join(
        f"{name} {value:.{METRIC_PRECISION}f}" for name, value in metrics.items()
    )


def plot_ground_truth_graph(plot_dir: PlotDir, labels: Labels, seed: int, dpi: int):
    """Render the ground-truth graph and save the layout that predictions reuse."""
    plot_dir.ensure()
    truth = build_ground_truth_graph(labels)
    layout = graph_layout(truth, seed)
    save_layout(layout, plot_dir.graph_layout)
    edges = merge_reciprocal(truth_edges(truth))
    count = count_edges(edges, ordered=True)[EdgeStatus.TRUTH]
    text = PlotText(f"{plot_dir.dataset} · ground truth", f"{count} directed edges")
    render_graph(edges, layout, text, plot_dir.graph_ground_truth, dpi)


def plot_predicted_graph(
    plot_dir: PlotDir,
    graph: ReferenceGraph,
    labels: Labels,
    model: str,
    order: str,
    metrics: Mapping[str, float],
    ordered: bool,
    dpi: int,
) -> None:
    """Render ``graph`` against the ground truth on the saved layout.

    :param ordered: compare directed edges exactly (multiclass models) instead
        of ignoring direction (binary models), mirroring the stored metrics.
    """
    truth = build_ground_truth_graph(labels)
    edges = merge_reciprocal(classify_edges(graph, truth, ordered))
    counts = count_edges(edges, ordered)
    summary = " · ".join(f"{s.upper()} {counts[s]}" for s in EdgeStatus if s != "truth")
    text = PlotText(
        f"{plot_dir.dataset} · {model} · {order}",
        f"{format_metrics(metrics)} · {summary}",
    )
    layout = load_layout(plot_dir.graph_layout)
    render_graph(edges, layout, text, plot_dir.graph_prediction(model, order), dpi)
