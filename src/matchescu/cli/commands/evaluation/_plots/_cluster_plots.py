from collections.abc import Mapping

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._canvas import PlotText
from ._cluster_layout import cluster_layout
from ._cluster_render import render_predicted_clusters, render_true_clusters
from ._graph_plots import format_metrics
from ._layout import load_layout, save_layout
from ._paths import PlotDir

Clusters = frozenset[frozenset[RefId]]


def _multi_member_count(clusters: Clusters) -> int:
    return sum(1 for cluster in clusters if len(cluster) > 1)


def plot_ground_truth_clusters(plot_dir: PlotDir, clusters: Clusters, dpi: int) -> None:
    """Render the true clusters and save the layout that predictions reuse."""
    plot_dir.ensure()
    layout = cluster_layout(clusters)
    save_layout(layout, plot_dir.cluster_layout)
    text = PlotText(
        f"{plot_dir.dataset} · ground truth",
        f"{_multi_member_count(clusters)} multi-member clusters",
    )
    render_true_clusters(clusters, layout, text, plot_dir.cluster_ground_truth, dpi)


def plot_predicted_clusters(
    plot_dir: PlotDir,
    true_clusters: Clusters,
    predicted: Clusters,
    algorithm: str,
    matcher: str,
    order: str,
    metrics: Mapping[str, float],
    dpi: int,
) -> None:
    """Render ``predicted`` on the layout of the saved ground-truth picture."""
    text = PlotText(
        f"{plot_dir.dataset} · {algorithm} · {matcher} · {order}",
        f"{_multi_member_count(predicted)} multi-member clusters · "
        f"{format_metrics(metrics)}",
    )
    layout = load_layout(plot_dir.cluster_layout)
    path = plot_dir.cluster_prediction(algorithm, matcher, order)
    render_predicted_clusters(true_clusters, predicted, layout, text, path, dpi)
