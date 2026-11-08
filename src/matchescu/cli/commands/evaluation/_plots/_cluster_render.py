import colorsys
from collections.abc import Iterable
from os import PathLike

import numpy as np
from matchescu.typing import EntityReferenceIdentifier as RefId
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.spatial import ConvexHull, QhullError

from ._canvas import (
    LEGEND_ANCHOR,
    SIDE_INSET_RECT,
    Canvas,
    PlotText,
    annotate,
    new_canvas,
    save_canvas,
)
from ._cluster_match import Cluster, match_clusters
from ._layout import Layout, sorted_clusters

GOLDEN_RATIO_CONJUGATE = 0.618033988749895
SATURATIONS = (0.9, 0.6)
PALETTE_VALUE = 0.85
UNMATCHED_COLOR = to_rgb("#9e9e9e")
WRONG_OUTLINE = "#d32f2f"
PREDICTION_COLOR = "#1565c0"
NODE_AREA = 18.0
NODE_OUTLINE_WIDTH = 1.2
HULL_FILL_ALPHA = 0.14
HULL_EDGE_ALPHA = 0.7
HULL_EDGE_WIDTH = 1.0
HULL_ZORDER = 1
NODE_ZORDER = 3
LEGEND_FONT_SIZE = 8
INSET_FONT_SIZE = 6
INSET_LINE_WIDTH = 1.2
LEGEND_SWATCH_INDEX = 1


def _palette(index: int) -> tuple[float, float, float]:
    hue = (index * GOLDEN_RATIO_CONJUGATE) % 1.0
    return colorsys.hsv_to_rgb(hue, SATURATIONS[index % 2], PALETTE_VALUE)


def cluster_colors(true_clusters: frozenset[Cluster]) -> dict[Cluster, tuple]:
    """Give every multi-member true cluster its own colour, biggest first."""
    multi = [m for m in sorted_clusters(true_clusters) if len(m) > 1]
    return {frozenset(m): _palette(idx) for idx, m in enumerate(multi)}


def _hull(points: np.ndarray) -> np.ndarray:
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        order = np.lexsort((points[:, 1], points[:, 0]))
        return points[[order[0], order[-1]]]


def _draw_hulls(
    canvas: Canvas, groups: Iterable[tuple[Cluster, tuple]], layout: Layout
) -> None:
    patches, colors = [], []
    for members, color in groups:
        points = np.array([layout[ref] for ref in members], dtype=float)
        patches.append(Polygon(_hull(points), closed=True))
        colors.append(color)
    if not patches:
        return
    canvas.ax.add_collection(
        PatchCollection(
            patches,
            facecolors=[(*c[:3], HULL_FILL_ALPHA) for c in colors],
            edgecolors=[(*c[:3], HULL_EDGE_ALPHA) for c in colors],
            linewidths=HULL_EDGE_WIDTH,
            zorder=HULL_ZORDER,
        )
    )


def _draw_nodes(
    canvas: Canvas, nodes: list[tuple[RefId, tuple, bool]], layout: Layout
) -> None:
    if not nodes:
        return
    points = np.array([layout[ref] for ref, _, _ in nodes], dtype=float)
    canvas.ax.scatter(
        points[:, 0],
        points[:, 1],
        s=NODE_AREA,
        c=[face for _, face, _ in nodes],
        edgecolors=[WRONG_OUTLINE if wrong else "none" for _, _, wrong in nodes],
        linewidths=[NODE_OUTLINE_WIDTH if wrong else 0.0 for _, _, wrong in nodes],
        zorder=NODE_ZORDER,
    )


def _draw_size_distribution(
    canvas: Canvas,
    true_clusters: frozenset[Cluster],
    predicted: frozenset[Cluster] | None,
) -> None:
    ax = canvas.figure.add_axes(SIDE_INSET_RECT)
    series = [(true_clusters, UNMATCHED_COLOR, "ground truth")]
    if predicted is not None:
        series.append((predicted, PREDICTION_COLOR, "prediction"))
    for clusters, color, label in series:
        sizes = sorted((len(c) for c in clusters), reverse=True)
        ax.loglog(
            range(1, len(sizes) + 1),
            sizes,
            color=color,
            linewidth=INSET_LINE_WIDTH,
            label=label,
        )
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("cluster rank", fontsize=INSET_FONT_SIZE)
    ax.set_ylabel("cluster size", fontsize=INSET_FONT_SIZE)
    ax.tick_params(labelsize=INSET_FONT_SIZE)
    ax.legend(fontsize=INSET_FONT_SIZE, frameon=False)


def _multi(clusters: Iterable[Cluster]) -> list[Cluster]:
    return [c for c in clusters if len(c) > 1]


def render_true_clusters(
    true_clusters: frozenset[Cluster],
    layout: Layout,
    text: PlotText,
    path: str | PathLike,
    dpi: int,
) -> None:
    """Draw every true cluster as a coloured blob with a hull and save a PNG."""
    colors = cluster_colors(true_clusters)
    canvas = new_canvas(layout)
    _draw_hulls(canvas, colors.items(), layout)
    nodes = [(ref, color, False) for c, color in colors.items() for ref in c]
    _draw_nodes(canvas, nodes, layout)
    _draw_size_distribution(canvas, true_clusters, None)
    omitted = len(layout) - len(nodes)
    annotate(canvas, text, f"{omitted} singleton nodes not drawn")
    save_canvas(canvas, path, dpi)


def _prediction_nodes(
    layout: Layout,
    true_of: dict[RefId, Cluster],
    predicted_of: dict[RefId, Cluster],
    mapping: dict[Cluster, Cluster | None],
    colors: dict[Cluster, tuple],
) -> list[tuple[RefId, tuple, bool]]:
    nodes = []
    for ref in layout:
        true_cluster = true_of.get(ref)
        guess = predicted_of.get(ref, frozenset({ref}))
        if true_cluster is None and len(guess) == 1:
            continue
        partner = mapping.get(guess)
        face = colors[partner] if partner is not None else UNMATCHED_COLOR
        nodes.append((ref, face, true_cluster is None or true_cluster != partner))
    return nodes


def _legend_handles() -> list[Line2D]:
    def marker(face, outline, label) -> Line2D:
        return Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=face,
            markeredgecolor=outline,
            markersize=6,
            label=label,
        )

    return [
        marker(_palette(LEGEND_SWATCH_INDEX), "none", "in matching cluster"),
        marker(_palette(LEGEND_SWATCH_INDEX), WRONG_OUTLINE, "wrong cluster"),
        marker(UNMATCHED_COLOR, WRONG_OUTLINE, "unmatched cluster"),
    ]


def render_predicted_clusters(
    true_clusters: frozenset[Cluster],
    predicted: frozenset[Cluster],
    layout: Layout,
    text: PlotText,
    path: str | PathLike,
    dpi: int,
) -> None:
    """Draw ``predicted`` on the layout of the true clusters and save a PNG.

    A predicted cluster takes the colour of the true cluster it overlaps most
    and unmatched clusters are grey. A node outside the true cluster implied
    by its colour gets a red outline. Nodes that are singletons in both
    clusterings are left out and counted in the footnote.
    """
    colors = cluster_colors(true_clusters)
    mapping = match_clusters(true_clusters, predicted)
    true_of = {ref: c for c in _multi(true_clusters) for ref in c}
    predicted_of = {ref: c for c in predicted for ref in c}
    canvas = new_canvas(layout)
    hulls = [(c, colors.get(mapping[c], UNMATCHED_COLOR)) for c in _multi(predicted)]
    _draw_hulls(canvas, hulls, layout)
    nodes = _prediction_nodes(layout, true_of, predicted_of, mapping, colors)
    _draw_nodes(canvas, nodes, layout)
    _draw_size_distribution(canvas, true_clusters, predicted)
    canvas.figure.legend(
        handles=_legend_handles(),
        loc="upper right",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    annotate(canvas, text, f"{len(layout) - len(nodes)} singleton nodes not drawn")
    save_canvas(canvas, path, dpi)
