from dataclasses import dataclass
from os import PathLike

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ._canvas import (
    LEGEND_ANCHOR,
    Canvas,
    PlotText,
    annotate,
    new_canvas,
    node_radius_in_points,
    save_canvas,
)
from ._edges import DrawnEdge, EdgeStatus
from ._layout import Layout

NODE_COLOR = "#263238"
NODE_AREA = 14.0
NODE_PAD_FACTOR = 1.3
ONE_WAY_WIDTH = 0.7
BOTH_WAYS_WIDTH = 1.8
ARROW_LENGTH_POINTS = 6.0
ARROW_SHAFT_INCHES = 0.012
ARROW_HEAD_WIDTH = 6
ARROW_HEAD_LENGTH = 5
ARROW_HEAD_AXIS_LENGTH = 4.5
LEGEND_FONT_SIZE = 8
EDGE_ZORDER = 1
NODE_ZORDER = 3
MIN_NORM = 1e-9
NEUTRAL = "#757575"


@dataclass(frozen=True)
class EdgeStyle:
    color: str
    linestyle: str
    label: str


STYLES = {
    EdgeStatus.TRUTH: EdgeStyle("#37474f", "-", "ground truth"),
    EdgeStatus.TP: EdgeStyle("#2e7d32", "-", "true positive"),
    EdgeStatus.FP: EdgeStyle("#c62828", "-", "false positive"),
    EdgeStatus.FN: EdgeStyle("#9e9e9e", "--", "false negative"),
}
DRAW_ORDER = (EdgeStatus.FN, EdgeStatus.TRUTH, EdgeStatus.TP, EdgeStatus.FP)


def _coordinates(
    edges: list[DrawnEdge], layout: Layout
) -> tuple[np.ndarray, np.ndarray]:
    source = np.array([layout[edge.source] for edge in edges], dtype=float)
    target = np.array([layout[edge.target] for edge in edges], dtype=float)
    return source, target


def _unit_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - source
    norm = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), MIN_NORM)
    return delta / norm[:, None]


def _draw_shafts(canvas, edges, source, tip, style: EdgeStyle) -> None:
    widths = [BOTH_WAYS_WIDTH if e.bidirectional else ONE_WAY_WIDTH for e in edges]
    segments = np.stack([source, tip], axis=1)
    canvas.ax.add_collection(
        LineCollection(
            segments,
            colors=style.color,
            linewidths=widths,
            linestyles=style.linestyle,
            zorder=EDGE_ZORDER,
        )
    )


def _draw_heads(canvas: Canvas, tip, direction, style: EdgeStyle) -> None:
    vector = direction * ARROW_LENGTH_POINTS * canvas.data_per_point
    canvas.ax.quiver(
        tip[:, 0],
        tip[:, 1],
        vector[:, 0],
        vector[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        units="inches",
        width=ARROW_SHAFT_INCHES,
        headwidth=ARROW_HEAD_WIDTH,
        headlength=ARROW_HEAD_LENGTH,
        headaxislength=ARROW_HEAD_AXIS_LENGTH,
        pivot="tip",
        color=style.color,
        zorder=EDGE_ZORDER,
    )


def _draw_status(canvas: Canvas, edges: list[DrawnEdge], layout: Layout, status):
    if not edges:
        return
    style = STYLES[status]
    source, target = _coordinates(edges, layout)
    direction = _unit_vectors(source, target)
    pad = node_radius_in_points(NODE_AREA) * NODE_PAD_FACTOR * canvas.data_per_point
    one_way = np.array([not e.bidirectional for e in edges])
    tip = np.where(one_way[:, None], target - direction * pad, target)
    _draw_shafts(canvas, edges, source, tip, style)
    if one_way.any():
        _draw_heads(canvas, tip[one_way], direction[one_way], style)


def _drawn_nodes(edges: list[DrawnEdge]) -> list:
    return list(dict.fromkeys(ref for e in edges for ref in (e.source, e.target)))


def _draw_nodes(canvas: Canvas, nodes: list, layout: Layout) -> None:
    points = np.array([layout[ref] for ref in nodes], dtype=float)
    canvas.ax.scatter(
        points[:, 0],
        points[:, 1],
        s=NODE_AREA,
        c=NODE_COLOR,
        linewidths=0,
        zorder=NODE_ZORDER,
    )


def _legend_handles(edges: list[DrawnEdge]) -> list[Line2D]:
    present = {edge.status for edge in edges}
    handles = [
        Line2D([0], [0], color=s.color, linestyle=s.linestyle, label=s.label)
        for status, s in STYLES.items()
        if status in present
    ]
    handles.append(
        Line2D([0], [0], color=NEUTRAL, lw=BOTH_WAYS_WIDTH, label="both directions")
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=NEUTRAL,
            lw=ONE_WAY_WIDTH,
            marker=">",
            markersize=4,
            markevery=[1],
            label="one direction",
        )
    )
    return handles


def render_graph(
    edges: list[DrawnEdge],
    layout: Layout,
    text: PlotText,
    path: str | PathLike,
    dpi: int,
) -> None:
    """Draw ``edges`` at the positions in ``layout`` and save a PNG.

    A bidirectional edge is one thick line without arrowheads. Nodes without
    drawn edges are left out and counted in the footnote.
    """
    canvas = new_canvas(layout)
    for status in DRAW_ORDER:
        _draw_status(canvas, [e for e in edges if e.status == status], layout, status)
    nodes = _drawn_nodes(edges)
    _draw_nodes(canvas, nodes, layout)
    canvas.figure.legend(
        handles=_legend_handles(edges),
        loc="upper right",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    annotate(canvas, text, f"{len(layout) - len(nodes)} isolated nodes not drawn")
    save_canvas(canvas, path, dpi)
