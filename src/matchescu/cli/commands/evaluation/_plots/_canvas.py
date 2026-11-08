import math
from dataclasses import dataclass
from os import PathLike

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ._layout import Layout, layout_bounds

FIG_WIDTH_IN = 10
MIN_FIG_HEIGHT_IN = 4.0
MAX_FIG_HEIGHT_IN = 14.0
MAIN_RECT = (0.01, 0.06, 0.77, 0.87)
SIDE_INSET_RECT = (0.84, 0.10, 0.14, 0.22)
LEGEND_ANCHOR = (0.995, 0.95)
POINTS_PER_INCH = 72
PAD_FRACTION = 0.03
MIN_SPAN = 1.0
TITLE_POSITION = (0.01, 0.97)
FOOTNOTE_POSITION = (0.01, 0.015)
TITLE_FONT_SIZE = 11
FOOTNOTE_FONT_SIZE = 8


@dataclass(frozen=True)
class PlotText:
    title: str
    footnote: str


@dataclass(frozen=True)
class Canvas:
    figure: Figure
    ax: Axes
    data_per_point: float


def _padded_limits(layout: Layout) -> tuple[tuple[float, float], tuple[float, float]]:
    min_x, min_y, max_x, max_y = layout_bounds(layout)
    pad = PAD_FRACTION * max(max_x - min_x, max_y - min_y, MIN_SPAN)
    return (min_x - pad, max_x + pad), (min_y - pad, max_y + pad)


def new_canvas(layout: Layout) -> Canvas:
    """Create a canvas whose size depends only on ``layout``.

    Images drawn from the same layout therefore share size and scale.
    """
    (x0, x1), (y0, y1) = _padded_limits(layout)
    span_x, span_y = x1 - x0, y1 - y0
    axes_width = FIG_WIDTH_IN * MAIN_RECT[2]
    wanted = axes_width * span_y / span_x / MAIN_RECT[3]
    height = min(max(wanted, MIN_FIG_HEIGHT_IN), MAX_FIG_HEIGHT_IN)
    figure = Figure(figsize=(FIG_WIDTH_IN, height))
    FigureCanvasAgg(figure)
    ax = figure.add_axes(MAIN_RECT)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    points_per_unit = POINTS_PER_INCH * min(
        axes_width / span_x, height * MAIN_RECT[3] / span_y
    )
    return Canvas(figure, ax, 1.0 / points_per_unit)


def annotate(canvas: Canvas, text: PlotText, omitted: str) -> None:
    footnote = " · ".join(part for part in (text.footnote, omitted) if part)
    canvas.figure.text(
        *TITLE_POSITION,
        text.title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        va="top",
    )
    canvas.figure.text(
        *FOOTNOTE_POSITION, footnote, fontsize=FOOTNOTE_FONT_SIZE, va="bottom"
    )


def save_canvas(canvas: Canvas, path: str | PathLike, dpi: int) -> None:
    canvas.figure.savefig(path, dpi=dpi)


def node_radius_in_points(area: float) -> float:
    return math.sqrt(area) / 2.0
