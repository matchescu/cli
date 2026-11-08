import json
import math
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path

from matchescu.typing import EntityReferenceIdentifier as RefId

Point = tuple[float, float]
Size = tuple[float, float]
Layout = dict[RefId, Point]

NODE_SPACING = 1.0
BLOCK_GAP = 2.0 * NODE_SPACING
STRIP_SPACING = 0.3 * NODE_SPACING
STRIP_MARGIN = 3.0 * NODE_SPACING
SHELF_WIDTH_FACTOR = 1.4


def ref_sort_key(ref: RefId) -> tuple[str, str]:
    return str(ref.source), str(ref.label)


def sorted_clusters(clusters: Iterable[frozenset[RefId]]) -> list[list[RefId]]:
    """Order clusters by decreasing size; ties and members sort by reference."""
    members = [sorted(cluster, key=ref_sort_key) for cluster in clusters]
    return sorted(members, key=lambda m: (-len(m), ref_sort_key(m[0])))


def _row_limit(sizes: list[Size], gap: float) -> float:
    area = sum((w + gap) * (h + gap) for w, h in sizes)
    widest = max((w for w, _ in sizes), default=0.0)
    return max(widest, math.sqrt(area) * SHELF_WIDTH_FACTOR)


def pack_blocks(sizes: list[Size], gap: float = BLOCK_GAP) -> tuple[list[Point], Size]:
    """Pack rectangles in rows, in the given order.

    :return: the lower-left corner of every block and the packed extent.
    """
    limit = _row_limit(sizes, gap)
    offsets: list[Point] = []
    x = y = row_height = width = 0.0
    for w, h in sizes:
        if x > 0 and x + w > limit:
            x, y, row_height = 0.0, y + row_height + gap, 0.0
        offsets.append((x, y))
        x, row_height = x + w + gap, max(row_height, h)
        width = max(width, x - gap)
    return offsets, (width, y + row_height)


def strip_layout(refs: Iterable[RefId], origin: Point) -> Layout:
    """Arrange ``refs`` in a compact grid whose lower-left corner is ``origin``."""
    refs = list(refs)
    columns = max(1, math.ceil(math.sqrt(len(refs))))
    return {
        ref: (
            origin[0] + (idx % columns) * STRIP_SPACING,
            origin[1] + (idx // columns) * STRIP_SPACING,
        )
        for idx, ref in enumerate(refs)
    }


def strip_origin(main_width: float) -> Point:
    return main_width + STRIP_MARGIN, 0.0


def layout_bounds(layout: Mapping[RefId, Point]) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` of all positions."""
    xs, ys = zip(*layout.values())
    return min(xs), min(ys), max(xs), max(ys)


def save_layout(layout: Mapping[RefId, Point], path: str | PathLike) -> None:
    records = [
        {"label": ref.label, "source": ref.source, "x": x, "y": y}
        for ref, (x, y) in layout.items()
    ]
    Path(path).write_text(json.dumps(records))


def load_layout(path: str | PathLike) -> Layout:
    records = json.loads(Path(path).read_text())
    return {RefId(r["label"], r["source"]): (r["x"], r["y"]) for r in records}
