import math

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._layout import (
    NODE_SPACING,
    Layout,
    Point,
    pack_blocks,
    sorted_clusters,
    strip_layout,
    strip_origin,
)

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
SUNFLOWER_SPACING = 0.9 * NODE_SPACING


def _blob_radius(size: int) -> float:
    return SUNFLOWER_SPACING * math.sqrt(size) + NODE_SPACING


def _blob_points(size: int) -> list[Point]:
    """Sunflower spiral centred on the origin; neighbours stay evenly apart."""
    return [
        (
            SUNFLOWER_SPACING * math.sqrt(k + 0.5) * math.cos(k * GOLDEN_ANGLE),
            SUNFLOWER_SPACING * math.sqrt(k + 0.5) * math.sin(k * GOLDEN_ANGLE),
        )
        for k in range(size)
    ]


def _blob_layout(blobs: list[list[RefId]]) -> tuple[Layout, float]:
    sizes = [(2 * _blob_radius(len(blob)),) * 2 for blob in blobs]
    offsets, (width, _) = pack_blocks(sizes)
    layout = {}
    for blob, (ox, oy) in zip(blobs, offsets):
        radius = _blob_radius(len(blob))
        for ref, (x, y) in zip(blob, _blob_points(len(blob))):
            layout[ref] = (ox + radius + x, oy + radius + y)
    return layout, width


def cluster_layout(clusters: frozenset[frozenset[RefId]]) -> Layout:
    """Give each multi-member cluster a round blob, packed biggest first.

    Singleton clusters form a compact grid to the right of the blobs.
    """
    ordered = sorted_clusters(clusters)
    blobs = [members for members in ordered if len(members) > 1]
    singletons = [members[0] for members in ordered if len(members) == 1]
    layout, width = _blob_layout(blobs)
    return {**layout, **strip_layout(singletons, strip_origin(width))}
