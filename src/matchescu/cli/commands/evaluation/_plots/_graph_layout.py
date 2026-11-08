import math
import random

import igraph
from matchescu.similarity import ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._layout import (
    NODE_SPACING,
    Layout,
    Point,
    Size,
    pack_blocks,
    strip_layout,
    strip_origin,
)

COMPONENT_SPREAD = 1.6
MIN_EXTENT = 1e-9


def _component_layout(sub_graph: igraph.Graph) -> tuple[list[Point], Size]:
    """Fruchterman-Reingold coordinates scaled to a box that grows with size."""
    side = COMPONENT_SPREAD * NODE_SPACING * math.sqrt(sub_graph.vcount())
    coords = sub_graph.layout_fruchterman_reingold().coords
    xs, ys = zip(*coords)
    min_x, min_y = min(xs), min(ys)
    extent = max(max(xs) - min_x, max(ys) - min_y, MIN_EXTENT)
    scale = side / extent
    points = [((x - min_x) * scale, (y - min_y) * scale) for x, y in coords]
    return points, (max(p[0] for p in points), max(p[1] for p in points))


def _connected_components(
    refs: list[RefId], edges: list[tuple[RefId, RefId]]
) -> list[list[RefId]]:
    index = {ref: idx for idx, ref in enumerate(refs)}
    graph = igraph.Graph(n=len(refs), edges=[(index[u], index[v]) for u, v in edges])
    components = graph.connected_components(mode="weak")
    ordered = sorted(components, key=lambda members: (-len(members), members[0]))
    return [[refs[idx] for idx in members] for members in ordered]


def _group_edges(
    components: list[list[RefId]], edges: list[tuple[RefId, RefId]]
) -> list[list[tuple[RefId, RefId]]]:
    owner = {ref: idx for idx, members in enumerate(components) for ref in members}
    grouped: list[list[tuple[RefId, RefId]]] = [[] for _ in components]
    for u, v in edges:
        grouped[owner[u]].append((u, v))
    return grouped


def _local_graph(
    members: list[RefId], edges: list[tuple[RefId, RefId]]
) -> igraph.Graph:
    index = {ref: idx for idx, ref in enumerate(members)}
    graph = igraph.Graph(n=len(members), edges=[(index[u], index[v]) for u, v in edges])
    graph.simplify()
    return graph


def _place_components(
    components: list[list[RefId]], edges: list[tuple[RefId, RefId]]
) -> tuple[Layout, float]:
    grouped = _group_edges(components, edges)
    blocks = [
        _component_layout(_local_graph(m, e)) for m, e in zip(components, grouped)
    ]
    offsets, (width, _) = pack_blocks([size for _, size in blocks])
    layout = {
        ref: (ox + x, oy + y)
        for members, (points, _), (ox, oy) in zip(components, blocks, offsets)
        for ref, (x, y) in zip(members, points)
    }
    return layout, width


def graph_layout(graph: ReferenceGraph, seed: int) -> Layout:
    """Lay out ``graph`` with every connected component packed side by side.

    Nodes without edges form a compact grid to the right of the packed area.
    The result is deterministic for a given ``seed``.
    """
    igraph.set_random_number_generator(random.Random(seed))
    edges = list(graph.matches())
    connected = list(dict.fromkeys(ref for edge in edges for ref in edge))
    connected_set = set(connected)
    isolated = [ref for ref in graph.nodes if ref not in connected_set]
    layout, width = _place_components(_connected_components(connected, edges), edges)
    return {**layout, **strip_layout(isolated, strip_origin(width))}
