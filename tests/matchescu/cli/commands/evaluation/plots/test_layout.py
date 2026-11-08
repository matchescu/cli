import pytest
from matchescu.similarity import MatchResult

from matchescu.cli.commands.evaluation._plots._cluster_layout import cluster_layout
from matchescu.cli.commands.evaluation._plots._graph_layout import graph_layout
from matchescu.cli.commands.evaluation._plots._layout import (
    layout_bounds,
    load_layout,
    pack_blocks,
    save_layout,
)

SEED = 7


def _overlap(a, b, size_a, size_b) -> bool:
    return (
        a[0] < b[0] + size_b[0]
        and b[0] < a[0] + size_a[0]
        and a[1] < b[1] + size_b[1]
        and b[1] < a[1] + size_a[1]
    )


def test_pack_blocks_places_blocks_without_overlap():
    sizes = [(4.0, 3.0), (2.0, 2.0), (5.0, 1.0), (3.0, 3.0), (1.0, 1.0)]

    offsets, _ = pack_blocks(sizes)

    pairs = [(i, j) for i in range(len(sizes)) for j in range(i)]
    assert not any(
        _overlap(offsets[i], offsets[j], sizes[i], sizes[j]) for i, j in pairs
    )


def test_pack_blocks_reports_extent_covering_all_blocks():
    sizes = [(4.0, 3.0), (2.0, 2.0), (5.0, 1.0)]

    offsets, (width, height) = pack_blocks(sizes)

    assert all(
        x + w <= width + 1e-9 and y + h <= height + 1e-9
        for (x, y), (w, h) in zip(offsets, sizes)
    )


def test_layout_survives_json_round_trip(refs, tmp_path):
    layout = {refs[0]: (0.5, 1.5), refs[1]: (-2.0, 3.0)}
    path = tmp_path / "layout.json"

    save_layout(layout, path)

    assert load_layout(path) == layout


def test_layout_bounds_cover_every_point(refs):
    layout = {refs[0]: (0.0, 5.0), refs[1]: (-2.0, 3.0)}

    assert layout_bounds(layout) == (-2.0, 3.0, 0.0, 5.0)


@pytest.fixture
def two_components(refs, make_graph):
    graph = make_graph([(refs[0], refs[1]), (refs[1], refs[2]), (refs[3], refs[4])])
    graph.add(MatchResult(refs[5], refs[4], 0, [0.9, 0.1]))
    return graph


def test_graph_layout_positions_every_node(refs, two_components):
    assert set(graph_layout(two_components, SEED)) == set(refs)


def test_graph_layout_is_deterministic_for_a_seed(two_components):
    assert graph_layout(two_components, SEED) == graph_layout(two_components, SEED)


def test_graph_layout_places_isolates_outside_connected_area(refs, two_components):
    layout = graph_layout(two_components, SEED)

    connected = {r: layout[r] for r in refs[:5]}
    _, _, max_x, _ = layout_bounds(connected)
    assert layout[refs[5]][0] > max_x


def test_graph_layout_keeps_components_apart(refs, two_components):
    layout = graph_layout(two_components, SEED)

    first = layout_bounds({r: layout[r] for r in refs[:3]})
    second = layout_bounds({r: layout[r] for r in refs[3:5]})
    assert not _overlap(
        first[:2],
        second[:2],
        (first[2] - first[0], first[3] - first[1]),
        (second[2] - second[0], second[3] - second[1]),
    )


def test_cluster_layout_positions_every_node(refs):
    clusters = frozenset(
        {frozenset(refs[:3]), frozenset(refs[3:5]), frozenset(refs[5:])}
    )

    assert set(cluster_layout(clusters)) == set(refs)


def test_cluster_layout_is_deterministic(refs):
    clusters = frozenset({frozenset(refs[:3]), frozenset(refs[3:])})

    assert cluster_layout(clusters) == cluster_layout(clusters)


def test_cluster_layout_places_singletons_outside_cluster_area(refs):
    clusters = frozenset(
        {frozenset(refs[:3]), frozenset(refs[3:5]), frozenset(refs[5:])}
    )

    layout = cluster_layout(clusters)

    _, _, max_x, _ = layout_bounds({r: layout[r] for r in refs[:5]})
    assert layout[refs[5]][0] > max_x
