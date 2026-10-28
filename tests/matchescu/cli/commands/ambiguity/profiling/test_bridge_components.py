import pytest

from matchescu.cli.commands.ambiguity.profiling import BridgeComponents


def test_components_group_clusters_linked_through_bridges(profile_input):
    components = BridgeComponents.from_input(profile_input)

    assert [c.clusters for c in components] == [
        frozenset({1, 2, 3, 4}),
        frozenset({5, 7}),
    ]


@pytest.mark.parametrize(
    "attribute, expected",
    [
        pytest.param("n_clusters", [4, 2], id="clusters"),
        pytest.param("n_records", [8, 4], id="records"),
        pytest.param("n_bridges", [2, 1], id="bridges"),
        pytest.param("n_label2", [4, 1], id="label2"),
    ],
)
def test_component_sizes(profile_input, attribute, expected):
    components = BridgeComponents.from_input(profile_input)

    assert [getattr(c, attribute) for c in components] == expected


def test_largest_component_has_most_clusters(profile_input):
    components = BridgeComponents.from_input(profile_input)

    assert components.largest.clusters == frozenset({1, 2, 3, 4})


def test_rows_number_components_from_one(profile_input):
    rows = BridgeComponents.from_input(profile_input).rows()

    assert rows[1] == {
        "bridge_component": 2,
        "n_clusters": 2,
        "n_records": 4,
        "n_bridges": 1,
        "n_label2": 1,
    }


def test_dataset_without_bridges_has_no_components(make_input):
    profile_input = make_input(clusters={1: ("a1", "a2")}, target_links=())

    assert len(BridgeComponents.from_input(profile_input)) == 0


def test_dataset_without_bridges_has_no_largest_component(make_input):
    profile_input = make_input(clusters={1: ("a1", "a2")}, target_links=())

    assert BridgeComponents.from_input(profile_input).largest is None


@pytest.mark.parametrize(
    "cluster_id, expected",
    [
        pytest.param(3, 1, id="largest"),
        pytest.param(7, 2, id="second"),
        pytest.param(6, None, id="unbridged"),
    ],
)
def test_component_of_numbers_clusters_like_rows(profile_input, cluster_id, expected):
    components = BridgeComponents.from_input(profile_input)

    assert components.component_of(cluster_id) == expected
