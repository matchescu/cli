import pytest

from matchescu.cli.commands._ambiguity_generator import AmbiguityGenerator

_HOME_CLUSTER_ID = 1
_PROX_CLUSTER_ID = 2


@pytest.fixture
def make_generator():
    def _make(id_table, clusters, **kwargs):
        kwargs.setdefault("ambiguity_target_properties", ["name"])
        return AmbiguityGenerator(id_table, {}, clusters, **kwargs)

    return _make


def test_bridge_copies_home_secondary_value_when_values_differ(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()
    bridge = id_table.get(bridge_ref_id)

    assert bridge["year"] == "1990"


def test_secondary_copy_leaves_target_ref_value_unchanged(
    make_generator, id_table, clusters, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()

    assert id_table.get(prox_ref_id)["year"] == "2000"


def test_bridge_synthesises_target_property_value(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()

    assert id_table.get(bridge_ref_id)["name"] not in ("John Doe", "Jane Doe")


def test_bridge_joined_to_home_cluster(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, cluster_gt, _, _ = _sut()

    assert bridge_ref_id in cluster_gt[_HOME_CLUSTER_ID]


def test_bridge_refers_to_target_and_proximal_cluster(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()
    bridge = id_table.get(bridge_ref_id)

    assert bridge["target"].id == prox_ref_id
    assert bridge["target_cluster_id"] == _PROX_CLUSTER_ID


def test_bridge_source_is_ambiguity_generator(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()
    bridge = id_table.get(bridge_ref_id)

    assert bridge.id.source == "ambiguity-generator"


def test_directed_gt_marks_target_to_bridge_as_directional(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, _, directed_gt, _ = _sut()

    assert directed_gt[(prox_ref_id, bridge_ref_id)] == 2


def test_undirected_gt_marks_target_to_bridge_as_match(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, _, _, undirected_gt = _sut()
    forward = (prox_ref_id, bridge_ref_id)
    reverse = (bridge_ref_id, prox_ref_id)

    assert undirected_gt.get(forward, undirected_gt.get(reverse)) == 1


def test_secondary_features_absent_copies_empty_string(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters)
    _sut()
    bridge = id_table.get(bridge_ref_id)

    assert bridge["year"] == ""


def test_secondary_feature_overlapping_target_property_raises(id_table, clusters):
    with pytest.raises(ValueError, match="overlap"):
        AmbiguityGenerator(
            id_table,
            {},
            clusters,
            ambiguity_target_properties=["name"],
            secondary_features=["name"],
        )


@pytest.mark.parametrize("reserved", ["source", "target", "target_cluster_id"])
def test_secondary_feature_with_reserved_name_raises(id_table, clusters, reserved):
    with pytest.raises(ValueError, match="reserved"):
        AmbiguityGenerator(id_table, {}, clusters, secondary_features=[reserved])


@pytest.mark.parametrize(
    "home_properties",
    [
        pytest.param({"name": "John Doe", "city": "X"}, id="no-year-key"),
        pytest.param({"name": "John Doe", "year": None, "city": "X"}, id="none-value"),
        pytest.param({"name": "John Doe", "year": "", "city": "X"}, id="empty-string"),
    ],
)
def test_missing_home_secondary_value_warns_and_skips(
    make_generator, make_table, clusters, bridge_ref_id, home_properties
):
    _, _, id_table = make_table(
        home_properties, {"name": "Jane Doe", "year": "2000", "city": "Y"}
    )
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    with pytest.warns(UserWarning, match="year"):
        _sut()
    bridge = id_table.get(bridge_ref_id)

    assert "year" not in bridge.as_dict()


def test_equal_home_secondary_value_copied_verbatim(
    make_generator, make_table, clusters, bridge_ref_id
):
    _, _, id_table = make_table(
        {"name": "John Doe", "year": "1990", "city": "X"},
        {"name": "Jane Doe", "year": "1990", "city": "Y"},
    )
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()
    bridge = id_table.get(bridge_ref_id)

    assert bridge["year"] == "1990"


def test_summary_copied_secondary_features_printed(
    make_generator, id_table, clusters, capsys
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _sut()
    captured = capsys.readouterr()

    assert "secondary feature [year]: copied=1, missing=0, equal=0" in captured.out


def test_summary_absent_when_no_secondary_features(
    make_generator, id_table, clusters, capsys
):
    _sut = make_generator(id_table, clusters)
    _sut()
    captured = capsys.readouterr()

    assert "secondary feature" not in captured.out
