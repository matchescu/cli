from types import SimpleNamespace

import pytest

from matchescu.cli.commands.ambiguity._config import AmbiguityDataConfig
from matchescu.cli.commands.ambiguity._generate import AmbiguityGenerator
from matchescu.cli.commands.ambiguity._settings import GenerationSettings

_HOME_CLUSTER_ID = 1
_PROX_CLUSTER_ID = 2


def _run(_sut):
    """Run the generator and return (bridge, cluster_gt, directed_gt, undirected_gt).

    With the fixtures here there is exactly one pairing, hence one bridge, and the
    default configuration has a single alpha (0.5). Bridges are returned by the
    generator, not stored in the id table.
    """
    variant = _sut()[0.5]
    return (
        variant.bridges[0],
        variant.cluster_gt,
        variant.directed_gt,
        variant.undirected_gt,
    )


@pytest.fixture
def make_generator():
    def _make(id_table, clusters, **kwargs):
        kwargs.setdefault("ambiguity_target_properties", ["name"])
        return AmbiguityGenerator(id_table, clusters, **kwargs)

    return _make


def test_bridge_copies_home_secondary_value_when_values_differ(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    bridge, *_ = _run(_sut)

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
    bridge, *_ = _run(_sut)

    assert bridge["name"] not in ("John Doe", "Jane Doe")


def test_bridge_value_respects_max_length_ratio(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters, max_length_ratio=1.0)
    bridge, *_ = _run(_sut)

    assert len(bridge["name"].split()) <= 2


def test_length_ratio_keeps_shortest_candidates_when_none_fit(monkeypatch):
    monkeypatch.setattr(
        AmbiguityGenerator,
        "_candidates",
        staticmethod(lambda *_: {"Aa Bb Cc", "Dd Ee Ff Gg"}),
    )

    scored = AmbiguityGenerator._score_pool("aa bb", "cc dd", [], max_length_ratio=1.0)

    assert [candidate for candidate, *_ in scored] == ["Aa Bb Cc"]


def test_bridge_joined_to_home_cluster(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, cluster_gt, *_ = _run(_sut)

    assert bridge_ref_id in cluster_gt[_HOME_CLUSTER_ID]


def test_bridge_refers_to_target_and_proximal_cluster(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    bridge, *_ = _run(_sut)

    assert bridge["target"].id == prox_ref_id
    assert bridge["target_cluster_id"] == _PROX_CLUSTER_ID


def test_bridge_source_is_ambiguity_generator(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    bridge, *_ = _run(_sut)

    assert bridge.id.source == "ambiguity-generator"


def test_directed_gt_marks_target_to_bridge_as_directional(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, _, directed_gt, _ = _run(_sut)

    assert directed_gt[(prox_ref_id, bridge_ref_id)] == 2


def test_undirected_gt_marks_target_to_bridge_as_match(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    _, _, _, undirected_gt = _run(_sut)
    forward = (prox_ref_id, bridge_ref_id)
    reverse = (bridge_ref_id, prox_ref_id)

    assert undirected_gt.get(forward, undirected_gt.get(reverse)) == 1


def test_secondary_features_absent_copies_empty_string(
    make_generator, id_table, clusters, bridge_ref_id
):
    _sut = make_generator(id_table, clusters)
    bridge, *_ = _run(_sut)

    assert bridge["year"] == ""


def test_secondary_feature_overlapping_target_property_raises(id_table, clusters):
    with pytest.raises(ValueError, match="overlap"):
        AmbiguityGenerator(
            id_table,
            clusters,
            ambiguity_target_properties=["name"],
            secondary_features=["name"],
        )


@pytest.mark.parametrize("reserved", ["source", "target", "target_cluster_id"])
def test_secondary_feature_with_reserved_name_raises(id_table, clusters, reserved):
    with pytest.raises(ValueError, match="reserved"):
        AmbiguityGenerator(id_table, clusters, secondary_features=[reserved])


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
        bridge, *_ = _run(_sut)

    assert "year" not in bridge.as_dict()


def test_equal_home_secondary_value_copied_verbatim(
    make_generator, make_table, clusters, bridge_ref_id
):
    _, _, id_table = make_table(
        {"name": "John Doe", "year": "1990", "city": "X"},
        {"name": "Jane Doe", "year": "1990", "city": "Y"},
    )
    _sut = make_generator(id_table, clusters, secondary_features=["year"])
    bridge, *_ = _run(_sut)

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


def _bridge_values(variant):
    return [bridge.as_dict() for bridge in variant.bridges]


def test_variant_carries_its_alpha(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters, alphas=(0.5, 0.65))

    variants = _sut()

    assert {alpha: v.alpha for alpha, v in variants.items()} == {0.5: 0.5, 0.65: 0.65}


def test_repeated_call_returns_equal_variants(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters, alphas=(0.5, 0.8))

    first, second = _sut(), _sut()

    assert first == second


def test_repeated_call_returns_equal_bridge_values(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters, alphas=(0.5, 0.8))

    first, second = _sut()[0.8], _sut()[0.8]

    assert _bridge_values(first) == _bridge_values(second)


def test_variant_bridges_do_not_depend_on_alpha_grid(
    make_generator, id_table, clusters
):
    alone = make_generator(id_table, clusters, alphas=(0.65,))()[0.65]
    in_grid = make_generator(id_table, clusters, alphas=(0.5, 0.65, 0.95))()[0.65]

    assert _bridge_values(alone) == _bridge_values(in_grid)


def test_variant_bridge_log_does_not_depend_on_alpha_grid(
    make_generator, id_table, clusters
):
    alone = make_generator(id_table, clusters, alphas=(0.65,))()[0.65]
    in_grid = make_generator(id_table, clusters, alphas=(0.5, 0.65, 0.95))()[0.65]

    assert alone.bridge_log == in_grid.bridge_log


def test_variant_bridge_log_holds_only_its_alpha(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters, alphas=(0.5, 0.8))

    variant = _sut()[0.8]

    assert {entry["alpha"] for entry in variant.bridge_log} == {0.8}


def test_bridge_log_value_matches_bridge_value(make_generator, id_table, clusters):
    _sut = make_generator(id_table, clusters)
    variant = _sut()[0.5]

    (entry,) = variant.bridge_log

    assert entry["value"] == variant.bridges[0]["name"]


def test_bridge_log_value_no_floor_equals_value_without_floor(
    make_generator, id_table, clusters
):
    _sut = make_generator(id_table, clusters, target_floor=0.0)

    (entry,) = _sut()[0.5].bridge_log

    assert entry["value_no_floor"] == entry["value"]


def test_bridge_log_value_no_floor_equals_unfloored_pick(
    make_generator, id_table, clusters
):
    unfloored = make_generator(id_table, clusters, alphas=(0.8,), target_floor=0.0)
    floored = make_generator(id_table, clusters, alphas=(0.8,), target_floor=1.0)

    (expected,) = unfloored()[0.8].bridge_log
    (actual,) = floored()[0.8].bridge_log

    assert actual["value_no_floor"] == expected["value"]


def test_variant_link_log_records_target_link(
    make_generator, id_table, clusters, bridge_ref_id, prox_ref_id
):
    _sut = make_generator(id_table, clusters, alphas=(0.5, 0.8))

    (entry,) = _sut()[0.8].link_log

    assert (entry["alpha"], entry["bridge_label"], entry["target_label"]) == (
        0.8,
        bridge_ref_id.label,
        prox_ref_id.label,
    )


def test_settings_report_effective_generation_parameters(id_table, clusters):
    _sut = AmbiguityGenerator(
        id_table,
        clusters,
        ambiguity_target_properties=["name", "city"],
        secondary_features=["year"],
        min_ambiguity=1.5,
        alphas=(0.8, 0.5),
        target_floor=0.25,
        target_links=2,
        max_length_ratio=1.2,
        dataset_name="ds",
    )

    assert _sut.settings == GenerationSettings(
        dataset_name="ds",
        ambiguity_targets=("name", "city"),
        secondary_features=("year",),
        min_ambiguity=1.5,
        alphas=(0.5, 0.8),
        target_floor=0.25,
        target_links=2,
        max_length_ratio=1.2,
    )


def test_settings_without_optional_parameters_use_defaults(id_table, clusters):
    _sut = AmbiguityGenerator(id_table, clusters)

    assert _sut.settings == GenerationSettings(
        dataset_name="",
        ambiguity_targets=(),
        secondary_features=(),
        min_ambiguity=None,
        alphas=(0.5,),
        target_floor=0.0,
        target_links=1,
    )


def test_from_config_uses_dataset_entry_settings(id_table, clusters):
    entry = AmbiguityDataConfig.model_validate(
        {
            "dataset": {
                "type": "csv",
                "directory": "./ds",
                "sources": [{"fileName": "ids.csv", "idCol": "id"}],
                "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
            },
            "ambiguityTargets": ["name"],
            "secondaryFeatures": ["year"],
            "minAmbiguity": 1.5,
            "alphas": [0.8, 0.5],
            "targetFloor": 0.25,
            "targetLinks": 2,
            "maxLengthRatio": 1.2,
        }
    )
    data = SimpleNamespace(
        id_table=id_table,
        true_matches={},
        compute_clusters=lambda: clusters,
        name="ds",
    )

    assert AmbiguityGenerator.from_config(entry, data).settings == GenerationSettings(
        dataset_name="ds",
        ambiguity_targets=("name",),
        secondary_features=("year",),
        min_ambiguity=1.5,
        alphas=(0.5, 0.8),
        target_floor=0.25,
        target_links=2,
        max_length_ratio=1.2,
    )
