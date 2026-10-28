import polars as pl
import pytest
from matchescu.typing import EntityReference

from matchescu.cli.commands.ambiguity._variant import AmbiguousVariant
from matchescu.cli.commands.ambiguity._writer import (
    VariantNaming,
    VariantPaths,
    VariantWriter,
)

_HOME_CLUSTER_ID = 1
_PROX_CLUSTER_ID = 2


@pytest.fixture
def make_variant(home_ref, prox_ref, bridge_ref_id):
    def _make(alpha, bridge_log=None, link_log=None):
        bridge = EntityReference(
            bridge_ref_id,
            {
                "source": home_ref,
                "target": prox_ref,
                "target_cluster_id": _PROX_CLUSTER_ID,
                "name": "J Doe",
                "year": "1990",
                "city": "",
            },
        )
        return AmbiguousVariant(
            alpha=alpha,
            bridges=(bridge,),
            cluster_gt={
                _HOME_CLUSTER_ID: {home_ref.id, bridge_ref_id},
                _PROX_CLUSTER_ID: {prox_ref.id},
            },
            directed_gt={
                (home_ref.id, bridge_ref_id): 1,
                (prox_ref.id, bridge_ref_id): 2,
            },
            undirected_gt={
                (home_ref.id, bridge_ref_id): 1,
                (prox_ref.id, bridge_ref_id): 1,
            },
            bridge_log=(
                ({"alpha": alpha, "field": "name", "value": "J Doe"},)
                if bridge_log is None
                else bridge_log
            ),
            link_log=(
                ({"alpha": alpha, "target_label": prox_ref.id.label},)
                if link_log is None
                else link_log
            ),
        )

    return _make


@pytest.fixture
def make_writer(id_table, tmp_path):
    def _make(target_links=1):
        return VariantWriter(id_table, tmp_path, VariantNaming("ds", target_links))

    return _make


@pytest.mark.parametrize(
    "alpha, target_links, expected",
    [
        pytest.param(0.65, 1, "amb_ds_a065", id="single-link"),
        pytest.param(0.5, 2, "amb_ds_a050_k2", id="several-links"),
        pytest.param(0.957, 1, "amb_ds_a096", id="rounded-tag"),
    ],
)
def test_naming_undirected_names_alpha_and_links(alpha, target_links, expected):
    assert VariantNaming("ds", target_links).undirected(alpha) == expected


def test_naming_directed_appends_dir_suffix():
    assert VariantNaming("ds", 2).directed(0.5) == "amb_ds_a050_k2_dir"


def test_naming_tag_encodes_alpha_in_hundredths():
    assert VariantNaming("ds", 1).tag(0.8) == "a080"


def test_writer_returns_variant_directories(make_writer, make_variant, tmp_path):
    paths = make_writer().write({0.5: make_variant(0.5)})

    assert paths == {
        0.5: VariantPaths(tmp_path / "amb_ds_a050", tmp_path / "amb_ds_a050_dir")
    }


def test_writer_ids_csv_drops_reserved_bridge_properties(
    make_writer, make_variant, tmp_path
):
    make_writer().write({0.5: make_variant(0.5)})

    ids = pl.read_csv(tmp_path / "amb_ds_a050" / "amb_ds_a050_ids.csv")

    assert ids.columns == ["id", "source", "name", "year", "city"]


def test_writer_ids_csv_holds_original_records_and_bridges(
    make_writer, make_variant, tmp_path
):
    make_writer().write({0.5: make_variant(0.5)})

    ids = pl.read_csv(tmp_path / "amb_ds_a050_dir" / "amb_ds_a050_dir_ids.csv")

    assert ids["source"].to_list() == ["home", "prox", "ambiguity-generator"]


@pytest.mark.parametrize(
    "variant_dir, expected_labels",
    [
        pytest.param("amb_ds_a050", [1, 1], id="undirected"),
        pytest.param("amb_ds_a050_dir", [1, 2], id="directed"),
    ],
)
def test_writer_mapping_csv_holds_variant_labels(
    make_writer, make_variant, tmp_path, variant_dir, expected_labels
):
    make_writer().write({0.5: make_variant(0.5)})

    mapping = pl.read_csv(tmp_path / variant_dir / f"{variant_dir}_mapping.csv")

    assert mapping["label"].to_list() == expected_labels


def test_writer_cluster_mapping_puts_bridge_in_home_cluster(
    make_writer, make_variant, tmp_path, bridge_ref_id
):
    make_writer().write({0.5: make_variant(0.5)})

    clusters = pl.read_csv(tmp_path / "amb_ds_a050" / "amb_ds_a050_cluster_mapping.csv")
    bridge_row = clusters.filter(pl.col("source") == bridge_ref_id.source)

    assert bridge_row["cluster_id"].to_list() == [_HOME_CLUSTER_ID]


def test_writer_bridge_log_concatenates_variants(make_writer, make_variant, tmp_path):
    make_writer().write({0.5: make_variant(0.5), 0.8: make_variant(0.8)})

    log = pl.read_csv(tmp_path / "amb_ds_bridge_log.csv")

    assert log["alpha"].to_list() == [0.5, 0.8]


def test_writer_link_log_name_carries_links_suffix(make_writer, make_variant, tmp_path):
    make_writer(target_links=2).write({0.5: make_variant(0.5)})

    assert (tmp_path / "amb_ds_k2_link_log.csv").is_file()


@pytest.mark.parametrize(
    "log_file, empty_logs",
    [
        pytest.param("amb_ds_bridge_log.csv", {"bridge_log": ()}, id="bridge-log"),
        pytest.param("amb_ds_link_log.csv", {"link_log": ()}, id="link-log"),
    ],
)
def test_writer_skips_empty_log(
    make_writer, make_variant, tmp_path, log_file, empty_logs
):
    make_writer().write({0.5: make_variant(0.5, **empty_logs)})

    assert not (tmp_path / log_file).exists()
