import csv

import pytest

from matchescu.cli.commands.ambiguity._metadata import GenerationIdentity, MetadataStore
from matchescu.cli.commands.ambiguity._settings import GenerationSettings
from matchescu.cli.commands.ambiguity._writer import VariantNaming, VariantPaths


@pytest.fixture
def settings():
    return GenerationSettings(
        dataset_name="ds",
        ambiguity_targets=("name", "city"),
        secondary_features=(),
        min_ambiguity=None,
        alphas=(0.65,),
        target_floor=0.0,
        target_links=2,
    )


@pytest.fixture
def identity(settings, tmp_path):
    naming = VariantNaming("ds", 2)
    paths = VariantPaths(tmp_path / "amb_ds_a065_k2", tmp_path / "amb_ds_a065_k2_dir")
    return GenerationIdentity(settings, 0.65, naming, paths)


@pytest.fixture
def store(tmp_path):
    return MetadataStore(tmp_path / "metadata.csv")


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_identity_row_keys_dataset_by_undirected_variant_name(identity, tmp_path):
    assert identity.as_row(tmp_path)["dataset"] == "amb_ds_a065_k2"


def test_identity_row_holds_generation_settings(identity, tmp_path):
    row = identity.as_row(tmp_path)

    assert {k: row[k] for k in list(row)[1:9]} == {
        "source_dataset": "ds",
        "alpha": 0.65,
        "tag": "a065",
        "target_floor": 0.0,
        "target_links": 2,
        "min_ambiguity": None,
        "ambiguity_targets": "name,city",
        "secondary_features": "",
    }


def test_identity_row_paths_are_relative_to_root(identity, tmp_path):
    row = identity.as_row(tmp_path)

    assert (row["undirected_path"], row["directed_path"]) == (
        "amb_ds_a065_k2",
        "amb_ds_a065_k2_dir",
    )


def test_upsert_creates_missing_file(store, tmp_path):
    store.upsert([{"dataset": "a", "n": 1}])

    assert _read(tmp_path / "metadata.csv") == [{"dataset": "a", "n": "1"}]


def test_upsert_replaces_row_with_same_key(store, tmp_path):
    store.upsert([{"dataset": "a", "n": 1}, {"dataset": "b", "n": 2}])

    store.upsert([{"dataset": "a", "n": 3}])

    assert _read(tmp_path / "metadata.csv") == [
        {"dataset": "a", "n": "3"},
        {"dataset": "b", "n": "2"},
    ]


def test_upsert_appends_rows_with_new_keys(store, tmp_path):
    store.upsert([{"dataset": "a", "n": 1}])

    store.upsert([{"dataset": "c", "n": 4}])

    assert [r["dataset"] for r in _read(tmp_path / "metadata.csv")] == ["a", "c"]


def test_upsert_appends_new_columns_after_existing_ones(store, tmp_path):
    store.upsert([{"dataset": "a", "n": 1}])

    store.upsert([{"dataset": "b", "extra": "x", "n": 2}])

    assert list(_read(tmp_path / "metadata.csv")[0]) == ["dataset", "n", "extra"]


def test_upsert_leaves_missing_cells_empty(store, tmp_path):
    store.upsert([{"dataset": "a", "n": 1}])

    store.upsert([{"dataset": "b", "extra": "x"}])

    assert _read(tmp_path / "metadata.csv")[0]["extra"] == ""


def test_upsert_writes_none_as_empty_cell(store, tmp_path):
    store.upsert([{"dataset": "a", "n": None}])

    assert _read(tmp_path / "metadata.csv")[0]["n"] == ""


def test_upsert_rejects_row_without_key(store):
    with pytest.raises(ValueError, match="dataset"):
        store.upsert([{"n": 1}])
