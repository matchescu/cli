import csv

import pytest

from matchescu.cli.commands.ambiguity._dataset_generation import DatasetGeneration
from matchescu.cli.commands.ambiguity._generate import AmbiguityGenerator
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions

pytestmark = pytest.mark.filterwarnings("ignore:margin mixture skipped")


@pytest.fixture
def make_generator(id_table, clusters):
    def _make(alphas=(0.5, 0.8), targets=("name",)):
        return AmbiguityGenerator(
            id_table,
            clusters,
            ambiguity_target_properties=list(targets),
            alphas=alphas,
            dataset_name="ds",
        )

    return _make


@pytest.fixture
def sut(tmp_path):
    return DatasetGeneration(tmp_path, ProfileOptions())


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_writes_variant_directories(sut, make_generator, id_table, tmp_path):
    sut.run(make_generator(), id_table)

    assert (tmp_path / "amb_ds_a080_dir" / "amb_ds_a080_dir_mapping.csv").is_file()


def test_run_writes_one_metadata_row_per_alpha(sut, make_generator, id_table, tmp_path):
    sut.run(make_generator(), id_table)

    rows = _read(tmp_path / "metadata.csv")

    assert [r["dataset"] for r in rows] == ["amb_ds_a050", "amb_ds_a080"]


def test_metadata_row_joins_identity_and_profile(
    sut, make_generator, id_table, tmp_path
):
    sut.run(make_generator(), id_table)

    row = _read(tmp_path / "metadata.csv")[0]

    assert (row["directed_path"], row["n_bridges"]) == ("amb_ds_a050_dir", "1")


def test_rerun_upserts_metadata_rows(sut, make_generator, id_table, tmp_path):
    sut.run(make_generator(alphas=(0.5, 0.8)), id_table)

    sut.run(make_generator(alphas=(0.65, 0.8)), id_table)

    datasets = [r["dataset"] for r in _read(tmp_path / "metadata.csv")]
    assert datasets == ["amb_ds_a050", "amb_ds_a080", "amb_ds_a065"]


def test_run_writes_bridge_components(sut, make_generator, id_table, tmp_path):
    sut.run(make_generator(), id_table)

    rows = _read(tmp_path / "amb_ds_bridge_components.csv")

    assert rows == [
        {
            "bridge_component": "1",
            "n_clusters": "2",
            "n_records": "3",
            "n_bridges": "1",
            "n_label2": "1",
        }
    ]


def test_run_without_ambiguity_targets_raises_before_writing(
    sut, make_generator, id_table, tmp_path
):
    with pytest.raises(ValueError, match="field"):
        sut.run(make_generator(targets=()), id_table)

    assert list(tmp_path.iterdir()) == []
