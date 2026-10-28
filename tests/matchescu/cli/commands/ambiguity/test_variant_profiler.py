import warnings

import pytest

from matchescu.cli.commands.ambiguity._generate import AmbiguityGenerator
from matchescu.cli.commands.ambiguity._record_similarity import RecordSimilarity
from matchescu.cli.commands.ambiguity._variant_profiler import VariantProfiler
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions


@pytest.fixture
def variant(id_table, clusters):
    generator = AmbiguityGenerator(
        id_table, {}, clusters, ambiguity_target_properties=["name"]
    )
    return generator()[0.5]


@pytest.fixture
def sut(id_table):
    return VariantProfiler(id_table, RecordSimilarity(["name"]), ProfileOptions())


@pytest.fixture
def row(sut, variant):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sut.profile(variant)


@pytest.mark.parametrize(
    "column, expected",
    [
        pytest.param("n_records", 3, id="originals-and-bridges"),
        pytest.param("n_bridges", 1, id="bridges"),
        pytest.param("n_clusters", 2, id="clusters-from-variant"),
        pytest.param("n_label2", 1, id="directed-mapping"),
        pytest.param("n_fallback_values", 0, id="bridge-log"),
        pytest.param("link_ambiguity_stdev", 0.0, id="link-log"),
    ],
)
def test_profile_reads_variant(row, column, expected):
    assert row[column] == expected


def test_profile_scores_pairs_with_given_similarity(row, variant, id_table, home_ref):
    similarity = RecordSimilarity(["name"])
    expected = similarity(home_ref.as_dict(), variant.bridges[0].as_dict())

    assert row["sim_source_avg"] == pytest.approx(expected)


def test_components_link_home_and_target_clusters(sut, variant):
    assert sut.components(variant).rows() == [
        {
            "bridge_component": 1,
            "n_clusters": 2,
            "n_records": 3,
            "n_bridges": 1,
            "n_label2": 1,
        }
    ]
