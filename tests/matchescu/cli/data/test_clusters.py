from unittest.mock import Mock

import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.data import load_comparison_space_clusters


def _ref(label: int) -> RefId:
    return RefId(label, "s")


@pytest.fixture
def clusters_csv(tmp_path):
    path = tmp_path / "clusters.csv"
    path.write_text("id,source,cluster_id\n1,s,1\n2,s,1\n3,s,1\n4,s,2\n5,s,2\n")
    return path


def test_loaded_clusters_exclude_references_outside_comparison_space(clusters_csv):
    refs = [_ref(i) for i in (1, 2, 4, 5)]

    result = load_comparison_space_clusters(clusters_csv, Mock(), Mock(), refs)

    assert result == {frozenset(refs[:2]), frozenset(refs[2:])}


def test_loaded_clusters_drop_clusters_with_no_reference_in_comparison_space(
    clusters_csv,
):
    refs = [_ref(i) for i in (1, 2)]

    result = load_comparison_space_clusters(clusters_csv, Mock(), Mock(), refs)

    assert result == {frozenset(refs)}


def test_loaded_clusters_add_singletons_for_unclustered_references(clusters_csv):
    refs = [_ref(i) for i in (1, 2, 9)]

    result = load_comparison_space_clusters(clusters_csv, Mock(), Mock(), refs)

    assert frozenset({_ref(9)}) in result
