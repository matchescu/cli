import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._blocking import TokenBlocking
from matchescu.cli.commands.ambiguity.pairing._scope import PairingScope

A1 = RefId(1, "s")
A2 = RefId(2, "s")
B1 = RefId(3, "s")
B2 = RefId(4, "s")
C1 = RefId(5, "s")

CLUSTERS = {A1: 1, A2: 1, B1: 2, B2: 2, C1: 3}


@pytest.fixture
def id_table():
    return (
        InMemoryIdTable()
        .put(EntityReference(A1, {"name": "alpha beta", "city": ""}))
        .put(EntityReference(A2, {"name": "gamma", "city": "zeta"}))
        .put(EntityReference(B1, {"name": "alpha beta", "city": ""}))
        .put(EntityReference(B2, {"name": "beta delta", "city": ""}))
        .put(EntityReference(C1, {"name": "alpha", "city": "zeta"}))
    )


@pytest.fixture
def make_blocking(id_table):
    def _make(neighbours=1, scope=PairingScope.ANY, max_block_size=100):
        return TokenBlocking(
            CLUSTERS,
            id_table,
            ("name",),
            neighbours,
            scope,
            max_block_size,
        )

    return _make


def test_blocking_returns_top_neighbour_edges_when_scope_any(make_blocking):
    assert make_blocking()() == [(A1, B1), (A1, B2), (A1, C1)]


def test_blocking_returns_reciprocal_edges_when_scope_mutual(make_blocking):
    assert make_blocking(scope=PairingScope.MUTUAL)() == [(A1, B1)]


def test_blocking_breaks_weight_ties_by_ref_id(make_blocking):
    edges = make_blocking()()

    assert (A1, C1) in edges
    assert (B1, C1) not in edges


def test_blocking_purges_blocks_when_larger_than_max_block_size(make_blocking):
    assert make_blocking(max_block_size=2)() == []


def test_blocking_ignores_tokens_when_field_not_targeted(make_blocking):
    assert all(A2 not in edge for edge in make_blocking(neighbours=5)())


def test_blocking_skips_pairs_when_same_cluster(make_blocking):
    edges = make_blocking(neighbours=5)()

    assert (B1, B2) not in edges
    assert (B2, B1) not in edges


def test_blocking_orients_home_to_smaller_cluster_id(make_blocking):
    edges = make_blocking(neighbours=5)()

    assert all(CLUSTERS[home] < CLUSTERS[target] for home, target in edges)


def test_blocking_returns_equal_edges_when_called_twice(make_blocking):
    sut = make_blocking()

    assert sut() == sut()
