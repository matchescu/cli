import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._input import PairingInput
from matchescu.cli.commands.ambiguity.pairing._simple_pairs import SimplePairs
from matchescu.cli.commands.ambiguity.pairing._summary import PairingSummary

A = RefId(1, "s")
B = RefId(2, "s")
C = RefId(3, "s")

SCORES = {(A, B): 1.0, (A, C): 3.0}


@pytest.fixture
def id_table():
    return (
        InMemoryIdTable()
        .put(EntityReference(A, {"name": "alpha"}))
        .put(EntityReference(B, {"name": "beta"}))
        .put(EntityReference(C, {"name": "gamma"}))
    )


@pytest.fixture
def score_calls():
    return []


@pytest.fixture
def score(score_calls):
    def _score(left, right):
        score_calls.append((left, right))
        return SCORES[(left, right)]

    return _score


@pytest.fixture
def edges():
    return [(A, B), (A, C)]


@pytest.fixture
def sut(id_table, score, edges):
    return SimplePairs(
        PairingInput(
            clusters={1: {A}, 2: {B}, 3: {C}},
            id_table=id_table,
            score=score,
            score_refs=lambda *_: 0.0,
            target_fields=("name",),
            target_links=1,
            on_progress=lambda _: None,
        ),
        blocking=lambda: edges,
    )


def test_seed_links_only_paired_target(sut):
    first, _ = sut.seeds(None)

    assert first.linked_refs == (B,)


def test_seed_home_is_first_edge_member(sut):
    first, _ = sut.seeds(None)

    assert first.home_ref == A


def test_seed_keeps_home_and_target_cluster_ids(sut):
    first, _ = sut.seeds(None)

    assert (first.home_cluster_id, first.target_cluster_id) == (1, 2)


def test_seeds_drop_pairs_when_score_below_min_score(sut):
    assert [seed.target_ref for seed in sut.seeds(2.0)] == [C]


def test_pairs_scored_once_when_seeds_called_twice(sut, score_calls):
    sut.seeds(None)
    sut.seeds(2.0)

    assert len(score_calls) == 2


def test_label_keys_differ_when_targets_differ(sut):
    first, second = sut.seeds(None)

    assert first.label_key != second.label_key


def test_summaries_count_bridges_per_threshold(sut):
    assert [s.bridge_count for s in sut.summaries([None, 2.0, 4.0])] == [2, 1, 0]


def test_summaries_count_one_link_per_bridge(sut):
    assert sut.summaries([None]) == [PairingSummary(2, 2, 0)]
