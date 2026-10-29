import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._greedy_cluster_reps import (
    GreedyClusterReps,
)
from matchescu.cli.commands.ambiguity.pairing._input import PairingInput

HOME = RefId(1, "ds")
MIDDLE = RefId(2, "ds")
LAST = RefId(3, "ds")

SCORES = {
    (HOME, MIDDLE): 2.0,
    (HOME, LAST): 1.0,
    (MIDDLE, LAST): 0.5,
    (LAST, MIDDLE): 0.5,
    (MIDDLE, HOME): 2.0,
    (LAST, HOME): 1.0,
}


@pytest.fixture
def id_table():
    return (
        InMemoryIdTable()
        .put(EntityReference(HOME, {"name": "abc"}))
        .put(EntityReference(MIDDLE, {"name": "abc"}))
        .put(EntityReference(LAST, {"name": "xyz"}))
    )


@pytest.fixture
def clusters():
    return {1: {HOME}, 2: {MIDDLE, LAST}}


def _score(ref_id, other):
    return SCORES[(getattr(ref_id, "id", ref_id), getattr(other, "id", other))]


@pytest.fixture
def make_strategy(id_table, clusters):
    def _make(**kwargs):
        return GreedyClusterReps(
            PairingInput(
                clusters=clusters,
                id_table=id_table,
                score=_score,
                score_refs=_score,
                target_fields=("name",),
                target_links=kwargs.get("target_links", 1),
                on_progress=lambda _: None,
            )
        )

    return _make


def test_seed_home_ref_is_home_representative_when_paired(make_strategy):
    (seed,) = make_strategy().seeds(None)

    assert seed.home_ref == HOME


def test_seed_target_ref_is_target_representative_when_paired(make_strategy):
    (seed,) = make_strategy().seeds(None)

    assert seed.target_ref == MIDDLE


def test_seed_linked_refs_follow_target_ranking_when_several_links(make_strategy):
    (seed,) = make_strategy(target_links=2).seeds(None)

    assert seed.linked_refs == (MIDDLE, LAST)


def test_seed_label_key_is_home_cluster_id(make_strategy):
    (seed,) = make_strategy().seeds(None)

    assert seed.label_key == 1


def test_seed_keeps_pairing_score(make_strategy):
    (seed,) = make_strategy().seeds(None)

    assert seed.score == 2.0


def test_seeds_empty_when_score_below_min_score(make_strategy):
    assert make_strategy().seeds(2.5) == []


def test_scores_hold_every_pairing_when_no_threshold(make_strategy):
    assert make_strategy().scores() == [2.0]
