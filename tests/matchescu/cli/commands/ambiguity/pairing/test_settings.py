import pytest
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._greedy_cluster_reps import (
    GreedyClusterReps,
)
from matchescu.cli.commands.ambiguity.pairing._input import PairingInput
from matchescu.cli.commands.ambiguity.pairing._settings import (
    PairingSettings,
    PairingStrategyName,
)
from matchescu.cli.commands.ambiguity.pairing._simple_pairs import SimplePairs

REF = RefId(1, "s")


@pytest.fixture
def pairing_input():
    id_table = InMemoryIdTable().put(EntityReference(REF, {"name": "alpha"}))
    return PairingInput(
        clusters={1: {REF}},
        id_table=id_table,
        score=lambda *_: 0.0,
        score_refs=lambda *_: 0.0,
        target_fields=("name",),
        target_links=1,
        on_progress=lambda _: None,
    )


def test_build_returns_greedy_cluster_reps_by_default(pairing_input):
    assert isinstance(PairingSettings().build(pairing_input), GreedyClusterReps)


def test_build_returns_simple_pairs_when_selected(pairing_input):
    settings = PairingSettings(name=PairingStrategyName.SIMPLE_PAIRS)

    assert isinstance(settings.build(pairing_input), SimplePairs)


def test_target_links_for_keeps_value_when_greedy():
    assert PairingSettings().target_links_for(3) == 3


def test_target_links_for_returns_one_when_simple_pairs():
    settings = PairingSettings(name=PairingStrategyName.SIMPLE_PAIRS)

    assert settings.target_links_for(1) == 1


def test_target_links_for_warns_when_simple_pairs_and_several_links():
    settings = PairingSettings(name=PairingStrategyName.SIMPLE_PAIRS)

    with pytest.warns(UserWarning, match="targetLinks"):
        settings.target_links_for(2)
