import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._seed import BridgeSeed
from matchescu.cli.commands.ambiguity.pairing._summary import PairingSummary

HOME = RefId(0, "h")
TARGET = RefId(0, "t")
MEMBERS = [RefId(1, "t"), RefId(2, "t"), RefId(3, "t")]


def _seed(linked_refs, is_short):
    return BridgeSeed(
        home_cluster_id=1,
        home_ref=HOME,
        target_cluster_id=2,
        target_ref=TARGET,
        linked_refs=linked_refs,
        is_short=is_short,
        score=1.0,
        label_key=1,
    )


@pytest.fixture
def summary():
    return PairingSummary.of(
        [
            _seed(tuple(MEMBERS), False),
            _seed(tuple(MEMBERS[:2]), True),
        ]
    )


@pytest.mark.parametrize(
    "field, expected",
    [
        pytest.param("bridge_count", 2, id="bridges"),
        pytest.param("link_count", 5, id="links"),
        pytest.param("short_bridge_count", 1, id="short-bridges"),
    ],
)
def test_pairing_summary_value(summary, field, expected):
    assert getattr(summary, field) == expected
