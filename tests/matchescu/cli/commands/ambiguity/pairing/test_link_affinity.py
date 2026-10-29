import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.pairing._link_affinity import LinkAffinity

HOME_REP = RefId(0, "h")
HOME_OTHER = RefId(1, "h")
TARGET_REP = RefId(0, "t")
TARGET_MIDDLE = RefId(1, "t")
TARGET_LAST = RefId(2, "t")

SCORES = {
    (HOME_REP, TARGET_REP): 2.0,
    (HOME_REP, TARGET_MIDDLE): 0.5,
    (HOME_REP, TARGET_LAST): -1.0,
    (HOME_OTHER, TARGET_REP): -9.0,
    (HOME_OTHER, TARGET_MIDDLE): -9.0,
    (HOME_OTHER, TARGET_LAST): -9.0,
}


@pytest.fixture
def rankings():
    return {
        1: [HOME_REP, HOME_OTHER],
        2: [TARGET_REP, TARGET_MIDDLE, TARGET_LAST],
    }


@pytest.fixture
def score_calls():
    return []


@pytest.fixture
def score(score_calls):
    def _score(home, member):
        score_calls.append((home, member))
        return SCORES[(home, member)]

    return _score


@pytest.fixture
def make_affinity(rankings, score):
    def _make(target_links):
        return LinkAffinity(rankings, target_links, score)

    return _make


def test_affinity_scores_target_representative_when_one_link(make_affinity):
    assert make_affinity(1)(1, 2) == 2.0


def test_affinity_takes_minimum_over_linked_members_when_several_links(make_affinity):
    assert make_affinity(2)(1, 2) == 0.5


def test_affinity_ignores_members_beyond_target_links(make_affinity):
    assert make_affinity(2)(1, 2) != -1.0


def test_affinity_uses_available_members_when_target_is_short(make_affinity):
    assert make_affinity(5)(1, 2) == -1.0


def test_affinity_scores_from_home_representative(make_affinity):
    assert make_affinity(1)(1, 2) != SCORES[(HOME_OTHER, TARGET_REP)]


def test_affinity_scores_each_cluster_pair_once(make_affinity, score_calls):
    affinity = make_affinity(2)

    affinity(1, 2)
    affinity(1, 2)

    assert len(score_calls) == 2


@pytest.mark.parametrize(
    "target_links, expected",
    [pytest.param(3, False, id="links-fit"), pytest.param(4, True, id="links-exceed")],
)
def test_is_short_when_target_has_fewer_members_than_links(
    make_affinity, target_links, expected
):
    assert make_affinity(target_links).is_short(2) is expected
