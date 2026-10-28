import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileInput
from matchescu.cli.commands.ambiguity.profiling._pair_roles import PairRoles

from .conftest import ref, table_similarity


def _refs(*labels):
    return frozenset(ref(label) for label in labels)


def _pairs(*pairs):
    return frozenset((ref(left), ref(right)) for left, right in pairs)


def test_bridges_are_right_records_of_target_links(profile_input):
    assert PairRoles.from_input(profile_input).bridges == _refs("b1", "b2", "b3")


def test_target_pairs_are_target_links(profile_input):
    roles = PairRoles.from_input(profile_input)

    assert frozenset(roles.target) == _pairs(
        ("t1", "b1"), ("t2", "b1"), ("t1", "b2"), ("u1", "b2"), ("h1", "b3")
    )


def test_source_pairs_join_home_members_to_their_bridge(profile_input):
    roles = PairRoles.from_input(profile_input)

    assert frozenset(roles.source) == _pairs(
        ("a1", "b1"), ("a2", "b1"), ("e1", "b2"), ("f1", "b3"), ("f2", "b3")
    )


def test_extra_pairs_join_original_records(profile_input):
    roles = PairRoles.from_input(profile_input)

    assert frozenset(roles.extra) == _pairs(("a1", "a2"), ("t1", "t2"), ("f1", "f2"))


def test_source_pair_with_bridge_on_left_is_oriented_bridge_right(make_input):
    profile_input = make_input()
    flipped = dict(profile_input.directed_gt)
    flipped[(ref("b2"), ref("e1"))] = flipped.pop((ref("e1"), ref("b2")))
    swapped = ProfileInput(
        profile_input.records, profile_input.cluster_of, flipped, table_similarity
    )

    assert (ref("e1"), ref("b2")) in PairRoles.from_input(swapped).source


def test_pair_of_two_bridges_belongs_to_no_role(make_input):
    profile_input = make_input(
        clusters={1: ("a1", "b1", "b2"), 2: ("t1",)},
        target_links=(("t1", "b1"), ("t1", "b2")),
    )
    roles = PairRoles.from_input(profile_input)
    pair = (ref("b1"), ref("b2"))

    assert pair not in {*roles.source, *roles.extra}


def test_targets_by_bridge_group_target_records(profile_input):
    roles = PairRoles.from_input(profile_input)

    assert roles.targets_by_bridge[ref("b2")] == (ref("t1"), ref("u1"))


def test_sources_by_bridge_group_home_records(profile_input):
    roles = PairRoles.from_input(profile_input)

    assert roles.sources_by_bridge[ref("b1")] == (ref("a1"), ref("a2"))


def test_unknown_label_raises(make_input):
    profile_input = make_input()
    labels = {**profile_input.directed_gt, (ref("g1"), ref("h1")): 3}
    invalid = ProfileInput(
        profile_input.records, profile_input.cluster_of, labels, table_similarity
    )

    with pytest.raises(ValueError, match="label"):
        PairRoles.from_input(invalid)
