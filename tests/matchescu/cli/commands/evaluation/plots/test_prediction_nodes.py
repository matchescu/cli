from matchescu.cli.commands.evaluation._plots._cluster_match import match_clusters
from matchescu.cli.commands.evaluation._plots._cluster_render import (
    UNMATCHED_COLOR,
    _prediction_nodes,
    cluster_colors,
)


def _nodes(refs, truth, predicted):
    layout = {ref: (0.0, 0.0) for ref in refs}
    true_of = {r: c for c in truth if len(c) > 1 for r in c}
    predicted_of = {r: c for c in predicted for r in c}
    mapping = match_clusters(truth, predicted)
    nodes = _prediction_nodes(
        layout, true_of, predicted_of, mapping, cluster_colors(truth)
    )
    return {ref: (face, wrong) for ref, face, wrong in nodes}


def test_true_singleton_merged_into_cluster_is_flagged_wrong(refs):
    truth = frozenset({frozenset(refs[:2]), frozenset(refs[2:3])})
    predicted = frozenset({frozenset(refs[:3])})

    assert _nodes(refs[:3], truth, predicted)[refs[2]][1] is True


def test_node_in_matching_cluster_is_not_flagged(refs):
    truth = frozenset({frozenset(refs[:2])})

    assert _nodes(refs[:2], truth, truth)[refs[0]][1] is False


def test_clustered_node_predicted_as_singleton_is_grey_and_wrong(refs):
    truth = frozenset({frozenset(refs[:2])})
    predicted = frozenset({frozenset(refs[:1]), frozenset(refs[1:2])})

    assert _nodes(refs[:2], truth, predicted)[refs[0]] == (UNMATCHED_COLOR, True)


def test_node_singleton_in_both_clusterings_is_not_drawn(refs):
    truth = frozenset({frozenset(refs[:2]), frozenset(refs[2:3])})
    predicted = frozenset({frozenset(refs[:2]), frozenset(refs[2:3])})

    assert refs[2] not in _nodes(refs[:3], truth, predicted)


def test_true_singletons_merged_into_unmatched_cluster_are_flagged_wrong(refs):
    truth = frozenset({frozenset(refs[:2]), frozenset(refs[2:3]), frozenset(refs[3:4])})
    predicted = frozenset({frozenset(refs[:2]), frozenset(refs[2:4])})

    assert _nodes(refs[:4], truth, predicted)[refs[2]][1] is True
