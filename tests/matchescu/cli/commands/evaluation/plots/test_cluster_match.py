from matchescu.cli.commands.evaluation._plots._cluster_match import match_clusters


def _fs(refs, *idx):
    return frozenset(refs[i] for i in idx)


def test_identical_clusterings_match_each_cluster_to_itself(refs):
    clusters = frozenset({_fs(refs, 0, 1), _fs(refs, 2, 3, 4)})

    assert match_clusters(clusters, clusters) == {c: c for c in clusters}


def test_merged_prediction_matches_the_true_cluster_with_largest_overlap(refs):
    small, large = _fs(refs, 0, 1), _fs(refs, 2, 3, 4)
    merged = _fs(refs, 0, 1, 2, 3, 4)

    result = match_clusters(frozenset({small, large}), frozenset({merged}))

    assert result[merged] == large


def test_split_prediction_matches_only_one_part_to_the_true_cluster(refs):
    whole = _fs(refs, 0, 1, 2, 3)
    big, small = _fs(refs, 0, 1, 2), _fs(refs, 3, 4)

    result = match_clusters(frozenset({whole}), frozenset({big, small}))

    assert result == {big: whole, small: None}


def test_prediction_without_overlap_is_unmatched(refs):
    predicted = _fs(refs, 3, 4)

    result = match_clusters(frozenset({_fs(refs, 0, 1)}), frozenset({predicted}))

    assert result[predicted] is None


def test_predicted_singleton_is_unmatched(refs):
    singleton = _fs(refs, 0)

    result = match_clusters(frozenset({_fs(refs, 0, 1)}), frozenset({singleton}))

    assert result[singleton] is None
