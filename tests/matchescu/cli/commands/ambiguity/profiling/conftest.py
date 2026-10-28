"""Hand-built ambiguous dataset whose profile is computable by hand.

Clusters (bridges are the ``b*`` records, listed last in their home cluster)::

    1: a1 a2 b1     2: t1 t2     3: u1     4: e1 b2
    5: f1 f2 b3     6: g1        7: h1

Target links (label 2): b1 -> t1, t2 (cluster 2); b2 -> t1 (cluster 2) and
u1 (cluster 3); b3 -> h1 (cluster 7). Bridge components are therefore
{1, 2, 3, 4} and {5, 7}; cluster 6 is unbridged.
"""

from itertools import combinations

import pytest
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.commands.ambiguity.profiling import ProfileInput, ProfileOptions
from matchescu.cli.commands.ambiguity.profiling._context import ProfileContext

BRIDGE_SOURCE = "bridge"
ORIGINAL_SOURCE = "original"
LABEL = "label"

CLUSTERS = {
    1: ("a1", "a2", "b1"),
    2: ("t1", "t2"),
    3: ("u1",),
    4: ("e1", "b2"),
    5: ("f1", "f2", "b3"),
    6: ("g1",),
    7: ("h1",),
}
TARGET_LINKS = (("t1", "b1"), ("t2", "b1"), ("t1", "b2"), ("u1", "b2"), ("h1", "b3"))
SIMILARITIES = {
    ("a1", "b1"): 0.9,
    ("a2", "b1"): 0.7,
    ("e1", "b2"): 0.6,
    ("f1", "b3"): 0.2,
    ("f2", "b3"): 0.4,
    ("t1", "b1"): 0.3,
    ("t2", "b1"): 0.5,
    ("t1", "b2"): 0.8,
    ("u1", "b2"): 0.6,
    ("h1", "b3"): 0.4,
    ("a1", "a2"): 1.0,
    ("t1", "t2"): 0.1,
    ("f1", "f2"): 0.5,
}


def ref(label: str) -> RefId:
    return RefId(label, BRIDGE_SOURCE if label.startswith("b") else ORIGINAL_SOURCE)


def table_similarity(left, right) -> float:
    return SIMILARITIES.get((left[LABEL], right[LABEL]), 0.0)


def _directed_gt(clusters, target_links):
    directed = {
        (ref(left), ref(right)): 1
        for members in clusters.values()
        for left, right in combinations(members, 2)
    }
    directed.update({(ref(t), ref(b)): 2 for t, b in target_links})
    return directed


@pytest.fixture
def make_input():
    def _make(clusters=CLUSTERS, target_links=TARGET_LINKS, **logs):
        return ProfileInput(
            records={
                ref(label): {LABEL: label}
                for members in clusters.values()
                for label in members
            },
            cluster_of={
                ref(label): cid
                for cid, members in clusters.items()
                for label in members
            },
            directed_gt=_directed_gt(clusters, target_links),
            similarity=table_similarity,
            **logs,
        )

    return _make


@pytest.fixture
def profile_input(make_input):
    return make_input()


@pytest.fixture
def make_context(make_input):
    def _make(options=None, **input_kwargs):
        return ProfileContext.build(
            make_input(**input_kwargs), options or ProfileOptions()
        )

    return _make


@pytest.fixture
def context(make_context):
    return make_context()


@pytest.fixture
def unbridged_context(make_context):
    return make_context(clusters={1: ("a1", "a2"), 2: ("c1",)}, target_links=())


@pytest.fixture
def single_bridge_context(make_context):
    return make_context(
        clusters={1: ("a1", "b1"), 2: ("t1",)}, target_links=(("t1", "b1"),)
    )
