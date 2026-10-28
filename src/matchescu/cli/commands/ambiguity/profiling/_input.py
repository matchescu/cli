from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from matchescu.typing import EntityReferenceIdentifier as RefId

type RecordValues = Mapping[str, Any]
type RecordPair = tuple[RefId, RefId]


@dataclass(frozen=True)
class ProfileInput:
    """Everything a profile is computed from, independent of how it was made.

    A dataset is described by its records, its cluster mapping and its
    directed pairwise mapping, in which label 1 marks a match and label 2 a
    directional match ``(target, bridge)``. Bridges are therefore the right
    records of label-2 pairs, whatever produced them.

    :ivar records: field values of every record, bridges included.
    :ivar cluster_of: cluster id of every record.
    :ivar directed_gt: label of every positive pair (1 or 2).
    :ivar similarity: content similarity of two records' field values.
    :ivar bridge_log: optional per-bridge-value generation log; enables the
        bridge text metrics.
    :ivar link_log: optional per-target-link generation log; enables the
        link ambiguity metrics.
    """

    records: Mapping[RefId, RecordValues]
    cluster_of: Mapping[RefId, Hashable]
    directed_gt: Mapping[RecordPair, int]
    similarity: Callable[[RecordValues, RecordValues], float]
    bridge_log: Sequence[Mapping[str, Any]] = ()
    link_log: Sequence[Mapping[str, Any]] = ()

    @cached_property
    def clusters(self) -> dict[Hashable, frozenset[RefId]]:
        """Members of every cluster, keyed by cluster id."""
        members: dict[Hashable, set[RefId]] = defaultdict(set)
        for ref_id, cluster_id in self.cluster_of.items():
            members[cluster_id].add(ref_id)
        return {cluster_id: frozenset(refs) for cluster_id, refs in members.items()}

    def score(self, left: RefId, right: RefId) -> float:
        """Return the content similarity of two records, in that order."""
        return self.similarity(self.records[left], self.records[right])
