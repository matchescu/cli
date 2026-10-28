from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

type ComparisonData = tuple[RefId, RefId]

RESERVED_BRIDGE_PROPERTIES: frozenset[str] = frozenset(
    {"source", "target", "target_cluster_id"}
)
"""Bridge properties documenting the synthesis rather than the record."""


def ref_id_sort_key(ref_id: RefId) -> tuple:
    """Sort key ordering reference ids by source, then label."""
    return (ref_id.source, ref_id.label)


@dataclass(frozen=True)
class AmbiguousVariant:
    """One alpha variant of an ambiguous dataset, as produced in a single run.

    A variant is self-contained: it holds the bridges, every ground truth
    derived from them and the logs describing how they were synthesised, so
    writers and profilers need nothing else from the generator.

    :ivar alpha: blend weight of the harmonic-mean pick that produced it.
    :ivar bridges: the synthesised bridge records, in home-cluster order;
        their reserved properties (see `RESERVED_BRIDGE_PROPERTIES`) point
        at the home representative, the target representative and the
        target cluster id.
    :ivar cluster_gt: cluster id to member ids, bridges included in their
        home cluster.
    :ivar directed_gt: intra-cluster pairs labelled 1 and ``(target, bridge)``
        pairs labelled 2.
    :ivar undirected_gt: the same pairs as `directed_gt`, all labelled 1.
    :ivar bridge_log: one entry per synthesised bridge value (`alpha`,
        `bridge_label`, `home_cluster`, `target_cluster`, `field`, `value`,
        `value_no_floor`, `sim_home`, `sim_target`, `share_home`,
        `fallback`); `value_no_floor` is what the pick would have been
        without the target floor.
    :ivar link_log: one entry per bridge target link (`alpha`,
        `bridge_label`, `target_cluster`, `target_label`, `rank`,
        `ambiguity`).
    """

    alpha: float
    bridges: tuple[EntityReference, ...]
    cluster_gt: Mapping[int, set[RefId]]
    directed_gt: Mapping[ComparisonData, int]
    undirected_gt: Mapping[ComparisonData, int]
    bridge_log: tuple[Mapping[str, Any], ...]
    link_log: tuple[Mapping[str, Any], ...]
