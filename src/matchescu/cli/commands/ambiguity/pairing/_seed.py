from collections.abc import Hashable
from dataclasses import dataclass

from matchescu.typing import EntityReferenceIdentifier as RefId


@dataclass(frozen=True)
class BridgeSeed:
    """Describes how a bridge record is synthesised, and how it is linked.

    A strategy decides which records a bridge blends; bridge synthesis and
    ground-truth derivation read every decision from here.

    :ivar home_cluster_id: id of the cluster the bridge is a part of.
    :ivar home_ref: the record the bridge resembles in the home cluster.
    :ivar target_cluster_id: id of the cluster the bridge links to.
    :ivar target_ref: the record in the target cluster resembled by the bridge.
    :ivar linked_refs: target cluster members that receive a directional
        link, in ranking order.
    :ivar is_short: whether the target cluster could not fill every link.
    :ivar score: pairing score the minimum-ambiguity threshold is compared
        against.
    :ivar label_key: input of the bridge label strategy; equal keys yield
        equal labels.
    """

    home_cluster_id: int
    home_ref: RefId
    target_cluster_id: int
    target_ref: RefId
    linked_refs: tuple[RefId, ...]
    is_short: bool
    score: float
    label_key: Hashable
