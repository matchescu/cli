from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import ClassVar

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._pair_roles import PairRoles


@dataclass(frozen=True)
class BridgeFootprint:
    """Clusters one bridge touches: its home and the clusters it links into.

    :ivar home: cluster the bridge belongs to.
    :ivar targets: clusters holding the bridge's target records.
    :ivar n_links: number of label-2 pairs of the bridge.
    """

    HOMES_PER_BRIDGE: ClassVar[int] = 1

    home: Hashable
    targets: frozenset[Hashable]
    n_links: int

    @classmethod
    def of_bridges(
        cls, roles: PairRoles, cluster_of: Mapping[RefId, Hashable]
    ) -> dict[RefId, "BridgeFootprint"]:
        """Return the footprint of every bridge.

        :param roles: pair roles supplying the bridges and their targets.
        :param cluster_of: cluster id of every record.
        :returns: footprints keyed by bridge id.
        """
        return {
            bridge: cls(
                cluster_of[bridge],
                frozenset(cluster_of[t] for t in targets),
                len(targets),
            )
            for bridge, targets in roles.targets_by_bridge.items()
        }

    @property
    def clusters(self) -> frozenset[Hashable]:
        """Home and target clusters together."""
        return self.targets | {self.home}

    @property
    def n_homes(self) -> int:
        """Number of home clusters; a record belongs to exactly one cluster."""
        return self.HOMES_PER_BRIDGE
