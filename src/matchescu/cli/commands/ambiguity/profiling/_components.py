from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._disjoint_sets import DisjointSets
from ._footprint import BridgeFootprint
from ._input import ProfileInput
from ._pair_roles import PairRoles


@dataclass(frozen=True)
class BridgeComponent:
    """Clusters held together by bridges; they cannot be split apart.

    A bridge and its target records must land in the same split, or the
    split leaks cluster identities. A component is the closure of that
    constraint: every cluster reachable through bridge footprints.

    :ivar clusters: ids of the clusters in the component.
    :ivar n_records: records in those clusters, bridges included.
    :ivar n_bridges: bridges whose home is in the component.
    :ivar n_label2: label-2 pairs of those bridges.
    """

    clusters: frozenset[Hashable]
    n_records: int
    n_bridges: int
    n_label2: int

    @property
    def n_clusters(self) -> int:
        """Number of clusters in the component."""
        return len(self.clusters)


class BridgeComponents:
    """All bridge components of a dataset, largest first.

    Components are ordered by cluster count, then label-2 pairs, then
    records, all descending; ties keep discovery order.

    :param components: the components, in any order.
    """

    def __init__(self, components: Iterable[BridgeComponent]) -> None:
        self.__components = tuple(sorted(components, key=self.__size_rank))

    @classmethod
    def from_input(cls, profile_input: ProfileInput) -> "BridgeComponents":
        """Compute the bridge components of a dataset.

        :param profile_input: dataset whose bridges link clusters.
        :returns: the components, largest first.
        """
        roles = PairRoles.from_input(profile_input)
        footprints = BridgeFootprint.of_bridges(roles, profile_input.cluster_of)
        return cls.from_footprints(footprints, profile_input.clusters)

    @classmethod
    def from_footprints(
        cls,
        footprints: Mapping[RefId, BridgeFootprint],
        clusters: Mapping[Hashable, frozenset[RefId]],
    ) -> "BridgeComponents":
        """Merge bridge footprints that share a cluster into components.

        :param footprints: footprint of every bridge.
        :param clusters: members of every cluster, used to count records.
        :returns: the components, largest first.
        """
        sets = DisjointSets()
        for footprint in footprints.values():
            sets.union(footprint.clusters)
        grouped: dict[Hashable, list[BridgeFootprint]] = defaultdict(list)
        for footprint in footprints.values():
            grouped[sets.find(footprint.home)].append(footprint)
        return cls(cls.__component(group, clusters) for group in grouped.values())

    @property
    def largest(self) -> BridgeComponent | None:
        """The component with the most clusters, or ``None`` without bridges."""
        return self.__components[0] if self.__components else None

    def rows(self) -> list[dict[str, Any]]:
        """Return one CSV-ready row per component, numbered from 1."""
        return [
            {
                "bridge_component": number,
                "n_clusters": component.n_clusters,
                "n_records": component.n_records,
                "n_bridges": component.n_bridges,
                "n_label2": component.n_label2,
            }
            for number, component in enumerate(self.__components, start=1)
        ]

    def __iter__(self) -> Iterator[BridgeComponent]:
        return iter(self.__components)

    def __len__(self) -> int:
        return len(self.__components)

    @staticmethod
    def __component(footprints, clusters) -> BridgeComponent:
        members = frozenset().union(*(f.clusters for f in footprints))
        return BridgeComponent(
            clusters=members,
            n_records=sum(len(clusters[cluster_id]) for cluster_id in members),
            n_bridges=len(footprints),
            n_label2=sum(f.n_links for f in footprints),
        )

    @staticmethod
    def __size_rank(component: BridgeComponent) -> tuple[int, int, int]:
        return (-component.n_clusters, -component.n_label2, -component.n_records)
