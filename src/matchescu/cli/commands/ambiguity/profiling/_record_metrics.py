from collections import Counter
from typing import Any

from ._context import ProfileContext
from ._stats import ratio, spread

SINGLETON_SIZE: int = 1


class RecordMetrics:
    """Record and cluster counts, and how bridges spread over clusters.

    Per-cluster bridge counts average over every cluster, zeros included,
    so they show how much of the dataset bridges reach.
    """

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        return {
            **self.__record_counts(context),
            **self.__cluster_counts(context),
            **self.__bridged_cluster_counts(context),
            **self.__per_bridge_spreads(context),
            **self.__per_cluster_spreads(context),
        }

    @staticmethod
    def __record_counts(context: ProfileContext) -> dict[str, Any]:
        n_records = len(context.profile_input.records)
        n_bridges = len(context.roles.bridges)
        return {
            "n_records": n_records,
            "n_original_records": n_records - n_bridges,
            "n_bridges": n_bridges,
            "bridge_share": ratio(n_bridges, n_records),
        }

    @staticmethod
    def __cluster_counts(context: ProfileContext) -> dict[str, Any]:
        sizes = [len(members) for members in context.profile_input.clusters.values()]
        return {
            "n_clusters": len(sizes),
            "n_singleton_clusters": sizes.count(SINGLETON_SIZE),
            **spread("cluster_size", sizes),
        }

    @staticmethod
    def __bridged_cluster_counts(context: ProfileContext) -> dict[str, Any]:
        footprints = context.footprints.values()
        homes = {f.home for f in footprints}
        targets = frozenset().union(*(f.targets for f in footprints))
        return {
            "n_home_clusters": len(homes),
            "n_target_clusters": len(targets),
            "n_unbridged_clusters": len(context.profile_input.clusters)
            - len(homes | targets),
        }

    @staticmethod
    def __per_bridge_spreads(context: ProfileContext) -> dict[str, Any]:
        footprints = context.footprints.values()
        return {
            **spread("home_clusters_per_bridge", [f.n_homes for f in footprints]),
            **spread(
                "target_clusters_per_bridge", [len(f.targets) for f in footprints]
            ),
        }

    @staticmethod
    def __per_cluster_spreads(context: ProfileContext) -> dict[str, Any]:
        footprints = context.footprints.values()
        homes = Counter(f.home for f in footprints)
        targets = Counter(t for f in footprints for t in f.targets)
        clusters = context.profile_input.clusters
        return {
            **spread("bridges_per_home_cluster", [homes[c] for c in clusters]),
            **spread("bridges_per_target_cluster", [targets[c] for c in clusters]),
        }
