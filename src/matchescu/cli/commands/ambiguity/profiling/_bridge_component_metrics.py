from typing import Any, ClassVar

from ._components import BridgeComponent
from ._context import ProfileContext
from ._stats import ratio, spread


class BridgeComponentMetrics:
    """How bridges tie clusters together, and whether that fits a split.

    The split load is the share of the smallest split's label-2 quota the
    largest component takes up: its label-2 pairs over all label-2 pairs
    times the smallest split share. Above 1, a cluster-disjoint split cannot
    keep that component whole inside the smallest split.
    """

    LARGEST_COLUMNS: ClassVar[tuple[str, ...]] = (
        "largest_bridge_component_clusters",
        "largest_bridge_component_records",
        "largest_bridge_component_label2",
        "largest_bridge_component_split_load",
    )

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        components = context.components
        return {
            "bridge_component_count": len(components),
            **spread("bridge_component_size", [c.n_clusters for c in components]),
            **self.__largest(components.largest, context),
        }

    def __largest(
        self, largest: BridgeComponent | None, context: ProfileContext
    ) -> dict[str, Any]:
        if largest is None:
            return dict.fromkeys(self.LARGEST_COLUMNS)
        quota = len(context.roles.target) * context.options.split_ratio.smallest_share
        values = (
            largest.n_clusters,
            largest.n_records,
            largest.n_label2,
            ratio(largest.n_label2, quota),
        )
        return dict(zip(self.LARGEST_COLUMNS, values, strict=True))
