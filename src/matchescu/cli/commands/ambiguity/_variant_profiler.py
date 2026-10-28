from collections.abc import Callable, Mapping
from typing import Any

from matchescu.reference_store.id_table import IdTable

from ._variant import AmbiguousVariant
from .profiling import (
    BridgeComponents,
    ProfileInput,
    ProfileOptions,
    Profiler,
)


class VariantProfiler:
    """Profile generated variants of one source dataset.

    Translates a variant into the neutral profiling input: original records
    plus bridges, the variant's cluster mapping, its directed mapping and
    its generation logs.

    :param id_table: original records of the source dataset.
    :param similarity: content similarity used by the direction metrics.
    :param options: settings of every profile.
    """

    def __init__(
        self,
        id_table: IdTable,
        similarity: Callable[[Mapping, Mapping], float],
        options: ProfileOptions,
    ) -> None:
        self.__records = {ref.id: ref.as_dict() for ref in id_table}
        self.__similarity = similarity
        self.__options = options
        self.__profiler = Profiler()

    def profile(self, variant: AmbiguousVariant) -> dict[str, Any]:
        """Return the metadata row of one variant (profile columns only)."""
        return self.__profiler.profile(self.__input(variant), self.__options)

    def components(self, variant: AmbiguousVariant) -> BridgeComponents:
        """Return the bridge components of one variant."""
        return BridgeComponents.from_input(self.__input(variant))

    def __input(self, variant: AmbiguousVariant) -> ProfileInput:
        return ProfileInput(
            records=self.__all_records(variant),
            cluster_of={
                ref_id: cluster_id
                for cluster_id, ref_ids in variant.cluster_gt.items()
                for ref_id in ref_ids
            },
            directed_gt=variant.directed_gt,
            similarity=self.__similarity,
            bridge_log=variant.bridge_log,
            link_log=variant.link_log,
        )

    def __all_records(self, variant: AmbiguousVariant) -> dict[Any, Mapping[str, Any]]:
        bridges = {bridge.id: bridge.as_dict() for bridge in variant.bridges}
        return {**self.__records, **bridges}
