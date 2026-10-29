from collections.abc import Callable, Collection, Mapping

from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from .._ref_ids import ref_id_sort_key
from ._input import PairingInput


class ClusterRankings:
    """Rank every cluster's members by how ambiguous they are with their mates.

    The first entry of each ranking is the cluster representative: the member
    with the highest mean ambiguity against its cluster mates. The remaining
    entries follow in descending mean ambiguity; ties keep the member order.

    :param clusters: cluster id to member ids.
    :param id_table: references of every member.
    :param score_refs: ambiguity score of two references.
    :param on_progress: called once per cluster with a description.
    """

    def __init__(
        self,
        clusters: Mapping[int, Collection[RefId]],
        id_table: IdTable,
        score_refs: Callable[[EntityReference, EntityReference], float],
        on_progress: Callable[[str], None],
    ) -> None:
        self.__clusters = clusters
        self.__id_table = id_table
        self.__score_refs = score_refs
        self.__on_progress = on_progress

    @classmethod
    def of(cls, pairing_input: PairingInput) -> "ClusterRankings":
        return cls(
            pairing_input.clusters,
            pairing_input.id_table,
            pairing_input.score_refs,
            pairing_input.on_progress,
        )

    def __call__(self) -> dict[int, list[RefId]]:
        """Return the member ranking of every cluster, representative first.

        :returns: cluster id to ranked member ids.
        """
        return {
            cluster_id: self.__rank(cluster_id, ref_ids)
            for cluster_id, ref_ids in self.__clusters.items()
        }

    def __rank(self, cluster_id: int, ref_ids: Collection[RefId]) -> list[RefId]:
        self.__on_progress(f"nominating [{cluster_id}] cluster rep")
        candidates = sorted(ref_ids, key=ref_id_sort_key)
        refs = list(self.__id_table.get_all(candidates))
        if len(refs) == 1:
            return candidates
        mean_ambiguities = [
            self.__mean_ambiguity(ref, refs[:i] + refs[i + 1 :])
            for i, ref in enumerate(refs)
        ]
        rep_idx = self.__representative_index(mean_ambiguities)
        return self.__ranked_by_ambiguity(candidates, rep_idx, mean_ambiguities)

    def __mean_ambiguity(
        self, ref: EntityReference, others: list[EntityReference]
    ) -> float:
        scores = [self.__score_refs(ref, other) for other in others]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def __representative_index(mean_ambiguities: list[float]) -> int:
        rep_idx = 0
        max_ambiguity = 0.0
        for i, ambiguity in enumerate(mean_ambiguities):
            if max_ambiguity < ambiguity:
                max_ambiguity = ambiguity
                rep_idx = i
        return rep_idx

    @staticmethod
    def __ranked_by_ambiguity(
        candidates: list[RefId], rep_idx: int, mean_ambiguities: list[float]
    ) -> list[RefId]:
        ranking = [candidates[rep_idx]]
        others = sorted(
            (
                (mean_ambiguities[i], candidates[i])
                for i in range(len(candidates))
                if i != rep_idx
            ),
            key=lambda pair: -pair[0],
        )
        ranking.extend(ref_id for _, ref_id in others)
        return ranking
