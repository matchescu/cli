from dataclasses import dataclass

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._cluster_reps import ClusterRankings
from ._greedy import GreedyPairing, Pairing
from ._input import PairingInput
from ._link_affinity import LinkAffinity
from ._seed import BridgeSeed
from ._strategy import PairingStrategy


@dataclass(frozen=True)
class _Prepared:
    rankings: dict[int, list[RefId]]
    affinity: LinkAffinity


class GreedyClusterReps(PairingStrategy):
    """Pair cluster representatives greedily, one bridge per home cluster.

    Each cluster nominates a representative and ranks its members by
    ambiguity. Homes are processed in cluster-id order; a home takes the
    available target cluster whose representative is most ambiguous with its
    own and links into the target's ranked members. A cluster accepted as a
    home can no longer be a target.

    :param pairing_input: cluster layout, ambiguity scores and progress sink.
    """

    def __init__(self, pairing_input: PairingInput) -> None:
        self.__input = pairing_input
        self.__prepared: _Prepared | None = None

    def seeds(self, min_score: float | None) -> list[BridgeSeed]:
        """Return one bridge seed per greedily paired home cluster.

        :param min_score: minimum score for a seed to be accepted. Defaults
            to ``None``, meaning every pairing is accepted.
        :returns: the accepted seeds, in home processing order.
        """
        prepared = self.__prepare()
        greedy_pairing = GreedyPairing()
        pairings = greedy_pairing(
            list(prepared.rankings),
            prepared.affinity,
            min_score,
            self.__on_home_done,
        )
        return [self.__seed(pairing, prepared) for pairing in pairings]

    def __prepare(self) -> _Prepared:
        if self.__prepared is None:
            compute_cluster_rankings = ClusterRankings.of(self.__input)
            rankings = compute_cluster_rankings()
            self.__prepared = _Prepared(
                rankings,
                LinkAffinity(rankings, self.__input.target_links, self.__input.score),
            )
        assert self.__prepared is not None
        return self.__prepared

    @staticmethod
    def __seed(pairing: Pairing, prepared: _Prepared) -> BridgeSeed:
        home = pairing.home_cluster_id
        target = pairing.target_cluster_id
        return BridgeSeed(
            home_cluster_id=home,
            home_ref=prepared.rankings[home][0],
            target_cluster_id=target,
            target_ref=prepared.rankings[target][0],
            linked_refs=tuple(prepared.affinity.linked_members(target)),
            is_short=prepared.affinity.is_short(target),
            score=pairing.score,
            label_key=home,
        )

    def __on_home_done(self, cluster_id: int) -> None:
        self.__input.on_progress(f"finding [{cluster_id}] closest cluster rep")
