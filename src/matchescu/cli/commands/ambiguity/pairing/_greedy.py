from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from operator import attrgetter


@dataclass(frozen=True)
class Pairing:
    """Pairing of a home cluster with the target cluster its bridge points to.

    The home cluster receives the bridge record; the bridge links into the
    target cluster.

    :ivar home_cluster_id: id of the cluster the bridge record joins.
    :ivar target_cluster_id: id of the cluster the bridge record links to.
    :ivar score: ambiguity score of the pairing as measured by the affinity;
        the value a minimum-ambiguity threshold is compared against.
    """

    home_cluster_id: int
    target_cluster_id: int
    score: float


class ClusterPairing(ABC):
    """Select a target cluster for every home cluster that gets a bridge.

    Cluster-level algorithms are the building block of the
    ``greedy-cluster-reps`` strategy: they pick target clusters only, and the
    strategy turns each pairing into a bridge seed. Every algorithm shares the
    same call signature; algorithm-specific settings belong in the
    constructor.
    """

    @abstractmethod
    def __call__(
        self,
        cluster_ids: Sequence[int],
        affinity: Callable[[int, int], float],
        min_score: float | None = None,
        on_home_done: Callable[[int], None] | None = None,
    ) -> list[Pairing]:
        """Pair home clusters with target clusters.

        :param cluster_ids: ids of all clusters, in the order in which they
            are processed as home clusters.
        :param affinity: scores a ``(home, candidate)`` cluster pair; it may
            be expensive, so strategies should call it as rarely as possible.
        :param min_score: minimum score for a pairing to be accepted. Defaults
            to ``None``, meaning every pairing is accepted.
        :param on_home_done: called with each home cluster id once that home
            has been processed; lets callers report progress without the
            strategy knowing about progress displays. Defaults to ``None``,
            meaning no notification.
        :returns: at most one pairing per home, in home processing order; a
            home without an accepted pairing is absent.
        """


class GreedyPairing(ClusterPairing):
    """Pair each home with its highest-scoring available cluster, in order.

    Homes are processed in the given order. A home's candidates are all other
    clusters except those already accepted as homes, so a cluster that
    received a bridge cannot also be a target. The candidate with the highest
    affinity wins; ties go to the candidate that comes first in processing
    order. The pairing is accepted when there is no threshold or the score
    reaches it. A rejected home stays available as a target for later homes,
    which is why bridge counts are not necessarily monotonic in the threshold.
    """

    def __call__(
        self,
        cluster_ids: Sequence[int],
        affinity: Callable[[int, int], float],
        min_score: float | None = None,
        on_home_done: Callable[[int], None] | None = None,
    ) -> list[Pairing]:
        """Pair home clusters greedily.

        :param cluster_ids: ids of all clusters, in home processing order.
        :param affinity: scores a ``(home, candidate)`` cluster pair; called
            once per available candidate of each home.
        :param min_score: minimum score for a pairing to be accepted.
            Defaults to ``None``, meaning every pairing is accepted.
        :param on_home_done: called with each home id once it is processed.
            Defaults to ``None``.
        :returns: the accepted pairings, in home processing order.
        """
        pairings: list[Pairing] = []
        homes: set[int] = set()
        for home in cluster_ids:
            candidates = self._candidates(home, cluster_ids, affinity, homes)
            best = max(candidates, key=attrgetter("score"), default=None)
            if best is not None and (min_score is None or best.score >= min_score):
                pairings.append(best)
                homes.add(home)
            if on_home_done is not None:
                on_home_done(home)
        return pairings

    @staticmethod
    def _candidates(
        home: int,
        cluster_ids: Sequence[int],
        affinity: Callable[[int, int], float],
        homes: set[int],
    ) -> Iterator[Pairing]:
        return (
            Pairing(home, cluster_id, affinity(home, cluster_id))
            for cluster_id in cluster_ids
            if cluster_id != home and cluster_id not in homes
        )
