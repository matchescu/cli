from collections.abc import Callable, Mapping, Sequence

from matchescu.typing import EntityReferenceIdentifier as RefId


class LinkAffinity:
    """Score a pair of clusters by the links a bridge would form in the target.

    Bridge records link to the first ``target_links`` members of the target
    cluster, in ranking order. The pairing score is the minimum ambiguity
    score between the home representative and those members, so a pairing is
    only as strong as its weakest link. With one link this is the score
    between the two representatives.

    The ranking's first entry is the cluster representative, so the home
    representative is always its own ranking's first entry. Scores are cached
    per cluster pair; the cache assumes `score` is deterministic.

    :param rankings: per-cluster ids of the members in ambiguity-ranking
        order, representative first.
    :param target_links: number of links a bridge forms inside its target
        cluster. Clusters with fewer members link to all of them.
    :param score: ambiguity score of two references, home side first.
    """

    def __init__(
        self,
        rankings: Mapping[int, Sequence[RefId]],
        target_links: int,
        score: Callable[[RefId, RefId], float],
    ) -> None:
        self.__rankings = rankings
        self.__target_links = target_links
        self.__score = score
        self.__cache: dict[tuple[int, int], float] = {}

    def __call__(self, home: int, target: int) -> float:
        """Score a home cluster against a target cluster.

        :param home: id of the cluster sending the bridge; its representative
            is the home side of every score.
        :param target: id of the cluster receiving the bridge.
        :returns: the minimum ambiguity score over the target's linked
            members.
        """
        key = (home, target)
        if key not in self.__cache:
            self.__cache[key] = min(
                self.__score(self.__rankings[home][0], member)
                for member in self.linked_members(target)
            )
        return self.__cache[key]

    def linked_members(self, target: int) -> Sequence[RefId]:
        """Return the target members a bridge would link to.

        :param target: id of the target cluster.
        :returns: the first ``target_links`` ranked members, or all members
            when the cluster is smaller.
        """
        return self.__rankings[target][: self.__target_links]

    def is_short(self, target: int) -> bool:
        """Report whether a target cluster cannot fill every link.

        :param target: id of the target cluster.
        :returns: ``True`` when the cluster has fewer members than links.
        """
        return len(self.__rankings[target]) < self.__target_links
