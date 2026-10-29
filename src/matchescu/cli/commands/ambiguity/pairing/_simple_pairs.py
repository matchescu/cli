from collections.abc import Callable
from typing import ClassVar

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._input import PairingInput
from ._seed import BridgeSeed
from ._strategy import PairingStrategy


class SimplePairs(PairingStrategy):
    """Blend each proposed cross-cluster pair, one bridge per pair.

    Because this pairing strategy would typically have quadratic time complexity
    and because certain datasets might have many small clusters, we employ
    blocking to reduce the workload. The blocker decides which records are to
    be paired: only the proposed pairs may become bridge records that
    join their home cluster to the paired target's cluster. Pairs are scored
    lazily and cached, so every threshold reads the same population. This
    strategy does not require electing cluster representatives, and it may
    yield a higher number of bridge records, depending on configuration.

    :param pairing_input: cluster layout, ambiguity scores and progress sink.
    :param blocking: proposes the record pairs to bridge, home side first.
    """

    LABEL_SEPARATOR: ClassVar[str] = "\x1f"

    def __init__(
        self,
        pairing_input: PairingInput,
        blocking: Callable[[], list[tuple[RefId, RefId]]],
    ) -> None:
        self.__input = pairing_input
        self.__blocking = blocking
        self.__cluster_of = {
            ref_id: cluster_id
            for cluster_id, ref_ids in pairing_input.clusters.items()
            for ref_id in ref_ids
        }
        self.__scored: list[tuple[RefId, RefId, float]] | None = None

    def seeds(self, min_score: float | None) -> list[BridgeSeed]:
        """Return one bridge seed per proposed pair at or above a score.

        :param min_score: minimum score for a seed to be accepted. Defaults
            to ``None``, meaning every seed is accepted.
        :returns: the accepted seeds, in blocking order.
        """
        return [
            self.__seed(home, target, score)
            for home, target, score in self.__scored_pairs()
            if min_score is None or score >= min_score
        ]

    def __scored_pairs(self) -> list[tuple[RefId, RefId, float]]:
        if self.__scored is None:
            edges = self.__blocking()
            total = len(edges)
            scored = []
            for position, (home, target) in enumerate(edges, start=1):
                self.__input.on_progress(f"scoring pair {position}/{total}")
                scored.append((home, target, self.__input.score(home, target)))
            self.__scored = scored
        return self.__scored

    def __seed(self, home: RefId, target: RefId, score: float) -> BridgeSeed:
        return BridgeSeed(
            home_cluster_id=self.__cluster_of[home],
            home_ref=home,
            target_cluster_id=self.__cluster_of[target],
            target_ref=target,
            linked_refs=(target,),
            is_short=False,
            score=score,
            label_key=self.LABEL_SEPARATOR.join(
                map(
                    str,
                    (home.source, home.label, target.source, target.label),
                )
            ),
        )
