from abc import ABC, abstractmethod
from collections.abc import Sequence

from ._seed import BridgeSeed
from ._summary import PairingSummary


class PairingStrategy(ABC):
    """Decide which records of different clusters a bridge blends.

    The ambiguity generator delegates every record selection to a strategy so
    that different rules can be swapped in without touching bridge synthesis,
    ground-truth derivation or threshold analysis. Each strategy is built
    from a :class:`PairingInput`; its own settings belong in its constructor.

    :raises ValueError: when the strategy is built from inconsistent input,
        such as a cluster id missing from the cluster layout.
    """

    @abstractmethod
    def seeds(self, min_score: float | None) -> list[BridgeSeed]:
        """Return the bridge seeds at or above a pairing score.

        Calls are deterministic: the same input yields the same seeds, in the
        same order.

        :param min_score: minimum score for a seed to be accepted. Defaults to
            ``None``, meaning every seed is accepted.
        :returns: the accepted seeds, in bridge order.
        """

    def scores(self) -> list[float]:
        """Return the pairing score of every seed made without the filter."""
        return [seed.score for seed in self.seeds(None)]

    def summaries(self, min_scores: Sequence[float | None]) -> list[PairingSummary]:
        """Summarise the bridges and links each threshold would produce.

        Strategies whose accepted set depends only on the seed scores inherit
        this implementation; others override it.

        :param min_scores: thresholds to evaluate, each standing in for the
            minimum pairing score.
        :returns: one summary per threshold, in the same order.
        """
        return [PairingSummary.of(self.seeds(t)) for t in min_scores]
