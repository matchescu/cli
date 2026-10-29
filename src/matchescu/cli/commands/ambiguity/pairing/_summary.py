from collections.abc import Sequence
from dataclasses import dataclass

from ._seed import BridgeSeed


@dataclass(frozen=True)
class PairingSummary:
    """Bridge and directional-link volume of one set of bridge seeds.

    :ivar bridge_count: number of seeds, hence bridges.
    :ivar link_count: total one-way links the bridges form inside their
        target clusters.
    :ivar short_bridge_count: bridges whose target cluster cannot fill every
        link.
    """

    bridge_count: int
    link_count: int
    short_bridge_count: int

    @classmethod
    def of(cls, seeds: Sequence[BridgeSeed]) -> "PairingSummary":
        """Summarise the links a set of bridge seeds would produce.

        :param seeds: accepted seeds of one threshold.
        :returns: the bridge and link counts of those seeds.
        """
        return cls(
            bridge_count=len(seeds),
            link_count=sum(len(seed.linked_refs) for seed in seeds),
            short_bridge_count=sum(seed.is_short for seed in seeds),
        )
