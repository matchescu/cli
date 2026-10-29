import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._blocking import TokenBlocking
from ._greedy_cluster_reps import GreedyClusterReps
from ._input import PairingInput
from ._scope import PairingScope
from ._simple_pairs import SimplePairs
from ._strategy import PairingStrategy


class PairingStrategyName(StrEnum):
    """Names of the pairing strategies a configuration may select.

    :ivar GREEDY_CLUSTER_REPS: one bridge per home cluster, built from cluster
        representatives.
    :ivar SIMPLE_PAIRS: one bridge per blocked pair of records.
    """

    GREEDY_CLUSTER_REPS = "greedy-cluster-reps"
    SIMPLE_PAIRS = "simple-pairs"


@dataclass(frozen=True)
class PairingSettings:
    """How records of different clusters are chosen for bridging.

    Only the parameters of the selected strategy are read; the others keep
    their defaults and are ignored.

    :ivar name: pairing strategy to use.
    :ivar neighbours: highest-weight partners kept per record by
        ``simple-pairs``, at least 1.
    :ivar scope: whether a pair must be a top partner of one record or of
        both, read by ``simple-pairs``.
    :ivar max_block_size: largest token block kept, read by ``simple-pairs``.
    """

    DEFAULT_NEIGHBOURS: ClassVar[int] = 1
    DEFAULT_MAX_BLOCK_SIZE: ClassVar[int] = 100

    name: PairingStrategyName = PairingStrategyName.GREEDY_CLUSTER_REPS
    neighbours: int = DEFAULT_NEIGHBOURS
    scope: PairingScope = PairingScope.ANY
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE

    def build(self, pairing_input: PairingInput) -> PairingStrategy:
        """Build the configured strategy.

        :param pairing_input: cluster layout, ambiguity scores and progress
            sink the strategy reads.
        :returns: a fresh strategy of the configured kind.
        """
        builders: dict[PairingStrategyName, Callable[[], PairingStrategy]] = {
            PairingStrategyName.GREEDY_CLUSTER_REPS: lambda: GreedyClusterReps(
                pairing_input
            ),
            PairingStrategyName.SIMPLE_PAIRS: lambda: SimplePairs(
                pairing_input, self.__blocking(pairing_input)
            ),
        }
        return builders[self.name]()

    def target_links_for(self, target_links: int) -> int:
        """Return the links per bridge the strategy can honour.

        ``simple-pairs`` links only the paired record, so it uses one link
        and warns when more were requested.

        :param target_links: configured links per bridge.
        :returns: the effective links per bridge.
        """
        match self.name:
            case PairingStrategyName.GREEDY_CLUSTER_REPS:
                return target_links
            case PairingStrategyName.SIMPLE_PAIRS:
                if target_links > 1:
                    warnings.warn(
                        "targetLinks is ignored by the simple-pairs strategy; using 1"
                    )
                return 1
        raise ValueError(f"unknown pairing strategy: {self.name}")

    def as_row(self) -> dict[str, Any]:
        """Return the pairing settings as metadata columns.

        Strategies ignore the parameters of the other strategy, so those
        columns are empty.

        :returns: column name to value.
        """
        row: dict[str, Any] = {"pairing_strategy": str(self.name)}
        match self.name:
            case PairingStrategyName.SIMPLE_PAIRS:
                return row | {
                    "pairing_neighbours": self.neighbours,
                    "pairing_scope": str(self.scope),
                    "pairing_max_block_size": self.max_block_size,
                }
            case PairingStrategyName.GREEDY_CLUSTER_REPS:
                return row | {
                    "pairing_neighbours": None,
                    "pairing_scope": None,
                    "pairing_max_block_size": None,
                }
        raise ValueError(f"unknown pairing strategy: {self.name}")

    def __blocking(
        self, pairing_input: PairingInput
    ) -> Callable[[], list[tuple[RefId, RefId]]]:
        cluster_of = {
            ref_id: cluster_id
            for cluster_id, ref_ids in pairing_input.clusters.items()
            for ref_id in ref_ids
        }
        return TokenBlocking(
            cluster_of,
            pairing_input.id_table,
            pairing_input.target_fields,
            self.neighbours,
            self.scope,
            self.max_block_size,
        )
