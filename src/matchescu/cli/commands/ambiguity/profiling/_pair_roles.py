from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._input import ProfileInput, RecordPair


@dataclass(frozen=True)
class PairRoles:
    """Positive pairs grouped by the role they play around bridges.

    Bridge pairs are oriented ``(other, bridge)``, so the bridge always sits
    on the right, as it does in the directional match.

    :ivar bridges: right records of the label-2 pairs.
    :ivar source: label-1 pairs joining a bridge to a home cluster member.
    :ivar target: label-2 pairs joining a target record to a bridge.
    :ivar extra: label-1 pairs between two original records.
    """

    MATCH: ClassVar[int] = 1
    DIRECTIONAL_MATCH: ClassVar[int] = 2

    bridges: frozenset[RefId]
    source: tuple[RecordPair, ...]
    target: tuple[RecordPair, ...]
    extra: tuple[RecordPair, ...]

    @classmethod
    def from_input(cls, profile_input: ProfileInput) -> "PairRoles":
        """Assign every positive pair of a dataset to its role.

        Label-1 pairs between two bridges play no role and are left out.

        :param profile_input: dataset whose directed mapping is split.
        :returns: the pairs by role.
        :raises ValueError: when a pair carries a label other than 1 or 2.
        """
        by_label = cls.__by_label(profile_input.directed_gt)
        target = tuple(by_label[cls.DIRECTIONAL_MATCH])
        bridges = frozenset(bridge for _, bridge in target)
        source, extra = cls.__split_matches(by_label[cls.MATCH], bridges)
        return cls(bridges, source, target, extra)

    @cached_property
    def sources_by_bridge(self) -> dict[RefId, tuple[RefId, ...]]:
        """Home cluster members matched to each bridge."""
        return self.__by_bridge(self.source)

    @cached_property
    def targets_by_bridge(self) -> dict[RefId, tuple[RefId, ...]]:
        """Target records linked to each bridge."""
        return self.__by_bridge(self.target)

    @classmethod
    def __by_label(cls, directed_gt: Mapping[RecordPair, int]) -> dict:
        pairs: dict[int, list[RecordPair]] = {cls.MATCH: [], cls.DIRECTIONAL_MATCH: []}
        for pair, label in directed_gt.items():
            if label not in pairs:
                raise ValueError(f"unsupported label {label!r} for pair {pair}")
            pairs[label].append(pair)
        return pairs

    @staticmethod
    def __split_matches(matches: Iterable[RecordPair], bridges: frozenset[RefId]):
        source, extra = [], []
        for left, right in matches:
            match (left in bridges, right in bridges):
                case (False, False):
                    extra.append((left, right))
                case (False, True):
                    source.append((left, right))
                case (True, False):
                    source.append((right, left))
        return tuple(source), tuple(extra)

    @staticmethod
    def __by_bridge(pairs: Iterable[RecordPair]) -> dict[RefId, tuple[RefId, ...]]:
        grouped: dict[RefId, list[RefId]] = defaultdict(list)
        for other, bridge in pairs:
            grouped[bridge].append(other)
        return {bridge: tuple(others) for bridge, others in grouped.items()}
