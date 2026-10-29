import re
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import ClassVar

from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReferenceIdentifier as RefId
from stopwords import get_stopwords

from .._ref_ids import ref_id_sort_key
from ._scope import PairingScope

type Weights = dict[RefId, float]
type Adjacency = dict[RefId, Weights]


class TokenBlocking:
    """Propose cross-cluster record pairs by shared tokens, pruned to top ones.

    Records are blocked on the tokens of their ambiguity-target fields: a
    block is every record carrying one token. Blocks larger than
    `max_block_size` are dropped, so very common tokens cost no work. Each
    cross-cluster pair inside a block is weighted with an ARCS-style weight,
    the sum over shared blocks of one over the comparisons of that block.
    Each record then keeps its `neighbours` highest-weight partners, which
    makes the proposals deterministic and gives every record the same
    coverage. The result is oriented with the smaller cluster id as home.

    :param cluster_of: cluster id of every record.
    :param id_table: references of the records.
    :param target_fields: properties whose tokens form the blocks.
    :param neighbours: highest-weight partners kept per record, at least 1.
    :param scope: whether a pair must be a top partner of one record or of
        both.
    :param max_block_size: largest block kept; larger blocks are dropped.
    """

    TOKEN_PATTERN: ClassVar[re.Pattern] = re.compile(r"\b\w{2,}\b")

    def __init__(
        self,
        cluster_of: Mapping[RefId, int],
        id_table: IdTable,
        target_fields: tuple[str, ...],
        neighbours: int,
        scope: PairingScope,
        max_block_size: int,
    ) -> None:
        self.__cluster_of = cluster_of
        self.__id_table = id_table
        self.__target_fields = target_fields
        self.__neighbours = neighbours
        self.__scope = scope
        self.__max_block_size = max_block_size
        self.__stopwords = {word.lower() for word in get_stopwords("en")}

    def __call__(self) -> list[tuple[RefId, RefId]]:
        """Return the proposed record pairs, home side first.

        :returns: pairs of records of different clusters, in a deterministic
            order.
        """
        adjacency = self.__adjacency()
        top = {ref: self.__top(ref, weights) for ref, weights in adjacency.items()}
        kept = {
            self.__orient(left, right)
            for left, weights in adjacency.items()
            for right in weights
            if self.__keep(left, right, top)
        }
        return sorted(kept, key=self.__order)

    def __adjacency(self) -> Adjacency:
        adjacency: Adjacency = {}
        for block in self.__blocks():
            self.__add_block(block, adjacency)
        return adjacency

    def __blocks(self) -> Iterable[list[RefId]]:
        blocks: dict[str, list[RefId]] = {}
        for ref_id in sorted(self.__cluster_of, key=ref_id_sort_key):
            for token in sorted(self.__tokens(ref_id)):
                blocks.setdefault(token, []).append(ref_id)
        return [
            members
            for members in blocks.values()
            if 2 <= len(members) <= self.__max_block_size
        ]

    def __tokens(self, ref_id: RefId) -> set[str]:
        values = self.__id_table.get(ref_id).as_dict()
        tokens = {
            token
            for field in self.__target_fields
            if isinstance(value := values.get(field), str)
            for token in self.TOKEN_PATTERN.findall(value.lower())
        }
        return tokens - self.__stopwords

    def __add_block(self, block: list[RefId], adjacency: Adjacency) -> None:
        weight = 2 / (len(block) * (len(block) - 1))
        for left, right in combinations(block, 2):
            if self.__cluster_of[left] == self.__cluster_of[right]:
                continue
            adjacency.setdefault(left, {})
            adjacency.setdefault(right, {})
            adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
            adjacency[right][left] = adjacency[right].get(left, 0.0) + weight

    def __top(self, ref_id: RefId, weights: Weights) -> set[RefId]:
        ranked = sorted(
            weights, key=lambda other: (-weights[other], ref_id_sort_key(other))
        )
        return set(ranked[: self.__neighbours])

    def __keep(
        self, left: RefId, right: RefId, top: Mapping[RefId, set[RefId]]
    ) -> bool:
        is_top = right in top.get(left, set()), left in top.get(right, set())
        match self.__scope:
            case PairingScope.ANY:
                return any(is_top)
            case PairingScope.MUTUAL:
                return all(is_top)
        raise ValueError(f"unknown pairing scope: {self.__scope}")

    def __orient(self, left: RefId, right: RefId) -> tuple[RefId, RefId]:
        if self.__order((left, right)) <= self.__order((right, left)):
            return left, right
        return right, left

    def __order(self, edge: tuple[RefId, RefId]) -> tuple:
        home, target = edge
        return (
            self.__cluster_of[home],
            ref_id_sort_key(home),
            ref_id_sort_key(target),
        )
