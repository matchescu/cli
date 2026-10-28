from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SeparabilityBounds:
    """Band of content separability an alpha must fall into to be usable.

    Below `min_separability` the pair text carries too little direction for
    a matcher to learn from; above the separability equivalent of `max_auc`
    plain text similarity already solves the direction task, so it no longer
    discriminates between matcher architectures. Both ends are inclusive.

    :ivar min_separability: lowest acceptable ``|2 * AUC - 1|``, in [0, 1].
    :ivar max_auc: highest acceptable content AUC in its stronger direction,
        in (0.5, 1].
    :raises ValueError: when a bound lies outside its range or the band is
        empty.
    """

    DEFAULT_MIN_SEPARABILITY: ClassVar[float] = 0.2
    DEFAULT_MAX_AUC: ClassVar[float] = 0.9
    CHANCE_AUC: ClassVar[float] = 0.5

    min_separability: float = DEFAULT_MIN_SEPARABILITY
    max_auc: float = DEFAULT_MAX_AUC

    def __post_init__(self) -> None:
        if not 0 <= self.min_separability <= 1:
            raise ValueError(
                f"min_separability must be in [0, 1], got {self.min_separability}"
            )
        if not self.CHANCE_AUC < self.max_auc <= 1:
            raise ValueError(f"max_auc must be in (0.5, 1], got {self.max_auc}")
        if self.min_separability > self.max_separability:
            raise ValueError(
                f"min_separability {self.min_separability} exceeds the "
                f"separability {self.max_separability:.4g} of max_auc {self.max_auc}"
            )

    @property
    def max_separability(self) -> float:
        """Separability equivalent of `max_auc`: ``2 * max_auc - 1``."""
        return 2 * self.max_auc - 1

    def contains(self, separability: float | None) -> bool:
        """Return whether `separability` lies inside the band.

        :param separability: separability of one alpha; ``None`` (not
            computable) is never inside.
        """
        if separability is None:
            return False
        return self.min_separability <= separability <= self.max_separability
