from dataclasses import dataclass, field
from typing import ClassVar

from ._bounds import SeparabilityBounds
from ._criteria import ALPHA_CRITERIA


@dataclass(frozen=True)
class SelectionSettings:
    """How an alpha is chosen from a dataset's sweep.

    :ivar bounds: separability band an alpha must fall into.
    :ivar criterion: name of the criterion choosing among feasible alphas;
        a key of `ALPHA_CRITERIA`.
    :ivar target_resolvable: resolvable-bridge share the ``gmm-balance``
        criterion aims for, in [0, 1].
    :raises ValueError: when the criterion is unknown or
        `target_resolvable` lies outside [0, 1].
    """

    DEFAULT_CRITERION: ClassVar[str] = "smallest"
    DEFAULT_TARGET_RESOLVABLE: ClassVar[float] = 0.5

    bounds: SeparabilityBounds = field(default_factory=SeparabilityBounds)
    criterion: str = DEFAULT_CRITERION
    target_resolvable: float = DEFAULT_TARGET_RESOLVABLE

    def __post_init__(self) -> None:
        if self.criterion not in ALPHA_CRITERIA:
            raise ValueError(
                f"unknown alpha criterion {self.criterion!r}; "
                f"expected one of {sorted(ALPHA_CRITERIA)}"
            )
        if not 0 <= self.target_resolvable <= 1:
            raise ValueError(
                f"target_resolvable must be in [0, 1], got {self.target_resolvable}"
            )
