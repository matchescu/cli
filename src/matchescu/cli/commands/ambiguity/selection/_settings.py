from dataclasses import dataclass, field
from typing import Any, ClassVar

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
    :ivar max_detect_auc: highest bridge detectability the
        ``order-headroom`` criterion accepts, in [0.5, 1].
    :raises ValueError: when the criterion is unknown, `target_resolvable`
        lies outside [0, 1] or `max_detect_auc` lies outside [0.5, 1].
    """

    DEFAULT_CRITERION: ClassVar[str] = "smallest"
    DEFAULT_TARGET_RESOLVABLE: ClassVar[float] = 0.5
    DEFAULT_MAX_DETECT_AUC: ClassVar[float] = 0.7

    bounds: SeparabilityBounds = field(default_factory=SeparabilityBounds)
    criterion: str = DEFAULT_CRITERION
    target_resolvable: float = DEFAULT_TARGET_RESOLVABLE
    max_detect_auc: float = DEFAULT_MAX_DETECT_AUC

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
        if not SeparabilityBounds.CHANCE_AUC <= self.max_detect_auc <= 1:
            raise ValueError(
                f"max_detect_auc must be in [0.5, 1], got {self.max_detect_auc}"
            )

    def as_row(self) -> dict[str, Any]:
        """Return the choice parameters as report columns."""
        return {
            "alpha_criterion": self.criterion,
            "min_separability": self.bounds.min_separability,
            "max_auc": self.bounds.max_auc,
            "target_resolvable": self.target_resolvable,
            "max_detect_auc": self.max_detect_auc,
        }
