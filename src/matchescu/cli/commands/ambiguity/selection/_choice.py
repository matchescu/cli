from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class AlphaChoice:
    """Outcome of choosing an alpha for one dataset.

    :ivar alpha: the chosen alpha, or ``None`` when no alpha qualifies.
    :ivar reason: why the choice came out as it did; one of the class-level
        reason constants.
    :ivar n_feasible: number of alphas inside the separability bounds.
    """

    SELECTED: ClassVar[str] = "selected"
    NO_FEASIBLE_ALPHA: ClassVar[str] = "no_feasible_alpha"
    NO_BAYES_OPTIMAL_FIT: ClassVar[str] = "no_bayes_optimal_fit"
    NO_SURFACE_BASELINE: ClassVar[str] = "no_surface_baseline"
    NO_UNDETECTABLE_ALPHA: ClassVar[str] = "no_undetectable_alpha"

    alpha: float | None
    reason: str
    n_feasible: int
