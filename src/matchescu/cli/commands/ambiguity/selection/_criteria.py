from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from .._gmm_fit import GmmFit
from ._choice import AlphaChoice

if TYPE_CHECKING:
    from ._settings import SelectionSettings

type SweepRow = Mapping[str, Any]

ALPHA: str = "alpha"


class AlphaCriterion(Protocol):
    """Pick one alpha among the feasible ones of a dataset's sweep.

    :cvar UNAVAILABLE: reason reported when the criterion finds no
        candidate it can rank.
    :cvar REQUIRED_OPTIONS: names of the `ProfileOptions` switches the
        criterion's columns depend on; empty when it needs none.
    """

    UNAVAILABLE: ClassVar[str]
    REQUIRED_OPTIONS: ClassVar[tuple[str, ...]]

    @classmethod
    def from_settings(cls, settings: "SelectionSettings") -> "AlphaCriterion":
        """Build the criterion from the selection settings."""
        ...

    def __call__(self, feasible: Sequence[SweepRow]) -> SweepRow | None:
        """Return the chosen row, or ``None`` when no row can be ranked.

        :param feasible: sweep rows inside the separability bounds; never
            empty.
        """
        ...


class SmallestAlpha:
    """The least lopsided bridge text that still carries enough direction."""

    UNAVAILABLE: ClassVar[str] = AlphaChoice.NO_FEASIBLE_ALPHA
    REQUIRED_OPTIONS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_settings(cls, settings: "SelectionSettings") -> "SmallestAlpha":
        return cls()

    def __call__(self, feasible: Sequence[SweepRow]) -> SweepRow | None:
        return min(feasible, key=lambda row: row[ALPHA])


class GmmBalance:
    """The alpha whose resolvable-bridge share is closest to a target.

    Items of medium difficulty discriminate best between models of
    different ability; with a target of 0.5, half the bridges are
    resolvable from text and half stay ambiguous. Only Bayes-optimal margin
    fits qualify; ties go to the smaller alpha.

    :param target_resolvable: desired resolvable-bridge share, in [0, 1].
    """

    UNAVAILABLE: ClassVar[str] = AlphaChoice.NO_BAYES_OPTIMAL_FIT
    REQUIRED_OPTIONS: ClassVar[tuple[str, ...]] = ()
    PI: ClassVar[str] = "margin_pi_resolvable"
    BOUNDARY: ClassVar[str] = "margin_boundary_type"

    def __init__(self, target_resolvable: float) -> None:
        self.__target = target_resolvable

    @classmethod
    def from_settings(cls, settings: "SelectionSettings") -> "GmmBalance":
        return cls(settings.target_resolvable)

    def __call__(self, feasible: Sequence[SweepRow]) -> SweepRow | None:
        fitted = [
            row
            for row in feasible
            if row.get(self.BOUNDARY) == GmmFit.BAYES_OPTIMAL
            and row.get(self.PI) is not None
        ]
        return min(fitted, key=self.__rank, default=None)

    def __rank(self, row: SweepRow) -> tuple[float, float]:
        return (abs(row[self.PI] - self.__target), row[ALPHA])


class SurfaceGap:
    """The alpha where order-aware surface features gain most over symmetric.

    A large gap means direction is carried by surface asymmetry, which
    order-sensitive architectures can exploit and symmetric ones cannot.
    Ties go to the smaller alpha.
    """

    UNAVAILABLE: ClassVar[str] = AlphaChoice.NO_SURFACE_BASELINE
    REQUIRED_OPTIONS: ClassVar[tuple[str, ...]] = ("surface_baseline",)
    GAP: ClassVar[str] = "surface_auc_gap"

    @classmethod
    def from_settings(cls, settings: "SelectionSettings") -> "SurfaceGap":
        return cls()

    def __call__(self, feasible: Sequence[SweepRow]) -> SweepRow | None:
        measured = [row for row in feasible if row.get(self.GAP) is not None]
        return min(measured, key=lambda row: (-row[self.GAP], row[ALPHA]), default=None)


class OrderHeadroom:
    """The largest surface gap among alphas whose bridges are hard to spot.

    A matcher that spots synthetic records solves the one-way label without
    reading direction, so alphas whose bridges a shape-only model detects
    above the cap are excluded. Among the rest, the largest gap between
    order-aware and swap-invariant surface features wins: that is the
    direction signal only an order-sensitive architecture can use. Ties go
    to the smaller alpha.

    :param max_detect_auc: highest bridge detectability (``detect_auc_lr``)
        an alpha may have; alphas without a detectability value never
        qualify.
    """

    UNAVAILABLE: ClassVar[str] = AlphaChoice.NO_UNDETECTABLE_ALPHA
    REQUIRED_OPTIONS: ClassVar[tuple[str, ...]] = (
        "surface_baseline",
        "synthetic_detectability",
    )
    DETECTABILITY: ClassVar[str] = "detect_auc_lr"

    def __init__(self, max_detect_auc: float) -> None:
        self.__max_detect_auc = max_detect_auc
        self.__surface_gap = SurfaceGap()

    @classmethod
    def from_settings(cls, settings: "SelectionSettings") -> "OrderHeadroom":
        return cls(settings.max_detect_auc)

    def __call__(self, feasible: Sequence[SweepRow]) -> SweepRow | None:
        hidden = [row for row in feasible if self.__is_hidden(row)]
        return self.__surface_gap(hidden)

    def __is_hidden(self, row: SweepRow) -> bool:
        detectability = row.get(self.DETECTABILITY)
        return detectability is not None and detectability <= self.__max_detect_auc


ALPHA_CRITERIA: dict[str, type[AlphaCriterion]] = {
    "smallest": SmallestAlpha,
    "gmm-balance": GmmBalance,
    "surface-gap": SurfaceGap,
    "order-headroom": OrderHeadroom,
}
