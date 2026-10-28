from dataclasses import dataclass, field
from typing import ClassVar

from ._split_ratio import SplitRatio


@dataclass(frozen=True)
class ProfileOptions:
    """Settings that change what a profile computes.

    The core profile is always computed. The supplementary baselines are
    switched on individually because they cost more and typically belong in
    a supplementary data package rather than in a paper.

    :ivar split_ratio: split sizes the bridge component split load is
        measured against.
    :ivar negative_similarity: compute the similarity of sampled implicit
        label-0 pairs against the positives.
    :ivar neg_pos_ratio: label-0 pairs sampled per positive pair, as the
        split configuration's ``negPosRatio``; must be positive.
    :ivar surface_baseline: compute the surface-feature direction baseline.
    :ivar synthetic_detectability: compute how detectable bridges are.
    :ivar seed: seed of every random choice (sampling, fold assignment).
    :ivar cv_folds: folds of the grouped cross-validation, at least 2.
    :raises ValueError: when `neg_pos_ratio` is not positive or `cv_folds`
        is below 2.
    """

    DEFAULT_SEED: ClassVar[int] = 42
    DEFAULT_CV_FOLDS: ClassVar[int] = 5
    MIN_CV_FOLDS: ClassVar[int] = 2
    DEFAULT_NEG_POS_RATIO: ClassVar[float] = 4.0

    split_ratio: SplitRatio = field(default_factory=SplitRatio.default)
    negative_similarity: bool = False
    neg_pos_ratio: float = DEFAULT_NEG_POS_RATIO
    surface_baseline: bool = False
    synthetic_detectability: bool = False
    seed: int = DEFAULT_SEED
    cv_folds: int = DEFAULT_CV_FOLDS

    def __post_init__(self) -> None:
        if self.neg_pos_ratio <= 0:
            raise ValueError(f"neg_pos_ratio must be > 0, got {self.neg_pos_ratio}")
        if self.cv_folds < self.MIN_CV_FOLDS:
            raise ValueError(
                f"cv_folds must be >= {self.MIN_CV_FOLDS}, got {self.cv_folds}"
            )
