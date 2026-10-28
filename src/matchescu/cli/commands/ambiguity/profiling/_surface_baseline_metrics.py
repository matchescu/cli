import warnings
from typing import Any, ClassVar

from ._context import ProfileContext
from ._grouped_cv import GroupedCrossValidation, InsufficientGroupsError
from ._pair_surface_features import PairSurfaceFeatures
from ._stats import NEGATIVE, POSITIVE


class SurfaceBaselineMetrics:
    """How much direction a surface-feature model recovers from pair text.

    A logistic regression tells target pairs (label 2, positive) from source
    pairs (label 1) using token overlap features per compared field and for
    the whole record. It is fitted twice: on swap-invariant features and on
    order-aware ones. Their AUC gap measures direction carried by surface
    asymmetry, the kind of signal order-sensitive architectures can exploit
    and symmetric ones cannot. Folds are grouped by bridge component.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "surface_auc_symmetric",
        "surface_auc_full",
        "surface_auc_gap",
    )

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        roles = context.roles
        pairs = [*roles.source, *roles.target]
        labels = [NEGATIVE] * len(roles.source) + [POSITIVE] * len(roles.target)
        groups = [
            context.components.component_of(context.profile_input.cluster_of[bridge])
            for _, bridge in pairs
        ]
        features = PairSurfaceFeatures(context.profile_input, pairs)
        cv = GroupedCrossValidation(context.options.cv_folds, context.options.seed)
        try:
            symmetric = cv.auc(features.symmetric(), labels, groups)
            full = cv.auc(features.full(), labels, groups)
        except InsufficientGroupsError as err:
            warnings.warn(f"surface baseline skipped: {err}")
            return dict.fromkeys(self.COLUMNS)
        values = (symmetric, full, full - symmetric)
        return dict(zip(self.COLUMNS, values, strict=True))
