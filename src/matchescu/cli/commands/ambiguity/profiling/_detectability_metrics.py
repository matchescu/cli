import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._context import ProfileContext
from ._grouped_cv import GroupedCrossValidation, InsufficientGroupsError
from ._stats import NEGATIVE, POSITIVE, auc, oriented_auc
from ._tokens import field_text


class DetectabilityMetrics:
    """How easily bridge records stand out from the original records.

    A model that spots synthetic records can shortcut the directional task
    (a pair holding a bridge is never a plain match). Record shape features
    over the compared fields — empty fields, token count, character count —
    are scored one by one, oriented so that 0.5 means undetectable and 1.0
    perfectly detectable, and jointly by a logistic regression grouped by
    bridge component (unbridged clusters form their own groups).
    """

    FEATURES: ClassVar[tuple[str, ...]] = ("n_empty_fields", "n_tokens", "n_chars")
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "detect_auc_best_single",
        "detect_best_feature",
        "detect_auc_lr",
    )

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        if not context.roles.bridges:
            warnings.warn("detectability skipped: the dataset has no bridges")
            return dict.fromkeys(self.COLUMNS)
        refs = list(context.profile_input.records)
        labels = [POSITIVE if r in context.roles.bridges else NEGATIVE for r in refs]
        features = self.__features(context, refs)
        best_feature, best_auc = self.__best_single(features, labels)
        values = (
            best_auc,
            best_feature,
            self.__model_auc(context, refs, features, labels),
        )
        return dict(zip(self.COLUMNS, values, strict=True))

    def __features(self, context: ProfileContext, refs: Sequence[RefId]) -> np.ndarray:
        fields = context.profile_input.compared_fields
        records = context.profile_input.records
        return np.array([self.__shape(records[r], fields) for r in refs], dtype=float)

    @staticmethod
    def __shape(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[int, ...]:
        texts = [field_text(record, f) for f in fields]
        return (
            sum(not text.strip() for text in texts),
            sum(len(text.split()) for text in texts),
            sum(len(text) for text in texts),
        )

    def __best_single(
        self, features: np.ndarray, labels: Sequence[int]
    ) -> tuple[str, float | None]:
        positive = np.array(labels) == POSITIVE
        oriented = [
            oriented_auc(auc(list(column[positive]), list(column[~positive])))
            for column in features.T
        ]
        best = max(range(len(oriented)), key=lambda i: oriented[i] or 0.0)
        return self.FEATURES[best], oriented[best]

    def __model_auc(
        self,
        context: ProfileContext,
        refs: Sequence[RefId],
        features: np.ndarray,
        labels: Sequence[int],
    ) -> float | None:
        cv = GroupedCrossValidation(context.options.cv_folds, context.options.seed)
        try:
            return cv.auc(features, labels, self.__groups(context, refs))
        except InsufficientGroupsError as err:
            warnings.warn(f"detectability model skipped: {err}")
            return None

    @staticmethod
    def __groups(context: ProfileContext, refs: Sequence[RefId]) -> list[Hashable]:
        cluster_of = context.profile_input.cluster_of
        groups: list[Hashable] = []
        for ref_id in refs:
            component = context.components.component_of(cluster_of[ref_id])
            groups.append(
                ("cluster", cluster_of[ref_id])
                if component is None
                else ("component", component)
            )
        return groups
