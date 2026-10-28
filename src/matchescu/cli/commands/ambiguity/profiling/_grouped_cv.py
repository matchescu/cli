from collections.abc import Hashable, Sequence
from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._stats import NEGATIVE, POSITIVE


class InsufficientGroupsError(ValueError):
    """Signal that a class appears in fewer groups than there are folds.

    Small datasets have few bridge components; that is an expected outcome
    of profiling real data, so callers can skip the baseline and report it.
    """


class GroupedCrossValidation:
    """Out-of-fold ROC AUC of a scaled logistic regression, grouped.

    Rows sharing a group always land in the same fold. Profiles group rows
    by bridge component, so a model is never evaluated on clusters it saw
    during training — the same constraint a leak-free split imposes.

    :param folds: number of folds.
    :param seed: seed of the fold assignment.
    """

    MAX_ITER: ClassVar[int] = 1000

    def __init__(self, folds: int, seed: int) -> None:
        self.__folds = folds
        self.__seed = seed

    def auc(
        self, features: np.ndarray, labels: Sequence[int], groups: Sequence[Hashable]
    ) -> float:
        """Return the out-of-fold AUC of predicting `labels` from `features`.

        :param features: one row of features per sample.
        :param labels: 1 for the positive class, 0 otherwise.
        :param groups: group of every sample.
        :returns: ROC AUC of the pooled out-of-fold probabilities.
        :raises InsufficientGroupsError: when either class occurs in fewer
            groups than there are folds.
        """
        codes = self.__codes(groups)
        self.__check_groups(labels, codes)
        splitter = StratifiedGroupKFold(
            n_splits=self.__folds, shuffle=True, random_state=self.__seed
        )
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=self.MAX_ITER)
        )
        scores = cross_val_predict(
            model, features, labels, groups=codes, cv=splitter, method="predict_proba"
        )[:, POSITIVE]
        return float(roc_auc_score(labels, scores))

    def __check_groups(self, labels: Sequence[int], codes: Sequence[int]) -> None:
        for label in (NEGATIVE, POSITIVE):
            n_groups = len(
                {c for c, y in zip(codes, labels, strict=True) if y == label}
            )
            if n_groups < self.__folds:
                raise InsufficientGroupsError(
                    f"class {label} occurs in {n_groups} group(s), "
                    f"{self.__folds} folds need at least as many"
                )

    @staticmethod
    def __codes(groups: Sequence[Hashable]) -> list[int]:
        code_of = {group: code for code, group in enumerate(dict.fromkeys(groups))}
        return [code_of[group] for group in groups]
