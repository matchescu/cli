import warnings
from collections.abc import Callable, Mapping
from statistics import fmean
from typing import Any, ClassVar

import numpy as np
from matchescu.typing import EntityReferenceIdentifier as RefId

from .._gmm_fit import GmmFit, InsufficientScoresError
from ._context import ProfileContext
from ._stats import auc, avg, separability, spread

type ScoresByBridge = Mapping[RefId, list[float]]


class DirectionMetrics:
    """How much the pair text alone reveals the direction of a bridge link.

    Every bridge has source pairs (home member, bridge), labelled 1, and
    target pairs (target record, bridge), labelled 2. Both put the bridge on
    the right, so only the left record's origin differs: the content
    similarity AUC of source against target pairs measures how well text
    similarity alone tells the two apart. Separability ``|2 * AUC - 1|``
    reads the same whichever way the similarity leans.

    The per-bridge margin is the mean source similarity minus the mean
    target similarity. A two-component mixture fitted to the absolute
    margins splits bridges into an ambiguous and a resolvable population;
    `margin_pi_resolvable` is the weight of the resolvable one.
    """

    MARGIN_COLUMNS: ClassVar[tuple[str, ...]] = (
        "margin_pi_resolvable",
        "margin_theta",
        "margin_boundary_type",
    )
    TIE_SHIFT: ClassVar[float] = 1.0
    TIE_SCALE: ClassVar[float] = 2.0

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        score = context.profile_input.score
        source = self.__scored(context.roles.sources_by_bridge, score)
        target = self.__scored(context.roles.targets_by_bridge, score)
        extra = [score(left, right) for left, right in context.roles.extra]
        return {
            **self.__ranking(source, target, extra),
            **spread("sim_source", self.__flat(source)),
            **spread("sim_target", self.__flat(target)),
            **spread("sim_extra", extra),
            **self.__margins(source, target),
        }

    @staticmethod
    def __scored(
        others_by_bridge: Mapping[RefId, tuple[RefId, ...]],
        score: Callable[[RefId, RefId], float],
    ) -> dict[RefId, list[float]]:
        return {
            bridge: [score(other, bridge) for other in others]
            for bridge, others in others_by_bridge.items()
        }

    @staticmethod
    def __flat(scores: ScoresByBridge) -> list[float]:
        return [s for bridge_scores in scores.values() for s in bridge_scores]

    @staticmethod
    def __paired(source: ScoresByBridge, target: ScoresByBridge):
        return [(source[b], target[b]) for b in source if b in target]

    def __ranking(self, source, target, extra) -> dict[str, Any]:
        source_vs_target = auc(self.__flat(source), self.__flat(target))
        return {
            "content_auc_source_vs_target": source_vs_target,
            "content_separability": separability(source_vs_target),
            "content_auc_extra_vs_target": auc(extra, self.__flat(target)),
            "content_winrate_source_vs_target": self.__win_rate(source, target),
        }

    def __win_rate(self, source, target) -> float | None:
        outcomes = [
            (np.sign(fmean(s) - fmean(t)) + self.TIE_SHIFT) / self.TIE_SCALE
            for s, t in self.__paired(source, target)
        ]
        return avg(outcomes)

    def __margins(self, source, target) -> dict[str, Any]:
        margins = [abs(fmean(s) - fmean(t)) for s, t in self.__paired(source, target)]
        try:
            fit = GmmFit.from_scores(margins)
        except InsufficientScoresError as err:
            warnings.warn(f"margin mixture skipped: {err}")
            return dict.fromkeys(self.MARGIN_COLUMNS)
        values = (fit.pi_genuine, fit.threshold, fit.boundary_type)
        return dict(zip(self.MARGIN_COLUMNS, values, strict=True))
