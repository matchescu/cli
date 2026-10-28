from collections.abc import Callable
from operator import attrgetter
from typing import Any, ClassVar

from ._bridge_component_metrics import BridgeComponentMetrics
from ._bridge_text_metrics import BridgeTextMetrics
from ._calculator import ProfileCalculator
from ._class_metrics import ClassMetrics
from ._context import ProfileContext
from ._detectability_metrics import DetectabilityMetrics
from ._direction_metrics import DirectionMetrics
from ._input import ProfileInput
from ._negative_similarity_metrics import NegativeSimilarityMetrics
from ._options import ProfileOptions
from ._record_metrics import RecordMetrics
from ._surface_baseline_metrics import SurfaceBaselineMetrics

type Switch = Callable[[ProfileOptions], bool]


class Profiler:
    """Summarise an ambiguous dataset as one flat row of metrics.

    The row concatenates the columns of every core calculator, then of every
    optional calculator its options switch on, in registry order. Profiles
    made with the same options therefore share one schema and stack into a
    single metadata table.
    """

    CORE: ClassVar[tuple[ProfileCalculator, ...]] = (
        RecordMetrics(),
        ClassMetrics(),
        BridgeComponentMetrics(),
        BridgeTextMetrics(),
        DirectionMetrics(),
    )
    OPTIONAL: ClassVar[tuple[tuple[Switch, ProfileCalculator], ...]] = (
        (attrgetter("negative_similarity"), NegativeSimilarityMetrics()),
        (attrgetter("surface_baseline"), SurfaceBaselineMetrics()),
        (attrgetter("synthetic_detectability"), DetectabilityMetrics()),
    )

    def profile(
        self, profile_input: ProfileInput, options: ProfileOptions
    ) -> dict[str, Any]:
        """Compute the profile of one dataset.

        :param profile_input: the dataset to profile.
        :param options: settings of the profile, including which optional
            calculators run.
        :returns: column name to value; uncomputable values are ``None``.
        """
        context = ProfileContext.build(profile_input, options)
        row: dict[str, Any] = {}
        for calculator in self.__calculators(options):
            row.update(calculator(context))
        return row

    def __calculators(self, options: ProfileOptions) -> list[ProfileCalculator]:
        enabled = [calc for switch, calc in self.OPTIONAL if switch(options)]
        return [*self.CORE, *enabled]
