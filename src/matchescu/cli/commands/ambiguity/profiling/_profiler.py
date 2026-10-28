from typing import Any, ClassVar

from ._bridge_component_metrics import BridgeComponentMetrics
from ._bridge_text_metrics import BridgeTextMetrics
from ._calculator import ProfileCalculator
from ._class_metrics import ClassMetrics
from ._context import ProfileContext
from ._direction_metrics import DirectionMetrics
from ._input import ProfileInput
from ._options import ProfileOptions
from ._record_metrics import RecordMetrics


class Profiler:
    """Summarise an ambiguous dataset as one flat row of metrics.

    The row concatenates the columns of every calculator, in calculator
    order, so profiles of different datasets share one schema and stack
    into a single metadata table.
    """

    CORE: ClassVar[tuple[ProfileCalculator, ...]] = (
        RecordMetrics(),
        ClassMetrics(),
        BridgeComponentMetrics(),
        BridgeTextMetrics(),
        DirectionMetrics(),
    )

    def profile(
        self, profile_input: ProfileInput, options: ProfileOptions
    ) -> dict[str, Any]:
        """Compute the profile of one dataset.

        :param profile_input: the dataset to profile.
        :param options: settings of the profile.
        :returns: column name to value; uncomputable values are ``None``.
        """
        context = ProfileContext.build(profile_input, options)
        row: dict[str, Any] = {}
        for calculator in self.CORE:
            row.update(calculator(context))
        return row
