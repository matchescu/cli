from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ._context import ProfileContext
from ._stats import quartiles, spread

type LogEntries = Sequence[Mapping[str, Any]]


class BridgeTextMetrics:
    """How synthesised bridge values lean towards home or target.

    Reads the optional generation logs. Per-value scores are text
    similarities, so they are summarised by median and interquartile range;
    link ambiguity scores are summarised by mean and standard deviation.
    Without logs every column is ``None``.
    """

    FIELD_SCORES: ClassVar[tuple[str, ...]] = ("share_home", "sim_home", "sim_target")
    VALUE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "n_fallback_values",
        "floor_changed_share",
    )

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        bridge_log = context.profile_input.bridge_log
        ambiguities = [e["ambiguity"] for e in context.profile_input.link_log]
        return {
            **self.__field_scores(bridge_log),
            **self.__value_counts(bridge_log),
            **spread("link_ambiguity", ambiguities),
        }

    def __field_scores(self, bridge_log: LogEntries) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        for score in self.FIELD_SCORES:
            present = [e[score] for e in bridge_log if e[score] is not None]
            columns.update(quartiles(f"field_{score}", present))
        return columns

    def __value_counts(self, bridge_log: LogEntries) -> dict[str, Any]:
        if not bridge_log:
            return dict.fromkeys(self.VALUE_COLUMNS)
        changed = sum(e["value"] != e["value_no_floor"] for e in bridge_log)
        values = (
            sum(bool(e["fallback"]) for e in bridge_log),
            changed / len(bridge_log),
        )
        return dict(zip(self.VALUE_COLUMNS, values, strict=True))
