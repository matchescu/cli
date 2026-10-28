from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from matchescu.reference_store.id_table import IdTable

from ._variant_profiler import VariantProfiler
from .profiling import ProfileOptions
from .selection import AlphaChoice, AlphaSelector

if TYPE_CHECKING:
    from ._generate import AmbiguityGenerator


@dataclass(frozen=True)
class AlphaSweepResult:
    """Profiled alpha grid of one dataset and the alpha chosen from it.

    :ivar rows: one row per alpha: ``dataset``, ``alpha``, ``feasible``,
        ``selected``, then the profile columns.
    :ivar choice: the chosen alpha and why.
    """

    COMPONENT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "bridge_component_count",
        "bridge_component_size_avg",
        "bridge_component_size_stdev",
        "largest_bridge_component_clusters",
        "largest_bridge_component_records",
        "largest_bridge_component_label2",
        "largest_bridge_component_split_load",
    )

    rows: Sequence[Mapping[str, Any]]
    choice: AlphaChoice

    def summary(self) -> dict[str, Any]:
        """Return the choice and the bridge component columns.

        Bridges do not depend on alpha, so neither do their components; the
        component columns are read from the first row.
        """
        first = self.rows[0] if self.rows else {}
        return {
            "alpha_star": self.choice.alpha,
            "alpha_star_reason": self.choice.reason,
            "n_feasible_alphas": self.choice.n_feasible,
            **{column: first.get(column) for column in self.COMPONENT_COLUMNS},
        }


class AlphaSweep:
    """Profile every alpha of a generator's grid and choose one of them.

    The generator runs at its configured minimum ambiguity, so for a given
    threshold the swept bridges are exactly the bridges ``generate`` writes.

    :param selector: chooses the alpha from the profiled rows.
    :param options: settings of every profile.
    :raises ValueError: when the selector's criterion needs a profile option
        that `options` switches off; raised before any work is done.
    """

    def __init__(self, selector: AlphaSelector, options: ProfileOptions) -> None:
        selector.check_requirements(options)
        self.__selector = selector
        self.__options = options

    def run(
        self, generator: "AmbiguityGenerator", id_table: IdTable, dataset: str
    ) -> AlphaSweepResult:
        """Generate, profile and choose among the alphas of one dataset.

        :param generator: generator configured with the alpha grid.
        :param id_table: original records of the dataset.
        :param dataset: label written to the ``dataset`` column.
        :returns: the sweep rows and the chosen alpha.
        """
        profiler = VariantProfiler(
            id_table, generator.settings.ambiguity_targets, self.__options
        )
        profiled = [
            {"alpha": alpha, **profiler.profile(variant)}
            for alpha, variant in generator().items()
        ]
        choice = self.__selector.choose(profiled)
        rows = [self.__row(dataset, row, choice) for row in profiled]
        return AlphaSweepResult(tuple(rows), choice)

    def __row(
        self, dataset: str, profiled: Mapping[str, Any], choice: AlphaChoice
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "alpha": profiled["alpha"],
            "feasible": self.__selector.is_feasible(profiled),
            "selected": profiled["alpha"] == choice.alpha,
            **profiled,
        }
