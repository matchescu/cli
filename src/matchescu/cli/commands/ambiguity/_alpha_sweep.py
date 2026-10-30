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
    """Profiled alpha grid of one dataset and the alphas chosen from it.

    :ivar rows: one row per alpha: ``dataset``, ``alpha``, ``feasible``,
        ``selected_<criterion>`` per configured criterion, then the profile
        columns.
    :ivar choices: one choice per configured criterion, keyed by criterion
        name.
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
    choices: Mapping[str, AlphaChoice]

    def summary(self) -> dict[str, Any]:
        """Return every criterion's choice and the bridge component columns.

        Bridges do not depend on alpha, so neither do their components; the
        component columns are read from the first row.
        """
        first = self.rows[0] if self.rows else {}
        return {
            **{
                f"alpha_star_{criterion}": choice.alpha
                for criterion, choice in self.choices.items()
            },
            **{
                f"alpha_star_{criterion}_reason": choice.reason
                for criterion, choice in self.choices.items()
            },
            "n_feasible_alphas": next(
                (choice.n_feasible for choice in self.choices.values()), 0
            ),
            **{column: first.get(column) for column in self.COMPONENT_COLUMNS},
        }


class AlphaSweep:
    """Profile every alpha of a generator's grid and choose one per criterion.

    The generator runs at its configured minimum ambiguity, so for a given
    threshold the swept bridges are exactly the bridges ``generate`` writes.
    Every configured criterion sees the same profiled rows.

    :param selectors: one selector per criterion; each chooses an alpha from
        the profiled rows.
    :param options: settings of every profile.
    :raises ValueError: when a selector's criterion needs a profile option
        that `options` switches off; raised before any work is done.
    """

    def __init__(
        self, selectors: Sequence[AlphaSelector], options: ProfileOptions
    ) -> None:
        for selector in selectors:
            selector.check_requirements(options)
        self.__selectors = tuple(selectors)
        self.__options = options

    def run(
        self, generator: "AmbiguityGenerator", id_table: IdTable, dataset: str
    ) -> AlphaSweepResult:
        """Generate, profile and choose among the alphas of one dataset.

        :param generator: generator configured with the alpha grid.
        :param id_table: original records of the dataset.
        :param dataset: label written to the ``dataset`` column.
        :returns: the sweep rows and the per-criterion chosen alphas.
        """
        profiler = VariantProfiler(
            id_table, generator.settings.ambiguity_targets, self.__options
        )
        profiled = [
            {"alpha": alpha, **profiler.profile(variant)}
            for alpha, variant in generator().items()
        ]
        choices = {
            selector.criterion: selector.choose(profiled)
            for selector in self.__selectors
        }
        rows = [self.__row(dataset, row, choices) for row in profiled]
        return AlphaSweepResult(tuple(rows), choices)

    def __row(
        self,
        dataset: str,
        profiled: Mapping[str, Any],
        choices: Mapping[str, AlphaChoice],
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "alpha": profiled["alpha"],
            "feasible": any(
                selector.is_feasible(profiled) for selector in self.__selectors
            ),
            **{
                f"selected_{criterion}": profiled["alpha"] == choice.alpha
                for criterion, choice in choices.items()
            },
            **profiled,
        }
