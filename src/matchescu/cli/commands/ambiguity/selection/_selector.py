from collections.abc import Sequence
from typing import ClassVar

from ..profiling import ProfileOptions
from ._choice import AlphaChoice
from ._criteria import ALPHA, ALPHA_CRITERIA, SweepRow
from ._settings import SelectionSettings


class AlphaSelector:
    """Choose the alpha of a dataset from its profiled sweep rows.

    An alpha is feasible when its content separability lies inside the
    bounds; the configured criterion then picks one feasible alpha.

    :param settings: bounds and criterion of the choice.
    """

    SEPARABILITY: ClassVar[str] = "content_separability"

    def __init__(self, settings: SelectionSettings) -> None:
        self.__bounds = settings.bounds
        self.__criterion = ALPHA_CRITERIA[settings.criterion].from_settings(settings)

    def check_requirements(self, options: ProfileOptions) -> None:
        """Fail early when the profile will lack the criterion's columns.

        :param options: options the sweep rows will be profiled with.
        :raises ValueError: when the criterion needs a profile option that
            is switched off.
        """
        required = self.__criterion.REQUIRED_OPTION
        if required is not None and not getattr(options, required):
            raise ValueError(f"the alpha criterion needs the {required} option")

    def is_feasible(self, row: SweepRow) -> bool:
        """Return whether a sweep row lies inside the separability bounds."""
        return self.__bounds.contains(row.get(self.SEPARABILITY))

    def choose(self, rows: Sequence[SweepRow]) -> AlphaChoice:
        """Choose one alpha among the sweep rows of a dataset.

        :param rows: one profiled row per alpha, each with an ``alpha``
            column.
        :returns: the choice, with the reason when no alpha qualifies.
        """
        feasible = [row for row in rows if self.is_feasible(row)]
        if not feasible:
            return AlphaChoice(None, AlphaChoice.NO_FEASIBLE_ALPHA, 0)
        chosen = self.__criterion(feasible)
        if chosen is None:
            return AlphaChoice(None, self.__criterion.UNAVAILABLE, len(feasible))
        return AlphaChoice(chosen[ALPHA], AlphaChoice.SELECTED, len(feasible))
